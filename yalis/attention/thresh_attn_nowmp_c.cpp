#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

using at::Half;

extern "C" void nowmp_attn_cuda_launcher_gmem(
    const void* Qp, const void* Kp, const void* Vp,
    const float* Bias, void* Outp,
    float* alpha, float* b,
    float* kept_cum, float* total_cum,
    float* lr_a, float* lr_b,
    const int* token_counter,
    float r_target,
    float log_a_min,
    float log_a_max,
    float b_min,
    float b_max,
    int base_cutoff,
    int base_step,
    float anl_a,
    float anl_b,
    float scale,
    void* logits_gmem,
    float* keep_counts,
    int B, int H, int T, int D,
    cudaStream_t stream
);

static std::vector<torch::Tensor> nowmp_attn_fwd(
    torch::Tensor Q,            // [B, H, 1, D], fp16
    torch::Tensor K,            // [B, H, T, D], fp16
    torch::Tensor V,            // [B, H, T, D], fp16
    torch::Tensor attn_bias,    // [B, H, 1, T], fp32
    torch::Tensor token_counter,// [B], int32
    torch::Tensor alpha,        // [B, H], fp32
    torch::Tensor b,            // [B, H], fp32
    torch::Tensor kept_cum,     // [B, H], fp32
    torch::Tensor total_cum,    // [B, H], fp32
    torch::Tensor lr_a,         // [B, H], fp32
    torch::Tensor lr_b,         // [B, H], fp32
    double r_target,
    int64_t base_cutoff,
    int64_t base_step,
    double anl_a,
    double anl_b,
    double a_min,
    double a_max,
    double b_min,
    double b_max,
    double sm_scale
) {
  TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
  TORCH_CHECK(K.is_cuda(), "K must be CUDA");
  TORCH_CHECK(V.is_cuda(), "V must be CUDA");
  TORCH_CHECK(attn_bias.is_cuda(), "attn_bias must be CUDA");
  TORCH_CHECK(token_counter.is_cuda(), "token_counter must be CUDA");

  TORCH_CHECK(Q.dtype() == torch::kHalf, "Q must be fp16");
  TORCH_CHECK(K.dtype() == torch::kHalf, "K must be fp16");
  TORCH_CHECK(V.dtype() == torch::kHalf, "V must be fp16");
  TORCH_CHECK(attn_bias.dtype() == torch::kFloat, "attn_bias must be fp32");
  TORCH_CHECK(token_counter.dtype() == torch::kInt, "token_counter must be int32");

  TORCH_CHECK(alpha.dtype() == torch::kFloat, "alpha must be fp32");
  TORCH_CHECK(b.dtype() == torch::kFloat, "b must be fp32");
  TORCH_CHECK(kept_cum.dtype() == torch::kFloat, "kept_cum must be fp32");
  TORCH_CHECK(total_cum.dtype() == torch::kFloat, "total_cum must be fp32");
  TORCH_CHECK(lr_a.dtype() == torch::kFloat, "lr_a must be fp32");
  TORCH_CHECK(lr_b.dtype() == torch::kFloat, "lr_b must be fp32");

  TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
  TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
  TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
  TORCH_CHECK(attn_bias.is_contiguous(), "attn_bias must be contiguous");
  TORCH_CHECK(token_counter.is_contiguous(), "token_counter must be contiguous");
  TORCH_CHECK(alpha.is_contiguous(), "alpha must be contiguous");
  TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
  TORCH_CHECK(kept_cum.is_contiguous(), "kept_cum must be contiguous");
  TORCH_CHECK(total_cum.is_contiguous(), "total_cum must be contiguous");
  TORCH_CHECK(lr_a.is_contiguous(), "lr_a must be contiguous");
  TORCH_CHECK(lr_b.is_contiguous(), "lr_b must be contiguous");

  TORCH_CHECK(Q.dim() == 4, "Q must be [B, H, 1, D]");
  TORCH_CHECK(K.dim() == 4, "K must be [B, H, T, D]");
  TORCH_CHECK(V.dim() == 4, "V must be [B, H, T, D]");
  TORCH_CHECK(attn_bias.dim() == 4, "attn_bias must be [B, H, 1, T]");
  TORCH_CHECK(token_counter.dim() == 1, "token_counter must be [B]");
  TORCH_CHECK(alpha.dim() == 2, "alpha must be [B, H]");
  TORCH_CHECK(b.dim() == 2, "b must be [B, H]");

  int64_t B = Q.size(0);
  int64_t H = Q.size(1);
  int64_t D = Q.size(3);
  int64_t T = K.size(2);

  TORCH_CHECK(Q.size(2) == 1, "Q must have sequence length 1");
  TORCH_CHECK(K.size(0) == B && K.size(1) == H && K.size(3) == D, "K shape mismatch");
  TORCH_CHECK(V.size(0) == B && V.size(1) == H && V.size(2) == T && V.size(3) == D, "V shape mismatch");
  TORCH_CHECK(attn_bias.size(0) == B && attn_bias.size(1) == H && attn_bias.size(2) == 1 && attn_bias.size(3) == T, "attn_bias shape mismatch");
  TORCH_CHECK(alpha.size(0) == B && alpha.size(1) == H, "alpha shape mismatch");
  TORCH_CHECK(b.size(0) == B && b.size(1) == H, "b shape mismatch");
  TORCH_CHECK(kept_cum.size(0) == B && kept_cum.size(1) == H, "kept_cum shape mismatch");
  TORCH_CHECK(total_cum.size(0) == B && total_cum.size(1) == H, "total_cum shape mismatch");
  TORCH_CHECK(lr_a.size(0) == B && lr_a.size(1) == H, "lr_a shape mismatch");
  TORCH_CHECK(lr_b.size(0) == B && lr_b.size(1) == H, "lr_b shape mismatch");
  TORCH_CHECK(token_counter.size(0) == B, "token_counter must be [B]");
  TORCH_CHECK(D % 16 == 0 && D <= 128, "head_dim must be <=128 and divisible by 16");

  auto out = torch::empty({B, H, 1, D}, Q.options());
  auto keep_counts = torch::empty({B, H}, Q.options().dtype(torch::kFloat));
  auto logits_gmem = torch::empty({B * H, T}, Q.options());

  float log_a_min = std::log(a_min);
  float log_a_max = std::log(a_max);

  auto Qf = Q.view({B * H, D});
  auto Kf = K.view({B * H, T, D});
  auto Vf = V.view({B * H, T, D});
  auto Bf = attn_bias.view({B * H, T});
  auto Of = out.view({B * H, D});

  auto stream = at::cuda::getCurrentCUDAStream();
  nowmp_attn_cuda_launcher_gmem(
      reinterpret_cast<const void*>(Qf.data_ptr<Half>()),
      reinterpret_cast<const void*>(Kf.data_ptr<Half>()),
      reinterpret_cast<const void*>(Vf.data_ptr<Half>()),
      Bf.data_ptr<float>(),
      reinterpret_cast<void*>(Of.data_ptr<Half>()),
      alpha.data_ptr<float>(),
      b.data_ptr<float>(),
      kept_cum.data_ptr<float>(),
      total_cum.data_ptr<float>(),
      lr_a.data_ptr<float>(),
      lr_b.data_ptr<float>(),
      token_counter.data_ptr<int>(),
      static_cast<float>(r_target),
      log_a_min,
      log_a_max,
      static_cast<float>(b_min),
      static_cast<float>(b_max),
      static_cast<int>(base_cutoff),
      static_cast<int>(base_step),
      static_cast<float>(anl_a),
      static_cast<float>(anl_b),
      static_cast<float>(sm_scale),
      reinterpret_cast<void*>(logits_gmem.data_ptr<Half>()),
      keep_counts.data_ptr<float>(),
      static_cast<int>(B),
      static_cast<int>(H),
      static_cast<int>(T),
      static_cast<int>(D),
      stream.stream()
  );

  return {out, keep_counts};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nowmp_attn_fwd", &nowmp_attn_fwd, "nowmp attention forward (fused logits+threshold)");
}
