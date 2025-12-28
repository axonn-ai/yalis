#include <torch/extension.h>
using at::Half;

// forward declare your launcher as taking raw pointers
#if defined(__GNUC__) || defined(__clang__)
#define THRESH_ATTN_WEAK __attribute__((weak))
#else
#define THRESH_ATTN_WEAK
#endif

extern "C" void decode_attn_cuda_launcher(
    const void* Q, const void* K, const void* V,
    const float* Bias, void* Out,
    const float* Threshold, float scale,
    int B, int H, int T, int D
) THRESH_ATTN_WEAK;

extern "C" void decode_attn_cuda_launcher_gmem(
    const void* Q, const void* K, const void* V,
    const float* Bias, void* Out,
    const float* Threshold, float scale,
    void* logits_gmem,
    int B, int H, int T, int D
) THRESH_ATTN_WEAK;

static torch::Tensor decode_attn_fwd_impl(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor attn_bias,
    torch::Tensor thresholds,
    float sm_scale,
    bool use_gmem
) {
  TORCH_CHECK(Q.dtype()==torch::kHalf && Q.is_contiguous(), "Q must be fp16");

  auto B = Q.size(0), H = Q.size(1), D = Q.size(3);
  auto T = K.size(2);
  auto out = torch::empty({B,H,1,D}, Q.options());

  auto Qf = Q.view({B*H, D});
  auto Kf = K.view({B*H, T, D});
  auto Vf = V.view({B*H, T, D});
  auto Bf = attn_bias.view({B*H, T});
  auto Of = out.view({B*H, D});

  if (use_gmem) {
    TORCH_CHECK(decode_attn_cuda_launcher_gmem != nullptr,
                "gmem kernel requested but decode_attn_cuda_launcher_gmem is not linked. "
                "Compile with a gmem kernel source (e.g. thresh_attn_cuda_tiled_fused_v1.cu).");
    auto logits_gmem = torch::empty({B * H, T}, Q.options());
    decode_attn_cuda_launcher_gmem(
      /*Q=*/   reinterpret_cast<const void*>(Qf.data_ptr<Half>()),
      /*K=*/   reinterpret_cast<const void*>(Kf.data_ptr<Half>()),
      /*V=*/   reinterpret_cast<const void*>(Vf.data_ptr<Half>()),
      /*Bias=*/Bf.data_ptr<float>(),
      /*Out=*/ reinterpret_cast<void*>(   Of.data_ptr<Half>()),
      /*Threshold=*/thresholds.data_ptr<float>(),
      sm_scale,
      reinterpret_cast<void*>(logits_gmem.data_ptr<Half>()),
      B, H, T, D
    );
  } else {
    TORCH_CHECK(decode_attn_cuda_launcher != nullptr,
                "non-gmem kernel requested but decode_attn_cuda_launcher is not linked. "
                "Compile with a non-gmem kernel source (e.g. thresh_attn_cuda_bitmask_v1.cu).");
    decode_attn_cuda_launcher(
      /*Q=*/   reinterpret_cast<const void*>(Qf.data_ptr<Half>()),
      /*K=*/   reinterpret_cast<const void*>(Kf.data_ptr<Half>()),
      /*V=*/   reinterpret_cast<const void*>(Vf.data_ptr<Half>()),
      /*Bias=*/Bf.data_ptr<float>(),
      /*Out=*/ reinterpret_cast<void*>(   Of.data_ptr<Half>()),
      /*Threshold=*/thresholds.data_ptr<float>(),
      sm_scale, B, H, T, D
    );
  }

  return out;
}

torch::Tensor decode_attn_fwd_gmem(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor attn_bias,
    torch::Tensor thresholds,
    float sm_scale
) {
  return decode_attn_fwd_impl(Q, K, V, attn_bias, thresholds, sm_scale, true);
}
torch::Tensor decode_attn_fwd(
    torch::Tensor Q,            // [B,H,1,D], fp16
    torch::Tensor K,            // [B,H,T,D], fp16
    torch::Tensor V,            // [B,H,T,D], fp16
    torch::Tensor attn_bias,    // [B,H,1,T], fp32
    torch::Tensor thresholds,   // [B,H],         fp32
    float sm_scale
) {
  return decode_attn_fwd_impl(Q, K, V, attn_bias, thresholds, sm_scale, false);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_attn_fwd_gmem", &decode_attn_fwd_gmem, "thresh_attn w/ gmem fp16");
  m.def("decode_attn_fwd", &decode_attn_fwd, "thresh_attn fp16");
}
