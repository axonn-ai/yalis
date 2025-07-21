#include <torch/extension.h>
using at::Half;

// forward‐declare your launcher as taking raw pointers
extern "C" void decode_attn_cuda_launcher(
    const void* Q, const void* K, const void* V,
    const float* Bias, void* Out,
    const float* Threshold, float scale,
    int B, int H, int T, int D
);

torch::Tensor decode_attn_fwd(
    torch::Tensor Q,            // [B,H,1,D], fp16
    torch::Tensor K,            // [B,H,T,D], fp16
    torch::Tensor V,            // [B,H,T,D], fp16
    torch::Tensor attn_bias,    // [B,H,1,T], fp32
    torch::Tensor thresholds,   // [B,H],         fp32
    float sm_scale
) {
  TORCH_CHECK(Q.dtype()==torch::kHalf && Q.is_contiguous(), "Q must be fp16");
  // … same checks for K, V, bias, thresholds …

  auto B = Q.size(0), H = Q.size(1), D = Q.size(3);
  auto T = K.size(2);
  auto out = torch::empty({B,H,1,D}, Q.options());

  // flatten away the size-1 dims
  auto Qf = Q.view({B*H, D});
  auto Kf = K.view({B*H, T, D});
  auto Vf = V.view({B*H, T, D});
  auto Bf = attn_bias.view({B*H, T});
  auto Of = out.view({B*H, D});

  // call launcher, passing raw pointers
  decode_attn_cuda_launcher(
    /*Q=*/   reinterpret_cast<const void*>(Qf.data_ptr<Half>()),
    /*K=*/   reinterpret_cast<const void*>(Kf.data_ptr<Half>()),
    /*V=*/   reinterpret_cast<const void*>(Vf.data_ptr<Half>()),
    /*Bias=*/Bf.data_ptr<float>(),
    /*Out=*/ reinterpret_cast<void*>(   Of.data_ptr<Half>()),
    /*Threshold=*/thresholds.data_ptr<float>(),
    sm_scale, B, H, T, D
  );
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_attn_fwd", &decode_attn_fwd, "thresh_attn two-pass WMMA fp16");
}