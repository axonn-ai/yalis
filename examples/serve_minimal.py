import uvicorn
import os
from yalis import ModelConfig, InferenceConfig
from yalis.serving.server import create_app


def main():
    host = os.getenv("YALIS_SERVE_HOST", "0.0.0.0")
    port = int(os.getenv("YALIS_SERVE_PORT", "8000"))
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    max_batch_size = 8
    model_cfg = ModelConfig(model_name=model_id, precision="bf16", disable_tp=True)
    infer_cfg = InferenceConfig(
        max_batch_size=max_batch_size,
        max_length_of_generated_sequences=1024,
        top_p=0.0,
        temperature=0.0,
        tp_dims=None,
        attention_backend="flash",
        use_paged_kv_caching=True,
    )
    app = create_app(model_cfg, infer_cfg)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()


