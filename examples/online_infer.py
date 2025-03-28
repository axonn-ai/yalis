try:
    from mpi4py import MPI
except ImportError:
    pass

from yalis import ModelConfig, InferenceConfig, print_rank0, LLMEngine
from transformers import AutoTokenizer
import torch
import torch.distributed as dist

# needed to work with pytorch 2.3
from torch.profiler import _KinetoProfile
_KinetoProfile._get_distributed_info = lambda self: None

from contextlib import nullcontext
from flask import Flask, request, jsonify
import time

app = Flask(__name__)
# Keep structure for output
app.json.sort_keys = False

# Global configs
enable_profiling = False
tokens_to_gen = 512
system_prompt = "You are a helpful chatbot. Answer the following question.\n"
if enable_profiling:
    profiler_context = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=5, warmup=2, active=1),
    )
else:
    profiler_context = nullcontext()

# Global caches
global_tokenizers = {}
global_engines = {}

@app.route("/v1/completions", methods=["POST"])
def infer_endpoint():
    data = request.json
    print("==> Request received: ")
    print(data)

    # Arg checking in data
    if "prompt" not in data:
        return jsonify({"error": "No prompt provided"}), 400

    if "model" not in data:
        return jsonify({"error": "No model provided"}), 400
    
    n = 1
    if "n" in data:
        n = data["n"]

    user_prompts = data["prompt"]
    if not isinstance(user_prompts, list):
        user_prompts = [user_prompts]
    model_id = data["model"]

    print(f"Number of prompts = {len(user_prompts)}")

    # Check cache
    if model_id not in global_tokenizers:
        # Tokenizer for encoding the prompt
        global_tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)
    # Get value from cache
    tokenizer = global_tokenizers[model_id]

    input_prompts = []
    for user_prompt in user_prompts:
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        # Added support for "n"
        input_prompts.extend([formatted_prompt] * n)

    # Check cache
    cache_key = (model_id, len(input_prompts))
    if cache_key not in global_engines:
        # Right now set for only 1 batch size (1 prompt)
        model_config = ModelConfig(model_name=model_id, precision="bf16")
        inference_config = InferenceConfig(
            batch_size=len(input_prompts),
            max_length_of_generated_sequences=1024,
            top_p=0.80,
            temperature=1.0,
            tp_dims=(1,1,1)
        )
        global_engines[cache_key] = LLMEngine(model_config=model_config, inference_config=inference_config)
    # Get value from cache
    engine = global_engines[cache_key]
    
    with profiler_context as prof:
        for _ in range(10):
            output_tokens = engine.generate(
                input_prompts, report_throughput=True, tokens_to_generate=tokens_to_gen
            )
            if enable_profiling:
                prof.step()
            dist.barrier()
    
    output_tokens = output_tokens.cpu()
    detokenized_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    
    print("==> Detokenized text done.")

    choices = []
    for i in range(0, len(input_prompts)):
        json_obj = {
                "text": detokenized_text[i],
                "index": i,
                "logprobs": None,
                "finish_reason": "TEMP"
            }
        choices.append(json_obj)
    
    prompt_tokens = sum(len(tokenizer(prompt)["input_ids"]) for prompt in user_prompts)
    completion_tokens = sum(len(tokenizer(text)["input_ids"]) for text in detokenized_text)
    total_tokens = prompt_tokens + completion_tokens
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }

    response = {
        "id": "TEMP",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": choices,
        "usage": usage
    }

    print("==> Json being sent back to client :)")
    return jsonify(response)

if __name__ == "__main__":
    # Only rank 0 runs the server
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0:
        # print_rank0 erroring
        # print_rank0("Starting Flask server on rank 0. Listening on port 5000...")
        print("Starting Flask server on rank 0. Listening on port 5000...")
        app.run(host="0.0.0.0", port=5000)
