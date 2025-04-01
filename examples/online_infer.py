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

# Global configs
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
global_tokenizer = AutoTokenizer.from_pretrained(model_id)

model_config = ModelConfig(model_name=model_id, precision="bf16")
inference_config = InferenceConfig(
    batch_size=2, # HARD SET FOR NOW
    max_length_of_generated_sequences=1024,
    top_p=0.80,
    temperature=1.0,
    tp_dims=(2,1,1)
)
global_engine = LLMEngine(model_config=model_config, inference_config=inference_config)

@app.route("/v1/completions", methods=["POST"])
def infer_endpoint():
    print(f"==> Hi, I am process rank {dist.get_rank()} :)")
    # Only rank 0 gets the data, even though all other processes are triggered
    # Assumption: Data is not chunked, all the data is sent because yalis deals 
    # with distribution of compute

    if dist.get_rank() == 0:
        data = request.json
        print("==> Request received on rank 0:")
        print(data)
    else:
        data = None

    # Collective broadcast call
    if dist.is_initialized():
        data_list = [data]
        # src=0 ensures that only rank 0's data values are being stored
        dist.broadcast_object_list(data_list, src=0)
        data = data_list[0]

    print("==> All processes should now have data!")
    print(f"==> Process rank {dist.get_rank()}'s data: {data}")
    
    # Arg checking in data
    if "prompt" not in data:
        return jsonify({"error": "No prompt provided"}), 400

    if "model" not in data:
        return jsonify({"error": "No model provided"}), 400
    
    if data["model"] != "meta-llama/Meta-Llama-3-8B-Instruct":
        return jsonify({"error": "Not meta-llama/Meta-Llama-3-8B-Instruct model"}), 400
    
    n = 1
    if "n" in data:
        n = data["n"]

    user_prompts = data["prompt"]
    if not isinstance(user_prompts, list):
        user_prompts = [user_prompts]
    model_id = data["model"]

    print(f"Number of prompts = {len(user_prompts)}")

    input_prompts = []
    for user_prompt in user_prompts:
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        formatted_prompt = global_tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        # Added support for "n"
        input_prompts.extend([formatted_prompt] * n)
    
    print("==> Starting time for tok gen")
    req_start_time = time.time()

    with profiler_context as prof:
        for _ in range(10):
            output_tokens = global_engine.generate(
                input_prompts, report_throughput=True, tokens_to_generate=tokens_to_gen
            )
            if enable_profiling:
                prof.step()
            print("==> Checking which process hits this:")
            print(dist.get_rank())
            dist.barrier()
    
    print("==> Ending time after tok gen")
    req_end_time = time.time()
    print(f"==> Time taken: {req_end_time - req_start_time:.2f} seconds")

    output_tokens = output_tokens.cpu()
    detokenized_text = global_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    
    print("==> Detokenized text done.")

    if dist.get_rank() == 0:
        print(f"==> Rank {dist.get_rank()} process here, done work now sending to client :)")
        choices = []
        for i in range(0, len(input_prompts)):
            json_obj = {
                    "text": detokenized_text[i],
                    "index": i,
                    "logprobs": None,
                    "finish_reason": "TEMP"
                }
            choices.append(json_obj)
        
        prompt_tokens = sum(len(global_tokenizer(prompt)["input_ids"]) for prompt in user_prompts)
        completion_tokens = sum(len(global_tokenizer(text)["input_ids"]) for text in detokenized_text)
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
    else:
        print(f"==> Rank {dist.get_rank()} process here, done work now shutting down.")
        infer_endpoint()


if __name__ == "__main__":
    # Rank 0 process is the only one that has external access
    # Other ranks aren't accessible but still get triggered upon request
    if dist.get_rank() == 0:
        print("Starting Flask server on rank 0 process. Listening on port 5000...")
        app.run(host="0.0.0.0", port=5000)
    else:
        infer_endpoint()
