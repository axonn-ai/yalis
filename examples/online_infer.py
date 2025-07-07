from yalis import ModelConfig, InferenceConfig, LLMEngine
from transformers import AutoTokenizer
import logging
import torch
import torch.distributed as dist
from contextlib import nullcontext
from flask import Flask, request, jsonify
import time

# Needed to work with pytorch 2.3
from torch.profiler import _KinetoProfile

_KinetoProfile._get_distributed_info = lambda self: None


# Refer to this for API Reference:
# https://platform.openai.com/docs/api-reference/chat

# Configure python logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)

# Keep structure of output
app.json.sort_keys = False

# Global configs
enable_profiling = False
if enable_profiling:
    profiler_context = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=5, warmup=2, active=1),
    )
else:
    profiler_context = nullcontext()
system_prompt = "You are a helpful chatbot. Answer the following question.\n"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # NOTE: HARD SET
global_tokenizer = AutoTokenizer.from_pretrained(model_id)

model_config = ModelConfig(model_name=model_id, precision="bf16")
inference_config = InferenceConfig(
    batch_size=1,  # NOTE: HARD SET
    max_length_of_generated_sequences=1024,
    top_p=0.80,
    temperature=1.0,
    tp_dims=None,
)
global_engine = LLMEngine(
    model_config=model_config, inference_config=inference_config
)


@app.route("/v1/completions", methods=["POST"])
def infer_endpoint():
    while True:
        logging.info(f"Process rank {dist.get_rank()}.")

        # Rank 0, only, receives the data
        if dist.get_rank() == 0:
            data = request.json
            logging.info(f"Request received on rank {dist.get_rank()}.")
        else:
            data = None

        # Rank 0 distributes the data to other ranks
        data_list = [data]
        # Collective broadcast call
        # src=0 ensures that only rank 0's data values are being stored
        dist.broadcast_object_list(data_list, src=0)
        data = data_list[0]

        logging.info("All processes have data.")

        # Parsing data from request:

        if "prompt" not in data:
            # Required
            return jsonify({"error": "No prompt provided."}), 400
        else:
            user_prompts = data["prompt"]
            # Wrap single prompt in a list for parsing later
            if not isinstance(user_prompts, list):
                user_prompts = [user_prompts]

        if "model" not in data:
            # Required
            return jsonify({"error": "No model provided."}), 400
        else:
            model_id = data["model"]

        if data["model"] != model_id:
            # NOTE: only support "Meta-Llama-3-8B-Instruct" model for now
            return (
                jsonify(
                    {
                        "error": f"Model {data['model']} not supported, use {model_id} instead."  # noqa: E501
                    }
                ),
                400,
            )

        # Default value
        tokens_to_gen = 512
        if "max_completion_tokens" in data:
            tokens_to_gen = data["max_completion_tokens"]

        # Default value
        n = 1
        if "n" in data and data["n"] != 1:
            return (
                jsonify(
                    {"error": "YALIS only supports n=1 completion choices."}
                ),
                400,
            )

        logging.info(f"Number of prompts: {len(user_prompts)}.")

        input_prompts = []
        for user_prompt in user_prompts:
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            formatted_prompt = global_tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            # NOTE: Future support for n:
            input_prompts.extend([formatted_prompt] * n)

        logging.info("Starting time for tok gen.")
        req_start_time = time.time()

        with profiler_context as prof:
            output_tokens, _ = global_engine.generate(
                input_prompts,
                report_throughput=True,
                tokens_to_generate=tokens_to_gen,
            )
            if enable_profiling:
                prof.step()

        logging.info("Ending time after tok gen.")
        req_end_time = time.time()
        logging.info(
            f"Time taken for tok gen: {req_end_time - req_start_time:.2f} s."
        )

        output_tokens = output_tokens.cpu()
        detokenized_text = global_tokenizer.batch_decode(
            output_tokens, skip_special_tokens=True
        )

        # Rank 0 process sends back response to user
        if dist.get_rank() == 0:
            # Choices json object:
            choices = []
            for i in range(0, len(input_prompts)):
                choices_obj = {
                    "index": i,
                    "message": detokenized_text[i],
                    "logprobs": None,
                    "finish_reason": "length",
                }
                choices.append(choices_obj)

            # Usage json object:
            prompt_tokens = sum(
                len(global_tokenizer(prompt)["input_ids"])
                for prompt in user_prompts
            )
            completion_tokens = sum(
                len(global_tokenizer(text)["input_ids"])
                for text in detokenized_text
            )
            total_tokens = prompt_tokens + completion_tokens
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "prompt_tokens_details": {
                    "cached_tokens": 0,
                    "audio_tokens": 0,
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            }

            # Chat completion object: refer to the link @ top of the file
            response = {
                "id": f"yalis-{int(time.time()*1e3)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": choices,
                "usage": usage,
                "service_tier": "default",
                "system_fingerprint": "TEMP_FINGERPRINT",
            }
            logging.info("Json being sent back to client.")
            return jsonify(response)


if __name__ == "__main__":
    # Rank 0 is the only one that has external access to requests from user
    if dist.get_rank() == 0:
        logging.info(
            "Starting Flask server on rank 0. Listening on port 5000..."
        )
        app.run(host="0.0.0.0", port=5000)
    else:
        infer_endpoint()
