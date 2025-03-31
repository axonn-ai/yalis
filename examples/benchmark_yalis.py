from yalis import ModelConfig, InferenceConfig, print_rank0, LLMEngine
from transformers import AutoTokenizer
import torch
import torch.distributed as dist
import argparse
import os
import datetime
import pandas as pd

# needed to work with pytorch 2.3
from torch.profiler import _KinetoProfile

_KinetoProfile._get_distributed_info = lambda self: None

from contextlib import nullcontext

try:
    from mpi4py import MPI
except ImportError:
    pass


def get_prompts(number_of_prompts: int = 16):
    user_prompts = [
        "What is the process for applying to graduate school in the US?",
        "How to bake a cake?",
        "How to drive a car on a freeway?",
        "What are the best practices for time management?",
        "Explain quantum mechanics in simple terms.",
        "How do I write a great resume for a software engineer role?",
        "What are the steps to start a successful business?",
        "How can I improve my public speaking skills?",
        "What are the benefits of a balanced diet?",
        "How to train a dog to follow basic commands?",
        "How do I troubleshoot a slow internet connection?",
        "What is the meaning of life according to philosophy?",
        "How can I learn to play the guitar?",
        "What are the key elements of a good story?",
        "How do I stay motivated while working from home?",
        "What is the easiest way to learn a new language?",
    ]
    user_prompts = user_prompts * (
        (number_of_prompts + len(user_prompts)) // len(user_prompts)
    )
    return user_prompts[:number_of_prompts]


class WandbLogger:
    def __init__(self, config):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        if self.rank == 0:
            try:
                run_name = config["run_name"]
                project = config["project"]
                group = config["group"]
                tags = config["tags"]
                job_type = config["job_type"]

                self.run = wandb.init(
                    project=project,
                    config=config["param_config"],
                    name=run_name,
                    group=group,
                    job_type=job_type,
                    tags=[group] + tags,
                )
            except Exception as e:
                print(f"Error initializing wandb: {e}, Run with logging.")
                self.run = None
                exit(-1)

    def log(self, data, step=None):
        if self.rank == 0:
            self.run.log(data, step=step)
    
    def define_metric(self, **kwargs):
        if self.rank == 0:
            self.run.define_metric(**kwargs)

    def finish(self):
        if self.rank == 0:
            self.run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Enable profiling
    parser.add_argument(
        "--enable_profiling",
        action="store_true",
        help="Enable profiling",
    )

    # Use wandb
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use wandb for metric logging",
    )

    # Target Model ID
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model ID",
    )

    # Tokens to generate
    parser.add_argument(
        "--tokens_to_gen",
        type=int,
        default=256,
        help="Number of tokens to generate",
    )

    # TP
    parser.add_argument(
        "--tp",
        type=int,
        nargs=3,
        default=None,
        help="tensor parallelism dimensions",
    )

    args = parser.parse_args()
    model_id = args.model_id
    enable_profiling = args.enable_profiling
    tokens_to_gen = args.tokens_to_gen
    if args.tp is not None:
        tp_dims = tuple(args.tp)
    else:
        tp_dims = None

    user_prompts = get_prompts(number_of_prompts=64)
    system_prompt = (
        "You are a helpful chatbot. Answer the following question.\n"
    )

    # Tokenizer for encoding the prompt
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    input_prompts = []
    for user_prompt in user_prompts:
        conversation = [
            {
                "role": "system",
                "content": system_prompt,
            },  # not needed for gemma
            {"role": "user", "content": user_prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        input_prompts.append(formatted_prompt)

    # configs
    model_config = ModelConfig(model_name=model_id, precision="bf16")
    inference_config = InferenceConfig(
        batch_size=1,  # len(input_prompts),
        max_length_of_generated_sequences=tokens_to_gen * 2,
        top_p=0.0,
        temperature=0.0,
        tp_dims=tp_dims,
    )

    engine = LLMEngine(
        model_config=model_config, inference_config=inference_config
    )

    if tp_dims is None:
        tp_dims = (dist.get_world_size(), 1, 1)

    if args.use_wandb:
        import wandb

        run_name = (
            os.getenv("JOBID", "0")
            + "_"
            + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        wandb_config = {
            "run_name": run_name,
            "project": "Yalis-Experiments",
            "group": "tioga",
            "job_type": "test",
            "tags": ["flash-attn"],
            "param_config": {
                "model_name": model_id,
                "precision": model_config.precision,
                "tp_dims": tp_dims,
                "top_p": inference_config.top_p,
                "temperature": inference_config.temperature,
            },
        }
        wandb_logger = WandbLogger(wandb_config)
        wandb_logger.define_metric(name="Throughput", summary="mean", step_metric="BatchSize")
        wandb_logger.define_metric(name="TTFT", summary="mean", step_metric="BatchSize")
        wandb_logger.define_metric(name="TBT", summary="mean", step_metric="BatchSize")


    if args.enable_profiling:
        profiler_context = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=5, warmup=2, active=1),
        )
    else:
        profiler_context = nullcontext()

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    #batch_sizes = [1]

    df = None 
    with profiler_context as prof:
        for batch_size in batch_sizes:
            print_rank0(f"Running BatchSize - {batch_size}")

            prompts = input_prompts[:batch_size]
            engine.reset_kv_cache(batch_size)

            dist.barrier()
            torch.cuda.synchronize()
            batch_metrics = []

            for itr in range(10):
                output_tokens, metrics = engine.generate(
                    prompts,
                    report_throughput=True,
                    tokens_to_generate=tokens_to_gen,
                )

                # Skip the first 5 iterations
                if itr >= 5:
                    metrics["NumGpus"] = dist.get_world_size()
                    metrics["TP"] = f"{tp_dims}"
                    batch_metrics.append(metrics)

                if args.enable_profiling:
                    prof.step()
                dist.barrier()

            if df is None:
                df = pd.DataFrame(batch_metrics)
            else:
                df = pd.concat([df, pd.DataFrame(batch_metrics)], ignore_index=True)

            # Log to wandb
            if args.use_wandb:
                for metric in batch_metrics:
                    if metric is not None:
                        wandb_logger.log(data=metric, step=metric["BatchSize"])
                
                wandb_table = wandb.Table(dataframe=df, allow_mixed_types=True)
                wandb_logger.log({"MetricTable": wandb_table})


            # Print the metrics averaged
            if dist.get_rank() == 0:
                df_avg = pd.DataFrame(batch_metrics).groupby(["NumGpus", "TP", "BatchSize"]).mean().reset_index()
                assert len(df_avg) == 1, "There should be only one row in the dataframe"
                avg_dict = df_avg.to_dict(orient="records")
                print_rank0(f"Average metrics for batch size {batch_size}: {avg_dict}")

    output_tokens = output_tokens.cpu()

    # Decode the token IDs into text
    detokenized_text = tokenizer.batch_decode(
        output_tokens, skip_special_tokens=True
    )

    for prompt, output in zip(user_prompts, detokenized_text):
        print_rank0("==========================\n\n")
        print_rank0(f"prompt = {prompt}")
        print_rank0(f"output = {output}")
        print_rank0("==========================\n\n")

    if args.enable_profiling:
        print_rank0(
            prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=10
            )
        )
