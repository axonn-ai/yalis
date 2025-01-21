try:
    from mpi4py import MPI
except ImportError:
    pass
from yalis import ModelConfig, InferenceConfig, print_rank0, LLMEngine, SpecDecLLMEngine
from transformers import AutoTokenizer
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import random
import torch.distributed as dist
import numpy as np
from contextlib import nullcontext

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

from torch.profiler import _KinetoProfile
_KinetoProfile._get_distributed_info = lambda self: None
enable_profiling = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def oom_observer(device, alloc, device_alloc, device_free):
    # snapshot right after an OOM happened
    print('saving allocated state during OOM')
    snapshot = torch.cuda.memory._snapshot()
    dump(snapshot, open('oom_snapshot.pickle', 'wb'))

#import torch._dynamo

#torch._dynamo.config.suppress_errors = True

if __name__ == "__main__":
    # Assuming model and fabric setup functions exist as init_everything() and get_model()
    # target_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # target_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # target_model_id = "meta-llama/Llama-2-70b-chat-hf"
    target_model_id = "meta-llama/Llama-3.1-70B-Instruct"
    # target_model_id = "meta-llama/Llama-3.1-8B-Instruct"
    # draft_model_id = "meta-llama/Llama-2-7b-chat-hf"
    # draft_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # draft_model_id = "lmsys/vicuna-7b-v1.3"
    # draft_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    draft_model_id = "meta-llama/Llama-3.2-1B-Instruct"

    torch.cuda.memory._record_memory_history(max_entries=100000)
    torch._C._cuda_attach_out_of_memory_observer(oom_observer)


    BYTE_TO_GB = 1 / float(1024 * 1024 * 1024)

    print(f"Start - {torch.cuda.memory_allocated() * BYTE_TO_GB}, {torch.cuda.max_memory_allocated() * BYTE_TO_GB}")

    # Initialize prompt and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    user_prompts = [
        "How to bake a cake?",
        #"How to drive a car on a freeway?",
        #"What are the best practices for time management?",
        #"Explain quantum mechanics in simple terms.",
        #"How do I write a great resume for a software engineer role?",
        #"What are the steps to start a successful business?",
        #"How can I improve my public speaking skills?",
        #"What are the benefits of a balanced diet?",
        #"How to train a dog to follow basic commands?",
        #"What is the process for applying to graduate school in the US?",
        #"How do I troubleshoot a slow internet connection?",
        #"What is the meaning of life according to philosophy?",
        #"How can I learn to play the guitar?",
        #"What are the key elements of a good story?",
        #"How do I stay motivated while working from home?",
        #"What is the easiest way to learn a new language?",
    ]

    system_prompt = "You are a helpful chatbot. Answer the following question.\n"

    input_prompts = []
    for user_prompt in user_prompts:
        conversation = [
            {"role": "system", "content": system_prompt},  # not needed for gemma
            {"role": "user", "content": user_prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        input_prompts.append(formatted_prompt)

    tokens_to_gen = 256

    # Configs
    target_model_config = ModelConfig(model_name=target_model_id, precision="bf16")
    draft_model_config = ModelConfig(model_name=draft_model_id, precision="bf16")

    inference_config = InferenceConfig(
        batch_size=1, max_length_of_generated_sequences=512, top_p=0.0, temperature=0.0
    )

    if enable_profiling:
        profiler_context = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=5, warmup=2, active=1),
            record_shapes=True, profile_memory=True, with_stack=True,
        )
    else:
        profiler_context = nullcontext()

    #print_rank0(f"Before Engine Init - {torch.cuda.memory_allocated() * BYTE_TO_GB}, {torch.cuda.max_memory_allocated() * BYTE_TO_GB}")
    with profiler_context as prof:
        engine = SpecDecLLMEngine(
            target_model_config=target_model_config,
            draft_model_config=draft_model_config,
            inference_config=inference_config,
        )
        #print_rank0(f"After Engine Init - {torch.cuda.memory_allocated() * BYTE_TO_GB}, {torch.cuda.max_memory_allocated() * BYTE_TO_GB}")


        for prompt in input_prompts:
            prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids
            tputs = []
            acceptance_rates = []
            print_rank0(f"Before Running the Model - {torch.cuda.memory_allocated() * BYTE_TO_GB}, {torch.cuda.max_memory_allocated() * BYTE_TO_GB}")
            for _ in range(20):
                torch.manual_seed(0)
                random.seed(0)
                np.random.seed(0)

                # print (f"{profiler_context}")
                output_tokens, tput, acceptance_rate = engine.generate(
                    prompt_tokens, tokens_to_gen, gamma=5, report_throughput=True
                )
                tputs.append(tput)
                acceptance_rates.append(acceptance_rate)
                if enable_profiling:
                    prof.step()

                #print_rank0(f"Running Model - {torch.cuda.memory_allocated() * BYTE_TO_GB}, {torch.cuda.max_memory_allocated() * BYTE_TO_GB}")
                dist.barrier()
            

            
            # Average of last 5 throughput values
            avg_tput = sum(tputs[-5:]) / 5
            avg_acceptance_rate = sum(acceptance_rates[-5:]) / 5

            if dist.get_rank() == 0:
                print(f"Average_Throughput:{avg_tput:.2f}| Average_Acceptance_Rate:{avg_acceptance_rate:.2f}")

    try:
        torch.cuda.memory._dump_snapshot(f"./memory_profile_yalis_{dist.get_rank()}.pickle")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")

    torch.cuda.memory._record_memory_history(enabled=None)

    output_tokens = output_tokens.cpu()

    # Decode the token IDs into text
    detokenized_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    for prompt, output in zip(input_prompts, detokenized_text):
        print_rank0(f"prompt = {prompt}")
        print_rank0(f"output = {output}")
        print_rank0("==========================\n\n")

    if enable_profiling:
        # print_rank0(
        #     profiler_context.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        # )
        # Export json trace
        # prof.export_chrome_trace(f"trace_{dist.get_rank()}.json")
        prof.export_memory_timeline(f"memory_{dist.get_rank()}.html")



