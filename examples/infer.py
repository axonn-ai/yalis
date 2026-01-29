
from yalis import ModelConfig, InferenceConfig, print_rank0, LLMEngine
from transformers import AutoTokenizer
import torch
import torch.distributed as dist

# needed to work with pytorch 2.3
from torch.profiler import _KinetoProfile
_KinetoProfile._get_distributed_info = lambda self: None

from contextlib import nullcontext

try:
    from mpi4py import MPI
except ImportError:
    pass

if __name__ == "__main__":
    # Model ID from Hugging Face
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    # user_prompts = [
    #     "How to bake a cake?",
    #     "How to drive a car on a freeway?",
    #     "What are the best practices for time management?",
    #     "Explain quantum mechanics in simple terms.",
    #     "How do I write a great resume for a software engineer role?",
    #     "What are the steps to start a successful business?",
    #     "How can I improve my public speaking skills?",
    #     "What are the benefits of a balanced diet?",
    #     "How to train a dog to follow basic commands?",
    #     "What is the process for applying to graduate school in the US?",
    #     "How do I troubleshoot a slow internet connection?",
    #     "What is the meaning of life according to philosophy?",
    #     "How can I learn to play the guitar?",
    #     "What are the key elements of a good story?",
    #     "How do I stay motivated while working from home?",
    #     "What is the easiest way to learn a new language?",
    # ]
    
    # # take 16 prompts from this dataset
    # user_prompts = user_prompts[:16]

    user_prompts = [
        (
            "A city plans to turn a 7-mile rail corridor into a greenway. There is "
            "disagreement over bus lanes vs park and bike focus, and businesses fear "
            "construction disruption. Write a 4000 word essay comparing options, tradeoffs, "
            "a phased plan, and a recommendation with risk mitigation."
        ),
        (
            "Advise a national public health agency on antibiotic resistance. Cover history, "
            "mechanisms, agriculture, hospital policy, supply chain limits, and drug incentives. "
            "Include 20-year best and worst case scenarios and a public communication plan. "
            "Write a 4000 word essay as a policy blueprint."
        ),
        (
            "Write a guide for a 100 MW data center in a hot, water-scarce region. Cover cooling, "
            "power procurement, grid risks, backup generation, fire suppression, staffing, network "
            "design, and sustainability reporting. Include a timeline, budget assumptions, and a "
            "risk register. Write a 4000 word essay with an executive summary."
        ),
        (
            "Analyze cinematic science fiction from the 1950s to today. Connect political climates "
            "to themes and visuals, discuss at least five directors, compare American and non-American "
            "traditions, and explain how effects tech changed storytelling. Write a 4000 word essay "
            "with a conclusion on where the genre goes next."
        ),
    ]
    user_prompts = user_prompts[:4]
    print(f"Number of prompts = {len(user_prompts)}")

    system_prompt = "You are a helpful chatbot. Answer the following question.\n"

    # profile the run or not
    enable_profiling = False

    # Tokenizer for encoding the prompt
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    input_prompts = []
    for user_prompt in user_prompts:
        if getattr(tokenizer, "chat_template", None):
            conversation = [
                {"role": "system", "content": system_prompt},  # not needed for gemma
                {"role": "user", "content": user_prompt},
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
        else:
            formatted_prompt = f"{system_prompt}{user_prompt}\n"
        input_prompts.append(formatted_prompt)

    # Number of tokens to generate
    tokens_to_gen = 2048

    # configs
    model_config = ModelConfig(model_name=model_id, precision="fp16")
    inference_config = InferenceConfig(batch_size=len(input_prompts),
                                       max_length_of_generated_sequences=tokens_to_gen,
                                       top_p=0.80,
                                       temperature=1.0, 
                                       tp_dims=None,
                                       attention_backend="flash",
                                       threshold_percentile=0.5,
                                       num_warmup_steps=16,
                                       use_paged_kv_caching=False)


    engine = LLMEngine(model_config=model_config, inference_config=inference_config)

    if enable_profiling:
        profiler_context = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=5, warmup=2, active=1),
        )
    else:
        profiler_context = nullcontext()

    with profiler_context as prof:
        for iter in range(2):
            output_tokens, metrics, _ = engine.generate(
                input_prompts, report_throughput=True, tokens_to_generate=tokens_to_gen
            )
            if enable_profiling:
                prof.step()
            dist.barrier()

    output_tokens = output_tokens.cpu()

    # Decode the token IDs into text
    detokenized_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    for prompt, output in zip(user_prompts, detokenized_text):
        print_rank0("==========================\n\n")
        print_rank0(f"prompt = {prompt}")
        print_rank0(f"output = {output}")
        print_rank0("==========================\n\n")
        
             
    if enable_profiling:
        print_rank0(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        )
        
