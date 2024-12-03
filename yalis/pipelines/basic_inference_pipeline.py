import torch
from .base_pipline import BasePipeline, prefill, generate
from yalis import print_rank0
from typing import Optional
import time

class BasicInferencePipeline(BasePipeline):
    def __init__(self, 
                 model, 
                 tokenizer,
                 dtype,
                 device,):
        pass
        super().__init__(device)
        self.model = model
        self.dtype = dtype
        self.tokenizer = tokenizer


    def setup(self, 
              batch_size: Optional[int] = 1):
        self.model.set_kv_cache(batch_size=batch_size, device=self.device, dtype=self.dtype)



    def run(self, 
            prompt, 
            tokens_to_gen,
            profile: Optional[bool] = True):
        
        # Caching the attributes to local variables improves performance
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze().to(self.device)

        if profile:
            num_trials = 10
            num_warumups = 5
        else:
            num_trials = 1

        # Generation loop
        # Using Cuda events instead of time.time(). Synchronizing was important as adding that gave a slighlty lower performance
        #print("\nStarting token generation:")
        for TRIAL in range(10):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output_tokens = []
            with torch.no_grad(), torch.autocast(self.device, dtype=self.dtype, cache_enabled=False):
                for step in range(tokens_to_gen):
                    if step == 0: # prefill
                        next_token = prefill(self.model, prompt_tokens)
                        input_pos = torch.tensor([prompt_tokens.size(0)], device=self.device, dtype=torch.int64)
                        tokens = next_token.clone()
                    else:
                        with torch.nn.attention.sdpa_kernel(
                                    torch.nn.attention.SDPBackend.MATH):
                            next_token = generate(self.model, tokens, input_pos)
                            input_pos.add_(1)
                            tokens.copy_(next_token)
                    # Append token to output and log details
                    output_tokens.append(next_token.clone())
            end.record()
            torch.cuda.synchronize()
            time_taken = start.elapsed_time(end) / 1000

            if profile and TRIAL >= num_warumups:
                tokens_per_second = len(output_tokens) / time_taken
                print_rank0(f"[Iter:{TRIAL}]Output {tokens_per_second} tok/s") 

        generated_text = self.tokenizer.decode([x.item() for x in output_tokens])
        print_rank0("-" * 40)
        print_rank0("\nGenerated text:\n" + "-" * 40)
        print_rank0(generated_text)

        return output_tokens
