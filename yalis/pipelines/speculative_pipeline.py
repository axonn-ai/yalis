import torch
from .base_pipline import BasePipeline, prefill, generate, verify
from yalis.external.rejection_sampler import RejectionSampler
from yalis import print_rank0
from typing import Optional
import time

class SpeculativeDecodingPipeline(BasePipeline):
    def __init__(self, 
                 target_model, 
                 draft_model,
                 tokenizer,
                 dtype,
                 device,):
        pass
        super().__init__(device)
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.sampler = RejectionSampler()
        #self.sampler = torch.compile(self.sampler, fullgraph=True)


    def setup(self, 
              batch_size: Optional[int] = 1):
        self.target_model.set_kv_cache(batch_size=batch_size, device=self.device, dtype=self.dtype)
        self.draft_model.set_kv_cache(batch_size=batch_size, device=self.device, dtype=self.dtype)

    def run(self, 
            prompt, 
            tokens_to_gen,
            gamma,
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
                generated_tokens = 0
                while generated_tokens < tokens_to_gen:
                    if generated_tokens == 0: # Prefill
                        # Only the target model is used for prefilling
                        next_token = prefill(self.target_model, prompt_tokens)
                        target_input_pos = torch.tensor([prompt_tokens.size(0)], device=self.device, dtype=torch.int64)
                        tokens = next_token.clone()
                        generated_tokens += 1
                        output_tokens.append(next_token.clone())

                        _ = prefill(self.draft_model, prompt_tokens)
                    else: # Decode
                        with torch.nn.attention.sdpa_kernel(
                                    torch.nn.attention.SDPBackend.MATH):
                            # Draft tokens
                            draft_input_pos = target_input_pos.clone()
                            draft_tokens = tokens.clone()

                            draft_output_tokens = []
                            draft_output_tokens.append(draft_tokens.clone())
                            draft_probs = []
                            for draft_step in range(gamma):
                                # Get next draft token and its probabilities
                                next_token, probs = generate(self.draft_model, draft_tokens, draft_input_pos, get_probs=True)
                                draft_input_pos.add_(1)
                                draft_tokens.copy_(next_token)

                                draft_probs.append(probs.clone())
                                draft_output_tokens.append(next_token.clone())
                            
                            draft_probs = torch.cat(draft_probs, dim=1)
                            draft_output_tokens = torch.stack(draft_output_tokens)

                            # Verify the output of the draft model
                            next_token, target_probs = verify(self.target_model, draft_output_tokens, target_input_pos)
                            # Rejection Sampling
                            output_with_bonus_tokens = self.sampler(draft_probs, target_probs, draft_output_tokens[1:], next_token)
                            assert output_with_bonus_tokens.size(0) == 1 # Only batch size 1 is supported

                            output_with_bonus_tokens = output_with_bonus_tokens.squeeze()
                            mask_negative = (output_with_bonus_tokens == -1)
                            if mask_negative.any():
                                num_accepted_tokens = mask_negative.nonzero(as_tuple=True)[0][0]
                            else:
                                num_accepted_tokens = output_with_bonus_tokens.size(0)
                            output_with_bonus_tokens = output_with_bonus_tokens[:num_accepted_tokens]

                            accepted_tokens = torch.unbind(output_with_bonus_tokens)

                            target_input_pos.add_(num_accepted_tokens)
                            tokens.copy_(accepted_tokens[-1])
                            generated_tokens += num_accepted_tokens
                            output_tokens.extend(accepted_tokens)

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
