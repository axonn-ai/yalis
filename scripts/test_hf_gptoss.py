#!/usr/bin/env python3
"""Test GPT-OSS with HuggingFace transformers to verify checkpoint works."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_path = "yalis/external/checkpoints/openai/gpt-oss-20b"
    
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("Model loaded successfully!")
    print(f"Model type: {type(model)}")
    
    # Test generation
    prompts = [
        "How to bake a cake?",
        "What is the capital of France?",
    ]
    
    for prompt in prompts:
        print(f"\n=== Prompt: {prompt} ===")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.8,
                top_p=0.8,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Output: {response}")

if __name__ == "__main__":
    main()
