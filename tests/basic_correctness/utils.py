import random


# Utility functions
def random_prompt(tokenizer, length, seed=42):
    random.seed(seed)
    vocab = list(tokenizer.get_vocab().values())
    special_ids = set(tokenizer.all_special_ids)
    vocab = [tid for tid in vocab if tid not in special_ids]
    token_ids = random.choices(vocab, k=length)
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def alpaca_prompt(alpaca_dataset, tokenizer, length, batch_size):
    """Generate alpaca prompts for testing."""
    samples = alpaca_dataset.sample(
        tokenizer, batch_size, input_len=length, return_prompt_formatted=True
    )
    input_prompts = []
    for sample in samples:
        input_prompts.append(sample.prompt)
    return input_prompts
