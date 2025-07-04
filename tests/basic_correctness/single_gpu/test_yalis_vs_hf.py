import pytest
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from yalis import ModelConfig, InferenceConfig, LLMEngine
from types import SimpleNamespace
from tests.sample_dataset import AlpacaDataset
import warnings

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda" 
NUM_LOGPROBS = 5

@pytest.fixture(scope="module")
def model_id(request):
    return request.config.getini("model")

@pytest.fixture(scope="module")
def dtype(request):
    dt = request.config.getini("dtype").lower()

    yalis_dt = dt

    hf_map = {
      "bf16": torch.bfloat16,
      "fp16": torch.float16,
      "fp32": torch.float32,
    }

    hf_dt = hf_map[dt]

    return SimpleNamespace(yalis=yalis_dt, hf=hf_dt)

@pytest.fixture(scope="module")
def attn_backend(request):
    attnb = request.config.getini("attn_backend").lower()
    yalis_attnb = attnb

    hf_map = {
      "sdpa": "sdpa",
      "flash": "flash_attention_2",
      "flex": "flash_attention_2", # For some reason, flex does not work with in hf right now
    }

    hf_attnb = hf_map[attnb]

    return SimpleNamespace(yalis=yalis_attnb, hf=hf_attnb)

@pytest.fixture(scope="module")
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer

@pytest.fixture(scope="module")
def alpaca_dataset():
    dataset = AlpacaDataset(random_seed=42)
    return dataset

@pytest.fixture(scope="module")
def hf_model(dtype, attn_backend):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, attn_implementation=attn_backend.hf, torch_dtype=dtype.hf, device_map="auto", trust_remote_code=True).to(DEVICE)
    model.eval()
    return model

@pytest.fixture(scope="module", autouse=True)
def yalis_engine(dtype, attn_backend):
    model_config = ModelConfig(model_name=MODEL_ID, precision=dtype.yalis)
    inference_config = InferenceConfig(
        batch_size=1,  # initial batch size, will be changed with reset_kv_cache
        max_length_of_generated_sequences=2048,
        top_p=0.0,
        temperature=0.0,
        tp_dims=None,
        attention_backend=attn_backend.yalis,
        use_paged_kv_caching=False,
    )
    return LLMEngine(model_config=model_config, inference_config=inference_config)

def random_prompt(tokenizer, length, seed=42):
    random.seed(seed)
    vocab = list(tokenizer.get_vocab().values())
    special_ids = set(tokenizer.all_special_ids)
    vocab = [tid for tid in vocab if tid not in special_ids]
    token_ids = random.choices(vocab, k=length)
    return tokenizer.decode(token_ids, skip_special_tokens=True)

def alpaca_prompt(alpaca_dataset, tokenizer, length, batch_size):
    samples = alpaca_dataset.sample(tokenizer, batch_size, input_len=length, return_prompt_formatted=True) 
    input_prompts = []
    for sample in samples:
        input_prompts.append(sample.prompt)
    return input_prompts 


def _get_logprobs(logits):
    # logits: list of [batch_size, vocab_size] tensors of length num_tokens
    num_tokens = len(logits)
    batch_size = logits[0].shape[0]

    logprob_list: list[torch.Tensor] = []
    topk_list: list[torch.Tensor] = []
    for logit in logits:
        logprobs = torch.log_softmax(logit, dim=-1, dtype=torch.float32)
        logprob_list.append(logprobs)

        topk_indices = torch.argsort(logprobs, dim=-1, descending=True, stable=True)[:, :NUM_LOGPROBS]
        topk_list.append(topk_indices)
    
    # We need to convert this to a list of [num_tokens, -1] tensors of length batch_size
    final_logprob_list = []
    final_topk_list = []
    for i in range(batch_size):
        per_prompt_logprobs = []
        per_prompt_topk = []
        for j in range(num_tokens):
            per_prompt_logprobs.append(logprob_list[j][i, :].cpu())
            per_prompt_topk.append(topk_list[j][i, :].cpu())

        per_prompt_logprobs = torch.stack(per_prompt_logprobs)
        per_prompt_topk = torch.stack(per_prompt_topk)

        final_logprob_list.append(per_prompt_logprobs)
        final_topk_list.append(per_prompt_topk)

    return final_logprob_list, final_topk_list


def _get_hf_output(tokenizer, model, prompts, num_tokens):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad(), torch.autocast(
            DEVICE, dtype=torch.float16, cache_enabled=False
        ):
        output = model.generate(
            **inputs,
            max_new_tokens=num_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.0,
            top_p=0.0,
            output_logits=True,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
    
    # For batch, return list of new tokens for each prompt
    new_tokens = []
    for i in range(len(prompts)):
        input_len = inputs["input_ids"][i].shape[0]
        new_tokens.append(output.sequences[i][input_len:input_len+num_tokens].cpu())
    
    # new_tokens: list of [num_tokens] tensors of length batch_size
    # output.logits: list of [batch_size, vocab_size] tensors of length num_tokens
    return new_tokens, output.logits

def _get_yalis_output(engine, prompts, num_tokens):
    output_tokens, _, logits = engine.generate(prompts, report_throughput=False, tokens_to_generate=num_tokens, get_logits=True)
    # output_tokens: (batch, num_tokens)
    return [output_tokens[i][:num_tokens].cpu() for i in range(len(prompts))], logits

# This test does not mean a failure
def _compare_tokens_and_text(tokenizer, tokens1, tokens2):
    token_mismatches = 0
    text_mismatches = 0
    assert len(tokens1) == len(tokens2), f"Batch size mismatch: {len(tokens1)} vs {len(tokens2)}"
    for t1, t2 in zip(tokens1, tokens2):
        assert len(t1) == len(t2), f"Token length mismatch: {len(t1)} vs {len(t2)}"
        num_matches = sum(a == b for a, b in zip(t1, t2))
        if num_matches < len(t1) - 1:
            warnings.warn(f"Token mismatch: {t1} vs {t2}")
            token_mismatches += 1
        text1 = tokenizer.decode(t1, skip_special_tokens=True)
        text2 = tokenizer.decode(t2, skip_special_tokens=True)
        if text1.strip()[:10] != text2.strip()[:10]:
            warnings.warn(f"Text mismatch: {text1} vs {text2}")
            text_mismatches += 1

    return token_mismatches == 0 and text_mismatches == 0

def _compare_logprobs(hf_logits, hf_tokens, yalis_logits, yalis_tokens):
    hf_logprobs, hf_topk = _get_logprobs(hf_logits)
    yalis_logprobs, yalis_topk = _get_logprobs(yalis_logits)

    assert len(hf_logprobs) == len(yalis_logprobs), f"Batch size mismatch: {len(hf_logprobs)} vs {len(yalis_logprobs)}"
    assert len(hf_topk) == len(hf_tokens), f"HF tokens and topk length mismatch: {len(hf_topk)} vs {len(hf_tokens)}"
    assert len(yalis_topk) == len(yalis_tokens), f"Yalis tokens and topk length mismatch: {len(yalis_topk)} vs {len(yalis_tokens)}"
    assert len(hf_logprobs) == len(hf_tokens), f"HF logprobs and tokens length mismatch: {len(hf_logprobs)} vs {len(hf_tokens)}"

    for hf_token, yalis_token, hf_logprob, yalis_logprob, hf_topk, yalis_topk in zip(hf_tokens, yalis_tokens, hf_logprobs, yalis_logprobs, hf_topk, yalis_topk):
        assert hf_token.shape == yalis_token.shape, f"Token shape mismatch: {hf_token.shape} vs {yalis_token.shape}"

        for i in range(len(hf_token)):
            hf_token_i = hf_token[i]
            yalis_token_i = yalis_token[i]

            token_mismatch = hf_token_i != yalis_token_i

            if token_mismatch:
                warnings.warn(f"Token mismatch {i}: HF Token: {hf_token_i} vs Yalis Token: {yalis_token_i}")
                # Check if the tokens are in the top NUM_LOGPROBS of each other
                hf_token_i_in_topk = hf_token_i in yalis_topk[i]
                yalis_token_i_in_topk = yalis_token_i in hf_topk[i]
                
                assert hf_token_i_in_topk, f"HF token {hf_token_i} not in Yalis topk {yalis_topk[i]}, {i}: HF topk {len(hf_topk)} - {hf_logprob[i]}, {yalis_logprob[i]}"
                assert yalis_token_i_in_topk, f"Yalis token {yalis_token_i} not in HF topk {hf_topk[i]}, {i}: Yalis topk {len(yalis_topk)} - {yalis_logprob[i]}, {hf_logprob[i]}"

                # Now the tokens will diverge, so need to break
                break


BATCH_SIZES = [1, 4, 8]
PROMPT_LENGTHS = [128, 256, 512, 1024]

@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("prompt_length", PROMPT_LENGTHS)
@pytest.mark.filterwarnings("ignore:.*do_sample.*:UserWarning")
@pytest.mark.filterwarnings("ignore:.*co_lnotab.*:DeprecationWarning")
def test_01_prefill(tokenizer, hf_model, yalis_engine, batch_size, prompt_length, dtype, alpaca_dataset):
    yalis_engine.reset_kv_cache(batch_size)
    prompts = alpaca_prompt(alpaca_dataset, tokenizer, prompt_length, batch_size)
    hf_tokens, hf_logits = _get_hf_output(tokenizer, hf_model, prompts, num_tokens=1)
    yalis_tokens, yalis_logits = _get_yalis_output(yalis_engine, prompts, num_tokens=1)
    _compare_logprobs(hf_logits, hf_tokens, yalis_logits, yalis_tokens)

@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("prompt_length", PROMPT_LENGTHS)
@pytest.mark.filterwarnings("ignore:.*do_sample.*:UserWarning")
@pytest.mark.filterwarnings("ignore:.*co_lnotab.*:DeprecationWarning")
def test_02_decode(tokenizer, hf_model, yalis_engine, batch_size, prompt_length, dtype, alpaca_dataset):
    yalis_engine.reset_kv_cache(batch_size)
    prompts = alpaca_prompt(alpaca_dataset, tokenizer, prompt_length, batch_size)
    hf_tokens, hf_logits = _get_hf_output(tokenizer, hf_model, prompts, num_tokens=32)
    yalis_tokens, yalis_logits = _get_yalis_output(yalis_engine, prompts, num_tokens=32)
    _compare_logprobs(hf_logits, hf_tokens, yalis_logits, yalis_tokens)

# TODO: Add perplexity test
