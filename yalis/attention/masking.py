from torch.nn.attention.flex_attention import create_block_mask

@staticmethod
def flex_decode_mask(token_counter):
    def _inner_mask(b, h, q_idx, kv_idx):
        return (kv_idx <= token_counter[b])
    return _inner_mask

@staticmethod
def flex_prefill_mask():
    def _inner_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    return _inner_mask

def create_causal_block_mask_for_flex_attention(T, token_counter, kv_len, batch_size):
    """
    Create a causal block mask for flex attention backend.

    Inputs:
        T: The length of the query tensor.
        token_counter: A list of integers of length batch_size, where each integer is the number of tokens in the batch.
        kv_len: The length of the key and value tensors.
        batch_size: The number of batches.

    Returns:
        A block mask for flex attention.
    """
    if T==1:
        return create_block_mask(flex_decode_mask(token_counter), B=batch_size, H=None, Q_LEN=1, KV_LEN=kv_len)
    else:
        return create_block_mask(flex_prefill_mask(), B=batch_size, H=None, Q_LEN=T, KV_LEN=T)
