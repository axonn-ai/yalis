from torch.nn.attention.flex_attention import create_block_mask

@staticmethod
def flex_decode_mask(token_counter):
    def _inner_mask(b, h, q_idx, kv_idx):
        return (kv_idx <= token_counter[b])
    return _inner_mask

def create_causal_block_mask_for_flex_attention(token_counter, kv_len, batch_size):
    return create_block_mask(flex_decode_mask(token_counter), B=batch_size, H=None, Q_LEN=1, KV_LEN=kv_len)