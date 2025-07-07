# attention/registry.py

ATTENTION_REGISTRY = {}


def register_attention(name):
    def decorator(fn):
        ATTENTION_REGISTRY[name] = fn
        return fn

    return decorator


def get_attention(name):
    if name not in ATTENTION_REGISTRY:
        raise ValueError(f"Attention backend '{name}' not registered.")
    return ATTENTION_REGISTRY[name]
