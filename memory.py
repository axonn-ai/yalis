import torch

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory
        reserved = torch.cuda.memory_reserved(i)
        allocated = torch.cuda.memory_allocated(i)
        free = reserved - allocated
        print(f"GPU {i}: Total: {total / 1e6:.1f} MB | Free: {free / 1e6:.1f} MB | Allocated: {allocated / 1e6:.1f} MB")
else:
    print("No GPU available.")
