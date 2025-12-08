import torch
# import yalis_moe_ops  # this should load the .so and run static initializers

print("namespaces:", [ns for ns in dir(torch.ops) if not ns.startswith("_")])

print("\nHas moe_ops namespace?", hasattr(torch.ops, "moe_ops"))
if hasattr(torch.ops, "moe_ops"):
    print("Ops in torch.ops.moe_ops:", dir(torch.ops.moe_ops))
