import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from fractions import Fraction

def can_divide(tensor_dim, pct: float) -> (bool, int, int):
    """
    Takes the pct of a tensor that a rank should get, as a float,
    indicates whether or not this will result in the rank getting
    a whole number of elements, also returns a fraction version of
    the float
    """
    num, den = Fraction(pct).limit_denominator().as_integer_ratio()
    return ((num * tensor_dim) % den == 0, num, den)

def compute_offset(tensor_dim, asymmetric_map, rank):
    offset = 0
    for i in range(0, rank):
        pct = asymmetric_map[i]
        check, num, den = can_divide(tensor_dim, pct)
        assert check
        offset += (tensor_dim * num) // den
    return offset

def compute_all_shapes(full_shape, asymmetric_map):
    shapes = []
    for i in range(len(asymmetric_map)):
        pct = asymmetric_map[i]
        check, num, den = can_divide(full_shape[0], pct)
        assert check
        new_shape = list(full_shape)
        new_shape[0] = (full_shape[0] * num) // den
        new_shape = torch.Size(new_shape)
        shapes.append(new_shape)
    return shapes

def yalis_all_reduce(input_tensor, process_group=None):
    input_tensor = input_tensor.contiguous()
    if not dist.get_world_size(process_group) > 1:
        return input_tensor
    dist.all_reduce(input_tensor, group=process_group)
    return input_tensor

"""
To use w/ asymmetric tensors, pass a list of fractions of the tensor
that each rank should handle, rank is implicitly the position in the list
"""

def yalis_all_gather(input_tensor, dim, process_group=None, asymmetric=None, original_shape=(0, 0)):
    input_tensor = input_tensor.contiguous()
    world_size = dist.get_world_size(process_group)
    if not world_size > 1:
        return input_tensor
    rank = dist.get_rank(process_group)
    if asymmetric == None:
        tensor_list = [
            torch.empty_like(input_tensor) for _ in range(dist.get_world_size(process_group))
        ]
        tensor_list[rank] = input_tensor
        dist.all_gather(tensor_list, input_tensor, group=process_group)
        output = torch.cat(tensor_list, dim=dim).contiguous()
        return output
    else:
        shape_list = compute_all_shapes(original_shape, asymmetric)
        #print(shape_list)
        #print(input_tensor.shape)
        tensor_list = [
            input_tensor.new_empty(shape) for shape in shape_list
        ]
        dist.all_gather(tensor_list, input_tensor, group=process_group)
        output = torch.cat(tensor_list, dim=dim).contiguous()
        return output


def yalis_drop(input_tensor, dim, process_group=None, asymmetric=None):
    input_tensor = input_tensor.contiguous()
    world_size = dist.get_world_size(process_group) 
    if not world_size > 1:
        return input_tensor
    this_chunk = dist.get_rank(process_group)
    input_shape = input_tensor.shape[dim] 
    if asymmetric == None:
        #print("symmetric drop")
        assert input_tensor.shape[dim] % world_size == 0
        chunk_size = input_shape // world_size
        out = torch.narrow(input_tensor, dim, this_chunk * chunk_size, chunk_size)
        #print("sym out shape: ", out.shape)
        return out
    else:
        #print("asymmetric drop")
        pct = asymmetric[this_chunk]
        check, num, den = can_divide(input_shape, pct)
        assert check
        chunk_size = (input_shape * num) // den
        out = torch.narrow(
            input_tensor, dim,
            compute_offset(input_shape, asymmetric, this_chunk),
            chunk_size
        ).contiguous()
        #print("asym out shape: ", out.shape)
        return out

BACKEND = "nccl"
DEVICE = "cuda"

def test():
    # setup distributed
    rank = os.environ["RANK"]
    dist.init_process_group(backend=BACKEND)
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    world_size = dist.get_world_size()
    pg = dist.group.WORLD
    name = "cuda:0"
    device = torch.device(name)
    print("On device: ", torch.cuda.get_device_name(device))

    asymmetric_map = [0.25, 0.75]

    # set up tensors
    dist.barrier()
    if rank == "0":
        print("ORIGINAL TENSORS: ")
    x = torch.full((8, 8), float(rank) + 1.0, device=DEVICE)
    original_shape = x.shape
    #print("Can divide?: ", can_divide(x.shape[0], 0.25))
    print(x)

    dist.barrier()
    if rank == "0":
        print("NEW TENSORS: ")
    x = yalis_drop(x, 0, process_group=pg, asymmetric=asymmetric_map)
    #yalis_all_reduce(x, process_group=pg)
    print(x)
    
    dist.barrier()
    if rank == "0":
        print("NEW NEW TENSORS: ")
    x = yalis_all_gather(x, 0, process_group=pg, asymmetric=asymmetric_map, original_shape=original_shape)
    print(x)

    # check tensors
    #expected = sum(float(r) for r in range(world_size))
    #print(f"[rank {rank}] after all_reduce: {no}  yes")
    dist.destroy_process_group()

if __name__ == "__main__":
    test()

