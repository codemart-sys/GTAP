import itertools
import math
import torch
MASKS = tuple(itertools.product((0, 1), repeat=3))
MASK_ID = {mask: index for index, mask in enumerate(MASKS)}
def exact_shapley(values):
    if values.ndim != 2 or values.size(1) != 8: raise ValueError('expected [B,8] coalition values')
    values = values if values.is_floating_point() else values.float()
    phi=values.new_zeros(values.size(0),3)
    for player in range(3):
        for coalition in MASKS:
            if coalition[player]: continue
            k=sum(coalition); added=list(coalition); added[player]=1
            phi[:,player]+=math.factorial(k)*math.factorial(2-k)/math.factorial(3)*(values[:,MASK_ID[tuple(added)]]-values[:,MASK_ID[coalition]])
    return phi
