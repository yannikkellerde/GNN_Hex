import torch
import numpy as np
from time import perf_counter
from torch_scatter import scatter

globs = torch.Tensor([[1,2],[3,4],[5,6]]).cuda()
x = torch.Tensor([
    [1,2],
    [3,4],
    [5,6],
    [7,8],
    [9,10],
    [11,12]
]).cuda()
graph_slices = torch.Tensor([0,2,4,6]).long().cuda()
graph_indices = torch.Tensor([0,0,1,1,2,2]).long().cuda()

print(scatter(x,graph_indices,dim=0,reduce="max"))

print(torch.max(torch.stack([x[graph_slices[i]:graph_slices[i+1]] for i in range(len(globs))]),1).values)
exit()
"""
binary_matrix = torch.zeros(x.size(0),globs.size(0)).cuda()
for i in range(len(graph_slices)-1):
    binary_matrix[graph_slices[i]:graph_slices[i+1],i] = 1

what_i_need = torch.Tensor(
    [[]]
)
print(torch.matmul(x.T,binary_matrix))

what_i_want = torch.Tensor([
    [[1,3],[2,4]],
    [[5,7],[6,8]],
    [[9,11],[10,12]]
]).cuda()
print(torch.max(what_i_want,2).values)
exit()
"""
# globs.shape = [64, 16]
# x.shape = [4335, 16]
# graph_slices.shape = [65]

globs = torch.rand([64, 16]).cuda()
x = torch.rand([4335, 16]).cuda()
graph_slices = torch.linspace(0,len(x),len(globs)+1).long().cuda()

binary_matrix = torch.zeros(x.size(0),globs.size(0)).cuda()
for i in range(len(graph_slices)-1):
    binary_matrix[graph_slices[i]:graph_slices[i+1],i] = 1

binT = binary_matrix.T

#print(f"{globs.shape = }, {x.shape = }, {graph_slices.shape = }, {fancy_matrix.shape = }")
#print(fancy_matrix)
#print(globs)


def some_test(x):
    binary_matrix = torch.zeros(x.size(0),globs.size(0)).cuda()
    for i in range(len(graph_slices)-1):
        binary_matrix[graph_slices[i]:graph_slices[i+1],i] = 1

def default_method(x):
    for i in range(len(globs)):
        x[graph_slices[i]:graph_slices[i+1]] += globs[i]
    return x

def other_method(x):
    x+=torch.matmul(binary_matrix,globs)
    return x

def default_max_pool(x):
    return torch.stack([torch.max(x[graph_slices[i]:graph_slices[i+1]],0).values for i in range(len(globs))])

def second_max_pool(x):
    return torch.max(torch.stack([x[graph_slices[i]:graph_slices[i+1]] for i in range(len(globs))]),1).values

def fancy_max_pool(x):
    return torch.max(torch.matmul(binT,x),0).values

def test_method(func):
    timings = []
    for _ in range(100):
        y = x.clone()
        start = perf_counter()
        func(y)
        timings.append(perf_counter() - start)
    return np.mean(timings)

print(fancy_max_pool(x))
print(default_max_pool(x))

assert(default_method(x).equal(other_method(x)))
assert(second_max_pool(x).equal(default_max_pool(x)))
print(f"default: {test_method(default_method)}")
print(f"other: {test_method(other_method)}")
print(f"some_test: {test_method(some_test)}")
print(f"default_max_pool: {test_method(default_max_pool)}")
print(f"second_max_pool: {test_method(second_max_pool)}")