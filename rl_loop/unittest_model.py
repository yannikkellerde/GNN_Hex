import torch
# from GN0.models import get_pre_defined
from GN0.torch_script_models import get_current_model
import time
import random
import numpy as np

device = "cuda"

def load_model():
    model = get_current_model()
    traced = torch.jit.script(model)
    model.to(device)
    return model

def test_no_swap(model):
    x = torch.zeros([10,3]).to(device)
    edge_index = torch.tensor([[0,3,4,5,6,8],[1,2,3,7,9,6]]).to(device)
    graph_indices = torch.tensor([0,0,0,0,0,1,1,1,1,1]).to(device)
    batch_ptr = torch.tensor([0,5,10]).to(device)

    pi,value,gi,bp = model(x,edge_index,graph_indices,batch_ptr)
    assert pi.size() == (6,)
    assert value.size() == (2,)
    assert (gi == torch.tensor([0,0,0,1,1,1],device=gi.device)).all()
    assert (bp == torch.tensor([0,3,6],device=bp.device)).all()

def test_all_swap(model):
    x = torch.ones([10,3]).to(device)
    edge_index = torch.tensor([[0,3,4,5,6,8],[1,2,3,7,9,6]]).to(device)
    graph_indices = torch.tensor([0,0,0,0,0,1,1,1,1,1]).to(device)
    batch_ptr = torch.tensor([0,5,10]).to(device)

    pi,value,gi,bp = model(x,edge_index,graph_indices,batch_ptr)
    assert pi.size() == (8,)
    assert value.size() == (2,)
    assert (gi == torch.tensor([0,0,0,0,1,1,1,1],device=gi.device)).all()
    assert (bp == torch.tensor([0,4,8],device=bp.device)).all()

def test_some_swap(model):
    x = torch.zeros([14,3]).to(device)
    x[5:,2] = 1
    edge_index = torch.tensor([[0,3,4,5,6,8,11,13,12],[1,2,3,7,9,6,13,11,10]]).to(device)
    graph_indices = torch.tensor([0,0,0,0,0,1,1,1,1,1,2,2,2,2]).to(device)
    batch_ptr = torch.tensor([0,5,10,14]).to(device)

    pi,value,gi,bp = model(x,edge_index,graph_indices,batch_ptr)
    assert pi.size() == (10,)
    assert value.size() == (3,)
    assert (gi == torch.tensor([0,0,0,1,1,1,1,2,2,2],device=gi.device)).all()
    assert (bp == torch.tensor([0,3,7,10],device=bp.device)).all()

def test_randomized(model):
    random.seed(42)
    np.random.seed(4)
    for _ in range(100):
        num_graphs = random.randint(1,30)
        num_nodes = random.randint(3,30)*num_graphs
        num_edges = random.randint(30,60)*num_graphs
        x = torch.zeros([num_nodes,3],device=device)
        edge_index = torch.randint(0,num_nodes,(2,num_edges),device=device)
        graph_indices = torch.tensor(list(sorted(sum([[i,i,i] for i in range(num_graphs)],[])+np.random.randint(0,num_graphs,num_nodes-3*num_graphs).tolist())),dtype=torch.long,device=device)
        batch_ptr = [0]
        cur = 0
        for i in range(len(graph_indices)):
            if graph_indices[i]!=cur:
                batch_ptr.append(i)
                cur = graph_indices[i]
        batch_ptr.append(len(graph_indices))
        batch_ptr = torch.tensor(batch_ptr,dtype=torch.long,device=device)
        did_swap = []
        for start,fin in zip(batch_ptr[:-1],batch_ptr[1:]): 
            if random.random()>0.5:
                did_swap.append(True)
                x[start:fin,2] = 1
            else:
                did_swap.append(False)
        pi,value,gi,bp = model(x,edge_index,graph_indices,batch_ptr)

        assert value.size() == (num_graphs,)
        assert bp.size() == (num_graphs+1,)
        assert pi.size() == gi.size() == (num_nodes-num_graphs*2+np.sum(did_swap))
        minus = 0
        for i,(start,fin) in enumerate(zip(bp[:-1],bp[1:])):
            assert (gi[start:fin] == gi[start]).all()
            if fin<len(gi):
                assert gi[start]!=gi[fin]
            assert start==batch_ptr[i]-minus
            # print(torch.sum(torch.exp(pi[start:fin])),start,fin)
            assert torch.isclose(torch.sum(torch.exp(pi[start:fin])),torch.tensor([1],dtype=torch.float,device=device))
            minus+=2-did_swap[i]


if __name__ == "__main__":
    model = load_model()
    test_no_swap(model)
    test_all_swap(model)
    test_some_swap(model)
    test_randomized(model)
