import torch
from GN0.models import get_pre_defined
import time

device = "cuda"

def load_model():
    model = get_pre_defined("HexAra")
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


if __name__ == "__main__":
    model = load_model()
    test_no_swap(model)
    test_all_swap(model)
    test_some_swap(model)
