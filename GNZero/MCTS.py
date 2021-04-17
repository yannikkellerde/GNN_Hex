from __future__ import annotations
import numpy as np
from typing import NamedTuple,Union,Callable
from graph_tool.all import Graph
from game.graph_tools_game import Graph_game,Graph_Store
import GNZero.util as util

class Node(NamedTuple):
    parent:Union[Node,None]
    storage:Graph_Store
    children:tuple
    moves:Union[None,np.ndarray]
    priors:np.ndarray
    visits:np.ndarray
    total_value:np.ndarray
    Q:np.ndarray

class Leafnode(NamedTuple):
    move:int
    parent:Node
    done:bool

def upper_confidence_bound(node:Node,exploration_constant:float):
    return node.Q+exploration_constant*((node.priors*np.sqrt(np.sum(node.visits)))/(1+node.visits))

class MCTS():
    def __init__(self,game:Graph_game,NN:Callable[[Graph],(np.ndarray,np.ndarray,float)]):
        self.game = game
        self.root = Leafnode(move=-1,parent=None,done=False)
        self.exploration_constant = 1
        self.NN = NN
    def reset(self,storage:Graph_Store):
        self.game.load_storage(storage)
        self.root = Leafnode(move=-1,parent=None,done=False)
    def choose_child(self,node:Node):
        return np.argmax(upper_confidence_bound(node,self.exploration_constant))
    def select_most_promising(self):
        node = self.root
        path = []
        while isinstance(node,Node):
            child_index = self.choose_child(node)
            node = node.children[child_index]
            path.append(child_index)
        if node!=self.root and not node.done:
            self.game.load_storage(node.parent.storage)
            node.done = self.game.make_move(node.move)
        return path,node
    def expand(self,leafnode:Leafnode): # Assumes that self.game is in correct state
        moves,probs,value = self.NN(self.game.view)
        children = (Leafnode(move=m,done=False) for m in moves)
        node = Node(parent=leafnode.parent,
                    storage=self.game.extract_storage(),
                    children=children,
                    priors=probs,
                    visits=np.zeros(probs.shape,dtype=int),
                    total_value=np.zeros_like(probs),
                    moves=moves if leafnode==self.root else None,
                    Q = np.zeros_like(probs))
        for child in children:
            child.parent = node
        leafnode.parent.children[leafnode.parent.children.index(leafnode)] = node
        if leafnode == self.root:
            self.root = node
        return value
    def backtrack(self,path,value):
        node = self.root
        for ind in path:
            if isinstance(node,Node):
                node.visits[ind]+=1
                node.total_value[ind]+=value
                node.Q[ind] = node.total_value/node.visits
                node = node.children[ind]
            else:
                assert node.done
    def run(self,iterations):
        for i in iterations:
            path,leaf = self.select_most_promising()
            if leaf.done:
                if self.game.onturn == "b":
                    value = -1
                else:
                    value = 1
            else:
                value = self.expand(leaf)
            self.backtrack(path,value)
    def extract_result(self,temperature):
        assert isinstance(self.root,Node)
        if temperature == 0:
            probs = util.get_one_hot(self.root.visits.length,np.argmax(self.root.visits))
        else:
            powered = self.root.visits**(1/temperature)
            probs = powered/np.sum(powered)
        return self.root.moves,probs