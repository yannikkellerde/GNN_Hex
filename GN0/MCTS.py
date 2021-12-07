from __future__ import annotations
import numpy as np
from typing import NamedTuple,Union,Callable
from graph_tool.all import Graph
from game.graph_tools_game import Graph_game,Graph_Store
import GNZero.util as util

class Node(NamedTuple):
    """A non-leaf node in the MCTS tree."""
    parent:Union[Node,None]
    storage:Graph_Store
    children:tuple
    moves:Union[None,np.ndarray]
    priors:np.ndarray
    visits:np.ndarray
    total_value:np.ndarray
    Q:np.ndarray

class Leafnode(NamedTuple):
    """A leaf node in the MCTS tree."""
    move:int
    parent:Node
    done:bool

def upper_confidence_bound(node:Node,exploration_constant:float):
    """Computes the upper confidence bound for a node.
    
    Args:
        node: The node to compute the upper confidence bound for.
        exploration_constant: A temperature parameter that controls exploration.
    Returns:
        The upper confidence bound for the node.
    """
    return node.Q+exploration_constant*((node.priors*np.sqrt(np.sum(node.visits)))/(1+node.visits))

class MCTS():
    """Implements MCTS for Graph games with a neural network for initial node probabilities
    
    Attributes:
        game: The graph game that is played.
        root: The root node of the MCTS tree.
        exploration_constant: A temperature parameter that controls exploration.
        NN: A function that takes a graph and returns a tuple of (moves,probs,value)
    """
    def __init__(self,game:Graph_game,NN:Callable[[Graph],(np.ndarray,np.ndarray,float)]):
        self.game = game
        self.root = Leafnode(move=-1,parent=None,done=False)
        self.exploration_constant = 1
        self.NN = NN

    def reset(self,storage:Graph_Store):
        """Resets the MCTS tree to the given storage.
        Removes all nodes except the root and sets the state
        of the game to the given storage.

        Args:
            storage: The storage to reset the tree to.
        """
        self.game.load_storage(storage)
        self.root = Leafnode(move=-1,parent=None,done=False)

    def choose_child(self,node:Node):
        """Chooses a child of a node according to the upper confidence bound.
        
        Args:
            node: The node to choose a child of.
        Returns:
            The index of the chosen child.
        """
        return np.argmax(upper_confidence_bound(node,self.exploration_constant))

    def select_most_promising(self):
        """Selects the most promising node in the MCTS tree.
        
        Returns:
            A tuple of the path to the most promising node and the node itself.
        """
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
        """Expands a leaf node in the MCTS tree.
        
        Args:
            leafnode: The leaf node to expand.
        Returns:
            The value estimate for the leaf node.
        """
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
        """Backtracks the MCTS tree to update the values and visits of the nodes.
        
        Args:
            path: The path to the node to backtrack from.
            value: The value estimate of the node.
        """
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
        """Runs the MCTS algorithm for the given number of iterations.
        
        Args:
            iterations: The number of iterations to run the MCTS algorithm for.
        """
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
        """Extracts the move probabilities from the MCTS tree.
        Assumes that MCTS has already been run and the MCTS tree has been built.
        
        Args:
            temperature: The temperature parameter that controls exploration.
        Returns:
            The moves and move probabilities for the root position of the MCTS tree.
        """
        assert isinstance(self.root,Node)
        if temperature == 0:
            probs = util.get_one_hot(self.root.visits.length,np.argmax(self.root.visits))
        else:
            powered = self.root.visits**(1/temperature)
            probs = powered/np.sum(powered)
        return self.root.moves,probs