from __future__ import annotations
import time
import numpy as np
from typing import NamedTuple,Union,Callable,List
from graph_tool.all import Graph
from graph_game.shannon_node_switching_game import Node_switching_game
from GN0.util.convert_graph import convert_node_switching_game
from dataclasses import dataclass
from GN0.util.util import get_one_hot
import torch

@dataclass
class Node:
    """A non-leaf node in the MCTS tree."""
    parent:Union[Node,None]
    storage:Graph
    children:list
    moves:np.ndarray
    priors:np.ndarray
    visits:np.ndarray
    total_value:np.ndarray
    Q:np.ndarray

@dataclass
class Leafnode:
    """A leaf node in the MCTS tree."""
    move:int
    done:bool
    parent:Node
    makerturn:bool
    value:int = -1

def upper_confidence_bound(node:Node,exploration_constant:float):
    """Computes the upper confidence bound for a node.
    
    Args:
        node: The node to compute the upper confidence bound for.
        exploration_constant: A temperature parameter that controls exploration.
    Returns:
        The upper confidence bound for the node.
    """
    return node.Q+exploration_constant*((node.priors*np.sqrt(np.sum(node.visits)+1))/(1+node.visits))

class MCTS():
    """Implements MCTS for Graph games with a neural network for initial node probabilities
    
    Attributes:
        game: The graph game that is played.
        root: The root node of the MCTS tree.
        exploration_constant: A temperature parameter that controls exploration.
        NN: A function that takes a graph and returns a tuple of (moves,probs,value)
    """
    def __init__(self,game:Node_switching_game,NN:Callable,remove_dead_captured=True):
        self.game = game
        self.root = Leafnode(move=-1,parent=None,done=False,makerturn=game.view.gp["m"])
        self.exploration_constant = 1
        self.NN = NN
        self.done = False
        self.winning_move = None
        self.remove_dead_captured = remove_dead_captured

    def reset(self,storage:Graph):
        """Resets the MCTS tree to the given storage.
        Removes all nodes except the root and sets the state
        of the game to the given storage.

        Args:
            storage: The storage to reset the tree to.
        """
        self.game.set_to_graph(storage)
        self.done = False
        self.winning_move = None
        self.root = Leafnode(move=-1,parent=None,done=False,makerturn=self.game.view.gp["m"])

    def next_iter_with_child(self,action,storage):
        assert isinstance(self.root,Node)
        index = np.where(self.root.moves==action)[0][0]
        self.root = self.root.children[index]
        self.root.parent = None
        self.done = False
        self.winning_move = None
        self.game.set_to_graph(storage)

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
            self.game.set_to_graph(Graph(node.parent.storage))
            self.game.make_move(node.move,remove_dead_and_captured=self.remove_dead_captured)
            winner = self.game.who_won()
            node.done = winner is not None
            if winner is not None:
                node.done = True
                if winner==self.game.onturn: # The color the player at the leafnode is on
                    node.value = 1
                else:
                    node.value = 0
                depth,node = self.backup_victory(node)
                for _ in range(depth):
                    if len(path)>0:
                        path.pop()
        return path,node

    def backup_victory(self,node):
        pnode = node
        depth = 0
        if node.value == 0:
            if node.parent == self.root:
                self.done = True
                self.winning_move = self.root.children.index(node)
                return 1,self.root
            new_parent = Leafnode(move=None,parent=node.parent.parent,done=True,makerturn=node.parent.storage.gp["m"],value=1)
            node.parent.parent.children[node.parent.parent.children.index(node.parent)] = new_parent
            depth,pnode=self.backup_victory(new_parent)
        elif node.value == 1:
            for c in node.parent.children:
                if not isinstance(c,Leafnode) or not c.done or c.value!=1:
                    return 0,node
            else:
                if node.parent == self.root:
                    self.done = True
                    self.winning_move = None
                    return 0,self.root
                new_parent = Leafnode(move=None,parent=node.parent.parent,done=True,makerturn=node.parent.storage.gp["m"],value=0)
                node.parent.parent.children[node.parent.parent.children.index(node.parent)] = new_parent
                depth,pnode=self.backup_victory(new_parent)
        return depth,pnode
        

    def expand(self,leafnode:Leafnode): # Assumes that self.game is in correct state
        """Expands a leaf node in the MCTS tree.
        
        Args:
            leafnode: The leaf node to expand.
        Returns:
            The value estimate for the leaf node.
        """
        moves,probs,value = self.NN(self.game)
        print("expanding",leafnode,"Currently on turn:",self.game.onturn,moves,value)
        children = [Leafnode(move=m,done=False,parent=None,makerturn=not self.game.view.gp["m"]) for m in moves]
        node = Node(parent=leafnode.parent,
                    storage=Graph(self.game.graph),
                    children=children,
                    priors=probs,
                    visits=np.zeros(probs.shape,dtype=int),
                    total_value=np.zeros_like(probs),
                    moves=moves,
                    Q = np.ones_like(probs)*0.5) # Encourage first exploration
        for child in children:
            child.parent = node
        if leafnode == self.root:
            self.root = node
        else:
            leafnode.parent.children[leafnode.parent.children.index(leafnode)] = node

        return node,value

    def backtrack(self,path,value,leaf_makerturn):
        """Backtracks the MCTS tree to update the values and visits of the nodes.
        
        Args:
            path: The path to the node to backtrack from.
            value: The value estimate of the node.
        """
        node = self.root
        for ind in path:
            if isinstance(node,Node):
                node.visits[ind]+=1
                if node.storage.gp["m"] == leaf_makerturn:
                    node.total_value[ind]+=value
                else:
                    node.total_value[ind]+=1-value
                node.Q[ind] = node.total_value[ind]/node.visits[ind]
                node = node.children[ind]
            else:
                assert node.done

    def single_iteration(self):
        if self.done:
            return 0 if self.winning_move is None else 1
        path,leaf = self.select_most_promising()
        if leaf==self.root and isinstance(self.root,Node):
            assert self.done
            return 0 if self.winning_move is None else 1
            
        leaf_makerturn = leaf.makerturn
        if leaf.done:
            value = leaf.value
        else:
            leaf,value = self.expand(leaf)
        self.backtrack(path,value,leaf_makerturn=leaf_makerturn)
        return value


    def run(self,iterations=None,max_time=None):
        """Runs the MCTS algorithm for the given number of iterations.
        
        Args:
            iterations: The number of iterations to run the MCTS algorithm for.
        """
        assert iterations is not None or max_time is not None
        start = time.perf_counter()
        it = 0
        while 1:
            if self.done:
                return
            if iterations is not None and it>=iterations:
                print(time.perf_counter()-start,"seconds")
                break
            if max_time is not None and time.perf_counter()-start>max_time:
                print(it,"iterations")
                break
            self.single_iteration()
            it+=1

    def extract_result(self,temperature):
        """Extracts the move probabilities from the MCTS tree.
        Assumes that MCTS has already been run and the MCTS tree has been built.
        
        Args:
            temperature: The temperature parameter that controls exploration.
        Returns:
            The moves and move probabilities for the root position of the MCTS tree.
        """
        assert isinstance(self.root,Node)
        if self.done:
            if self.winning_move is None:
                probs = np.ones(len(self.root.moves))/len(self.root.moves)
            else:
                probs = np.zeros(len(self.root.moves))
                probs[self.winning_move] = 1
        if temperature == np.inf:
            return np.ones(len(self.root.visits))/len(self.root.visits)
        elif temperature == 0:
            probs = get_one_hot(len(self.root.visits),np.argmax(self.root.visits+self.root.priors))
        else:
            powered = self.root.visits**(1/temperature)
            s = np.sum(powered)
            if s==0:
                probs = np.ones(len(self.root.moves))/len(self.root.moves)
            else:
                probs = powered/s
        return self.root.moves,probs
