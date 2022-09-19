"""TODO: Multiprocessing if it becomes nescessary"""

from graph_game.graph_tools_games import Hex_game
from GN0.convert_graph import convert_node_switching_game
from typing import List, Tuple, Union
from torch_geometric.data import Data, Batch
import numpy as np
from numpy.typing import NDArray
from torch import LongTensor, Tensor
import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Env_manager():
    last_obs:List[Data]

    def __init__(self,num_envs,hex_size,gamma=1,n_steps=[1]):
        self.num_envs = num_envs
        self.gamma = gamma
        self.global_onturn = "m"
        self.n_steps = n_steps
        self.change_hex_size(hex_size)

    def change_hex_size(self,new_size):
        self.hex_size = new_size
        self.global_onturn = "m"
        self.envs = [Hex_game(self.hex_size) for _ in range(self.num_envs)]
        self.base_game = Hex_game(self.hex_size)

    @property
    def starting_obs(self):
        return convert_node_switching_game(self.base_game.view,global_input_properties=[int(self.base_game.view.gp["m"])],need_backmap=True)
        

    def observe(self) -> List[Data]:
        f = [convert_node_switching_game(env.view,global_input_properties=[int(env.view.gp["m"])],need_backmap=True) for env in self.envs]
        # assert torch.all(Batch.from_data_list(f).x[:,2] == Batch.from_data_list(f).x[0,2])
        return f

    @staticmethod
    def validate_actions(states:List[Data],actions:List[int]):
        return [state.backmap[action].item() for state,action in zip(states,actions)]

    def get_valid_actions(self) -> List[np.ndarray]:
        return [x.get_actions() for x in self.envs]

    def sample(self) -> np.ndarray:
        return np.array([np.random.choice(x) for x in self.get_valid_actions()])

    def step(self,actions:Union[NDArray[np.int_],List[int],Tensor]) -> Tuple[List[Data],NDArray[np.double],NDArray[np.bool_],List[dict]]:
        rewards = np.zeros(self.num_envs,dtype=float) 
        dones = np.zeros(self.num_envs,dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        for i,(act,env) in enumerate(zip(actions,self.envs)):
            env.make_move(int(act),remove_dead_and_captured=True)
            winner = env.who_won()
            if winner is not None:
                dones[i] = True
                infos[i]["episode_metrics"] = {
                    "return": 1 if winner == "m" else -1,
                    "discounted_return": float(1*self.gamma**env.total_num_moves if winner == "m" else -1*self.gamma**env.total_num_moves),
                    "length": env.total_num_moves,
                    "time":time.perf_counter()-env.creation_time
                }
                if winner == env.not_onturn:
                    rewards[i] = 1
                else: # This can happen, because of "remove dead and captured"
                    rewards[i] = -1
                self.envs[i] = Hex_game(self.hex_size)
                self.envs[i].view.gp["m"] = self.global_onturn=="b" # We swap global onturn afterwards

        self.global_onturn = "m" if self.global_onturn=="b" else "b"
        assert self.global_onturn==self.envs[0].onturn
        states = self.observe()
        return states,rewards,dones,infos
                
    def reset(self):
        self.global_onturn = "m"
        self.envs = [Hex_game(self.hex_size) for _ in range(self.num_envs)]
        return self.observe()

    def get_transitions(self,starting_states:list,state_history:list,action_history:list,reward_history:list,done_history:list):
        """Converts history of s,r,d from the step function to s,r,d transitions as they will be used in RL

        This is nescessary, because we are playing a two player game and thus, the 'next' state is the state after the
        opponent made a move

        Args:
            starting_states: 1D np.ndarray of type object containing Data objects
            state_history: 2D np.ndarray of type object containing Data objects (or list of np.ndarray)
            action_history: 2D np.ndarray of type int (or list of np.ndarray)
            reward_history: 2D np.ndarray of type float (or list of np.ndarray)
            done_history: 2D np.ndarray of type bool (or list of np.ndarray)

        Returns:
            List of Transitions of shape (state,action,reward,next_state,is_done)
        """
        maker_transitions = []
        breaker_transitions = []
        sh = state_history.copy()
        sh.insert(0,starting_states)
        for i in range(len(action_history)):
            start_state = sh[i]
            action = action_history[i]
            transits = maker_transitions if start_state[0].x[0,2] == 1 else breaker_transitions
            for n_step in self.n_steps:
                if len(sh)>i+2*n_step:
                    for k in range(len(start_state)):
                        assert action[k]<len(start_state[k].x)
                        reward = 0
                        for j in range(i,i+2*n_step):
                            reward+=reward_history[j][k]*((-((j-i)%2))*2+1)
                            if done_history[j][k]:
                                sobs = self.starting_obs
                                sobs.x[:,2] = start_state[k].x[0,2]
                                transits.append((start_state[k],action[k],reward,sobs,True))
                                break
                        else:
                            transits.append((start_state[k],action[k],reward,sh[i+2*n_step][k],False))
        return maker_transitions,breaker_transitions


class Debugging_manager(): # A very simple environment for testing purposes
    def __init__(self,num_envs,*args,**kwargs):
        env_x_1 = torch.tensor([[0,0,0],[0,0,0],[0,0,0],[1,1,0]]).float()
        edge_index_1 = torch.tensor([[2],[3]]).long()
        self.data0 = Data(x=env_x_1,edge_index=edge_index_1)
        env_x_2 = torch.tensor([[0,0,1],[0,0,1],[0,0,1],[1,1,1]]).float()
        edge_index_2 = torch.tensor([[2],[3]]).long()
        self.data1 = Data(x=env_x_2,edge_index=edge_index_2)
        self.num_envs = num_envs
        self.global_onturn = "m"
        self.lengths = np.zeros(num_envs)
        self.returns = np.zeros(num_envs)

    def observe(self) -> List[Data]:
        if self.global_onturn == "m":
            return [self.data1]*self.num_envs
        else:
            return [self.data0]*self.num_envs

    @staticmethod
    def validate_actions(states:List[Data],actions:List[int]):
        return actions


    def get_valid_actions(self) -> List[np.ndarray]:
        return [np.array([2,3]) for _ in range(self.num_envs)]

    def reset(self):
        self.global_onturn = "m"
        return self.observe()

    def sample(self) -> np.ndarray:
        return np.array([np.random.choice(x) for x in self.get_valid_actions()])

    def step(self,actions):
        self.global_onturn = "m" if self.global_onturn=="b" else "b"
        rewards = np.array(actions,dtype=float)-2
        dones = np.random.choice([0,1],p=[0.98,0.02],size=len(actions)).astype(bool)
        self.returns[np.logical_not(dones)] += rewards[np.logical_not(dones)]
        self.lengths[np.logical_not(dones)]+=1
        infos = [{"episode_metrics":{"return":self.returns[i],"length":self.lengths[i]}} if dones[i] else {} for i in range(len(actions))]
        self.returns[dones] = 0
        self.lengths[dones] = 0
        return self.observe(),rewards,dones,infos

    def get_transitions(self,starting_states:np.ndarray,state_history:np.ndarray,action_history:np.ndarray,reward_history:np.ndarray,done_history:np.ndarray):
        """Converts history of s,r,d from the step function to s,r,d transitions as they will be used in RL

        This is nescessary, because we are playing a two player game and thus, the 'next' state is the state after the
        opponent made a move

        Args:
            starting_states: 1D np.ndarray of type object containing Data objects
            state_history: 2D np.ndarray of type object containing Data objects (or list of np.ndarray)
            action_history: 2D np.ndarray of type int (or list of np.ndarray)
            reward_history: 2D np.ndarray of type float (or list of np.ndarray)
            done_history: 2D np.ndarray of type bool (or list of np.ndarray)

        Returns:
            List of Transitions of shape (state,action,reward,next_state,is_done)
        """
        maker_transitions = []
        breaker_transitions = []
        last_states = starting_states
        state_history = state_history
        action_history = action_history
        reward_history = reward_history
        done_history = done_history

        last_dones = np.zeros(self.num_envs,dtype=bool)

        for i,(states,actions,rewards,dones) in enumerate(zip(state_history,action_history,reward_history,done_history)):
            if i!=0:
                is_done = np.logical_or(dones,last_dones)
                combined_reward = last_rewards#-np.logical_not(last_dones)*rewards
                # last_last_states = [x.to(device) for x in last_last_states]
                # states = [x.to(device) for x in states]
                # assert torch.all(Batch.from_data_list(last_last_states).x[:,2] == Batch.from_data_list(last_last_states).x[0,2])
                # assert torch.all(Batch.from_data_list(states).x[:,2] == Batch.from_data_list(last_last_states).x[0,2])
                if last_last_states[0].x[0,2] == 1:
                    maker_transitions.extend(list(zip(last_last_states.copy(),last_actions,combined_reward,states,is_done)))
                else:
                    breaker_transitions.extend(list(zip(last_last_states,last_actions,combined_reward,states,is_done)))


            last_last_states = last_states

            last_dones = dones
            last_states = states
            last_rewards = rewards
            last_actions = actions

        return maker_transitions,breaker_transitions

