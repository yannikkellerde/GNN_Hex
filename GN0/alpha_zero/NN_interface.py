from graph_game.shannon_node_switching_game import Node_switching_game
from GN0.util.convert_graph import convert_node_switching_game
import torch
from GN0.alpha_zero.replay_buffer import ReplayBuffer
from tqdm import trange, tqdm
import torch.nn.functional as F
from GN0.util.util import AverageMeter
from torch.distributions import Categorical
from typing import List
from torch_geometric.data import Batch
from collections import defaultdict
from time import perf_counter

class NNetWrapper():
    def __init__(self,nnet,device="cpu",lr=0.0001,weight_decay=1e-5):
        self.nnet:torch.nn.Module = nnet
        self.device = device
        self.optimizer = torch.optim.Adam(self.nnet.parameters(),lr=lr,weight_decay=weight_decay)
        self.batch_size = 128
        self.timers = defaultdict(list)

    def train(self,maker_buffer:ReplayBuffer,breaker_buffer:ReplayBuffer,num_epochs):
        self.nnet.train()
        all_pi_losses = AverageMeter()
        all_v_losses = AverageMeter()
        total_losses = AverageMeter()
        for epoch in range(num_epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            maker_loader = maker_buffer.get_data_loader(batch_size=self.batch_size)
            breaker_loader = breaker_buffer.get_data_loader(batch_size=self.batch_size)
            for loader in (maker_loader,breaker_loader):
                pi_losses = AverageMeter()
                v_losses = AverageMeter()
                t_losses = AverageMeter()
                for batch in tqdm(loader,total=len(loader)):
                    pi,v = self.nnet(batch)
                    l_pi = self.loss_pi(pi,batch.pi)
                    l_v = self.loss_v(v,batch.v)
                    total_loss = l_pi + l_v
                    pi_losses.update(l_pi)
                    v_losses.update(l_v)
                    t_losses.update(total_loss)

                    total_loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                all_pi_losses.update(pi_losses.avg)
                all_v_losses.update(v_losses.avg)
                total_losses.update(t_losses.avg)
        return all_pi_losses.avg,all_v_losses.avg,total_losses.avg

    def predict(self,data):
        with torch.no_grad():
            pi,v = self.nnet(data.to(self.device))
            pi = torch.exp(pi)  # Log-softmax to probability distribution
        return pi,v


    def loss_pi(self, targets, outputs):
        # return F.nll_loss(outputs,targets)  # This doesn't work for float targets
        return -torch.sum(targets * outputs) / targets.size()[0] # This should be equivalent

    def loss_v(self, targets, outputs):
        return F.mse_loss(targets,outputs)

    def predict_game(self,game):
        data = convert_node_switching_game(game.view,global_input_properties=[int(game.view.gp["m"])]).to(self.device)
        return self.predict(data)

    def predict_for_mcts(self,game:Node_switching_game):
        data = convert_node_switching_game(game.view,global_input_properties=[int(game.view.gp["m"])],need_backmap=True).to(self.device)
        policy,value = self.predict(data)
        return policy,value

    def predict_many_for_mcts(self,games:List[Node_switching_game]):
        # start = perf_counter()
        datas = [convert_node_switching_game(game.view,global_input_properties=[int(game.view.gp["m"])],need_backmap=True).to(self.device) for game in games]
        batch = Batch.from_data_list(datas)
        # self.timers["convert graph"].append(perf_counter()-start)
        # start = perf_counter()
        policy,value = self.predict(batch)
        # self.timers["nn_prediction"].append(perf_counter()-start)
        # start = perf_counter()
        policies = [policy[start:finish] for start,finish in zip(batch.ptr,batch.ptr[1:])]
        # self.timers["select_policy"].append(perf_counter()-start)
        if len(batch.ptr)==2:
            return [[policies[0],value]]
        else:
            return zip(policies,value)

    def choose_move(self,game:Node_switching_game,temperature=0):
        data = convert_node_switching_game(game.view,global_input_properties=[int(game.view.gp["m"])],need_backmap=True).to(self.device)
        policy,value = self.predict(data)
        moves = [int(data.backmap[x]) for x in range(len(policy))]
        print(moves,game.get_actions())
        if temperature == 0:
            return moves[torch.argmax(policy)]
        else:
            dist = Categorical(policy)
            return moves[int(dist.sample().item())]

    def be_evaluater(self,game:Node_switching_game,temperature=1):
        data = convert_node_switching_game(game.view,global_input_properties=[int(game.view.gp["m"])],need_backmap=True).to(self.device)
        policy,value = self.predict(data)
        moves = [int(data.backmap[x]) for x in range(len(policy))]
        vprop = game.view.new_vertex_property("double")
        for m,p in zip(moves,policy):
            vprop[m] = p
        print("Value:",value)
        return vprop

    def choose_moves(self,games:List[Node_switching_game],temperature=0):
        datas = [convert_node_switching_game(game.view,global_input_properties=[int(game.view.gp["m"])],need_backmap=True).to(self.device) for game in games]
        batch = Batch.from_data_list(datas)
        policy,value = self.predict(batch)
        actions = []
        for i,(start,fin) in enumerate(zip(batch.ptr,batch.ptr[1:])):
            policy_part = policy[start:fin]
            if temperature==0:
                action = torch.argmax(policy_part)
            else:
                try:
                    distrib = Categorical(policy_part.squeeze())
                except ValueError:
                    raise ValueError
                sample = distrib.sample()
                action = sample
            actions.append(action.item())
        actions = [datas[i].backmap[actions[i]].item() for i in range(len(actions))]
        return actions


    def save_checkpoint(self, path:str, args=None):
        torch.save({"state_dict": self.nnet.state_dict(),
                    "optim_dict": self.optimizer.state_dict(),
                    "args": args},path)

    def load_checkpoint(self, path:str):
        checkpoint = torch.load(path,map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_dict'])


