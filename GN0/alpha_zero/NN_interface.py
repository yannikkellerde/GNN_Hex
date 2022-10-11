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

class NNetWrapper():
    def __init__(self,nnet,device="cpu"):
        self.nnet:torch.nn.Module = nnet
        self.device = device
        self.optimizer = torch.optim.Adam(self.nnet.parameters())
        self.batch_size = 128

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

                    total_loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

    def predict(self,data):
        with torch.no_grad():
            pi,v = self.nnet(data)
            pi = torch.exp(pi)  # Log-softmax to probability distribution

            # Now care about not picking the terminal nodes
            if isinstance(data,Batch):
                for (start,fin) in zip(data.ptr,data.ptr[1:]):
                    pi[start:start+2] = 0
                    pi[start+2:fin]/=torch.sum(pi[start+2:fin])
            else:
                pi[0:2] = 0
                pi[2:]/=torch.sum(pi[2:])
        return pi,v


    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return F.mse_loss(targets,outputs)

    def predict_game(self,game):
        data = convert_node_switching_game(game.view,global_input_properties=int(game.view.gp["m"])).to(self.device)
        return self.predict(data)

    def predict_for_mcts(self,game:Node_switching_game):
        data = convert_node_switching_game(game.view,global_input_properties=int(game.view.gp["m"]),need_backmap=True).to(self.device)
        policy,value = self.predict(data)
        moves = [int(data.backmap[x]) for x in range(2,len(policy))]
        return moves,policy[2:],value

    def choose_move(self,game:Node_switching_game,temperature=0):
        moves,policy,_ = self.predict_for_mcts(game)

        if temperature == 0:
            return moves[torch.argmax(policy)]
        else:
            dist = Categorical(policy)
            return moves[int(dist.sample().item())]

    def choose_moves(self,games:List[Node_switching_game],temperature=0):
        datas = [convert_node_switching_game(game.view,global_input_properties=int(game.view.gp["m"]),need_backmap=True).to(self.device) for game in games]
        batch = Batch(datas)
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
            actions.append(action)
        return actions


    def save_checkpoint(self, path:str):
        torch.save({"state_dict": self.nnet.state_dict(),
                    "optim_dict": self.optimizer.state_dict()},path)

    def load_checkpoint(self, path:str):
        checkpoint = torch.load(path,map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_dict'])


