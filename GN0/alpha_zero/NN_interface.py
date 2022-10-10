from graph_game.shannon_node_switching_game import Node_switching_game
from GN0.util.convert_graph import convert_node_switching_game
import torch
from GN0.alpha_zero.replay_buffer import ReplayBuffer
from tqdm import trange, tqdm
import torch.nn.functional as F
from GN0.util.util import AverageMeter

class NNetWrapper():
    def __init__(self,nnet,device="cpu"):
        self.nnet = nnet
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
            return self.nnet(data)

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
        return moves,policy,value

    def save_checkpoint(self, path:str):
        torch.save({"state_dict": self.nnet.state_dict(),
                    "optim_dict": self.optimizer.state_dict()},path)

    def load_checkpoint(self, path:str):
        checkpoint = torch.load(path,map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_dict'])


