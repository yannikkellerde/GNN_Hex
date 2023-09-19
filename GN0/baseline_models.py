import torch
import torch.nn

class ResBlock(torch.nn.Module):
    def __init__(self,num_filters,batch_norm=True):
        super().__init__()
        self.block = torch.nn.Sequential(
                torch.nn.BatchNorm2d(num_filters) if batch_norm else torch.nn.Identity(),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=num_filters,out_channels=32,kernel_size=3,padding="same"),
                torch.nn.BatchNorm2d(32) if batch_norm else torch.nn.Identity(),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=num_filters,out_channels=32,kernel_size=3,padding="same"),
        )
    def forward(self,x):
        return self.block(x)+x

class Softmax2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x:torch.Tensor):
        y = x.reshape((x.shape[0],-1))
        y = torch.softmax(y,1)
        y = y.reshape(x.shape)
        return y


class Gao_baseline(torch.nn.Module):
    # http://webdocs.cs.ualberta.ca/~hayward/papers/transferable.pdf
    def __init__(self,num_res_blocks=10,num_planes=4,batch_norm=True):
        super().__init__()
        self.initial_conv = torch.nn.Conv2d(num_planes,32,3,padding="same")
        self.final_conv_acts = None
        self.final_conv_grads = None
        self.res_blocks = torch.nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResBlock(32,batch_norm=batch_norm))

        self.q_head = torch.nn.Sequential(
                torch.nn.Conv2d(32,1,1),
                torch.nn.Tanh()
        )

        self.p_head = torch.nn.Sequential(
                torch.nn.Conv2d(32,1,1),
                Softmax2d()
        )

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self,x,**kwargs):
        return self.q(x)
        # x = self.initial_conv(x)
        # for block in self.res_blocks:
        #     x = block(x)

        # q = self.q_head(x).squeeze(1)
        # p = self.p_head(x).squeeze(1)
        # return q,p

    def q(self,x,**kwargs):
        x = self.initial_conv(x)
        for block in self.res_blocks:
            x = block(x)

        self.final_conv_acts = x
        self.final_conv_acts.register_hook(self.activations_hook)

        q = self.q_head(x).squeeze(1)
        return q

    def p(self,x):
        x = self.initial_conv(x)
        for block in self.res_blocks:
            x = block(x)

        p = self.p_head(x).squeeze(1)
        return p
