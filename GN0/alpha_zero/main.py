from GN0.alpha_zero.train_alpha_zero import Trainer
from GN0.alpha_zero.argp import read_args 
from GN0.models import get_pre_defined
from GN0.alpha_zero.NN_interface import NNetWrapper
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: cuda not avaliabe, using cpu")

args = read_args()
nnet_creation_func = lambda :NNetWrapper(nnet=get_pre_defined("policy_value",args).to(device),device=device,lr=args.lr)
trainer = Trainer(nnet_creation_func=nnet_creation_func,args=args,device=device)
trainer.learn()
