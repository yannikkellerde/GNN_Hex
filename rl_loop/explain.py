from dataset_loader import get_loader
from GN0.util.util import downsample_cnn_outputs, downsample_gao_outputs
from torch_geometric.data import Batch
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler
from GN0.torch_script_models import get_current_model 
from GN0.models import get_pre_defined
import torch
from GN0.util.convert_graph import convert_node_switching_game_back, convert_node_switching_game
import torch.nn.functional as F
from graph_game.shannon_node_switching_game import Node_switching_game
from rl_loop.trainer_agent_pytorch import SoftCrossEntropyLoss
from GN0.tests.test_models_on_long_range_task import fill_game_with_long_range_position, load_a_model
from graph_game.graph_tools_games import Hex_game

device="cpu"

def plot_explaination(model,data,optimizer,nsg=None):
    if nsg is None:
        graph, vmap = convert_node_switching_game_back(data)
        nsg = Node_switching_game.from_graph(graph)
    data = data.to(device)
    optimizer.zero_grad()
    input = torch.autograd.Variable(data.x,requires_grad=True)
    out = model(input,data.edge_index,data.batch,data.ptr)
    print(out.shape)
    scel = torch.nn.MSELoss()
    # print(data.x.shape,data.policy.shape,out[0].shape)
    p_loss = scel(out, data.y.float().squeeze())
    # p_loss = scel(out[0], data.y.float())
    # v_loss = F.mse_loss(out[1], data.global_y.float())
    p_loss.backward()
    print(p_loss)
    # v_loss.backward()
    input_grads = input.grad
    # print(v_loss,torch.max(input.grad))
    # print(out[1],data.global_y.float(),v_loss)
    # print(model.final_conv_acts)
    grad_cam_weights = np.array(grad_cam(model.final_conv_acts,model.final_conv_grads))
    saliency_map_weights = saliency_map(input_grads)
    scaled_grad_cam_weights = MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(grad_cam_weights).reshape(-1, 1)).reshape(-1, )
    scaled_saliency_map_weights = MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(saliency_map_weights).reshape(-1, 1)).reshape(-1, )

    policy = nsg.view.new_vertex_property("double")
    policy.fa = out.detach().numpy()

    saliency = nsg.view.new_vertex_property("double")
    saliency.fa = scaled_saliency_map_weights
    nsg.draw_me(vprop1=policy,vprop3=saliency,decimal_places=2,fname="saliency.pdf",layout="grid")

    gc = nsg.view.new_vertex_property("double")
    gc.fa = scaled_grad_cam_weights
    nsg.draw_me(vprop1=policy,vprop3=gc,decimal_places=2,fname="grad_cam.pdf",layout="grid")

def plot_cnn_explaination(model,data,optimizer,cnn_policy,nsg):
    data = data.unsqueeze(0).to(device).float()
    cnn_policy = torch.from_numpy(cnn_policy).to(device).float()
    optimizer.zero_grad()
    input = torch.autograd.Variable(data,requires_grad=True)
    board_outputs = downsample_gao_outputs(model(data).reshape(data.shape[0],-1),nsg.board.size).squeeze()
    loss = F.mse_loss(board_outputs,cnn_policy)
    print(loss,board_outputs)
    loss.backward()
    input_grads = input.grad
    # final_conv_grads = downsample_gao_outputs(model.final_conv_grads.mean(dim=[1]).reshape(data.shape[0],-1),nsg.board.size).squeeze()
    # final_conv_acts = downsample_gao_outputs(model.final_conv_acts.mean(dim=[1]).reshape(data.shape[0],-1),nsg.board.size).squeeze()
    heatmap = grad_cam_cnn(model.final_conv_acts,model.final_conv_grads)
    heatmap = downsample_gao_outputs(heatmap[np.newaxis,...],nsg.board.size).squeeze()
    print(heatmap)
    nsg.board.matplotlib_me(label_numbers=board_outputs,fontsize=30)
    plt.show()

def grad_cam_cnn(final_conv_acts,final_conv_grads):

    pooled_gradients = torch.mean(final_conv_grads, dim=[0, 2, 3])

    # weight the channels by corresponding gradients
    for i in range(final_conv_acts.shape[1]):
        final_conv_acts[:, i, :, :] *= pooled_gradients[i]
        
    # average the channels of the activations
    heatmap = torch.mean(final_conv_acts, dim=1).squeeze()

    scaled_heatmap = MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(heatmap.detach()).reshape(-1, 1)).reshape(-1, )

    return scaled_heatmap

    

def saliency_map(input_grads):
    node_saliency_map = []
    for n in range(input_grads.shape[0]): # nth node
        node_grads = input_grads[n,:]
        node_saliency = torch.norm(F.relu(node_grads)).item()
        node_saliency_map.append(node_saliency)
    return torch.tensor(node_saliency_map)

def grad_cam(final_conv_acts, final_conv_grads):
    # print('grad_cam')
    node_heat_map = []
    alphas = torch.mean(final_conv_grads, dim=0) # mean gradient for each feature (512x1)
    for n in range(final_conv_acts.shape[0]): # nth node
        node_heat = F.relu(alphas @ final_conv_acts[n]).item()
        node_heat_map.append(node_heat)
    return node_heat_map

def create_long_range_example(hex_size,is_positive,defender="b"):
    attacker = "m" if defender=="b" else "b"
    game = Hex_game(hex_size)
    game.board_callback = game.board.graph_callback
    fill_game_with_long_range_position(game,hex_size,defender)
    good_policy = game.view.new_vertex_property("double")
    good_policy.fa = -1
    # good_policy.fa = 0
    good_board_policy = np.ones(game.board.squares)*(-1)
    if is_positive:
        good_policy[game.board.board_index_to_vertex[0]] = 1
        good_board_policy[0] = 1
        if defender=="b":
            game.make_move(game.board.board_index_to_vertex[(hex_size-1)*hex_size],force_color="m",remove_dead_and_captured=False)
            game.make_move(game.board.board_index_to_vertex[2+hex_size*(hex_size//2)+((hex_size-2)//2)],force_color="b",remove_dead_and_captured=False)
        else:
            game.make_move(game.board.board_index_to_vertex[hex_size-1],force_color="b",remove_dead_and_captured=False)
            game.make_move(game.board.board_index_to_vertex[hex_size*2+hex_size//2+((hex_size-2)//2)*hex_size],force_color="m",remove_dead_and_captured=False)
    else:
        if defender=="b":
            good_policy[game.board.board_index_to_vertex[2+hex_size*(hex_size//2)+((hex_size-2)//2)]] = 1
            good_board_policy[2+hex_size*(hex_size//2)+((hex_size-2)//2)]=1
        else:
            good_policy[game.board.board_index_to_vertex[hex_size*2+hex_size//2+((hex_size-2)//2)*hex_size]] = 1
            good_board_policy[hex_size*2+hex_size//2+((hex_size-2)//2)*hex_size]=1

    example = convert_node_switching_game(game.view,target_vp=good_policy,global_output_properties=np.array([1]),global_input_properties=[int(game.view.gp["m"])],old_style=True)
    cnn_example = game.board.to_gao_input_planes()
    return example,cnn_example,good_board_policy,game


if __name__=="__main__":
    example,cnn_example,cnn_policy,nsg = create_long_range_example(9,False,"b")
    model = load_a_model("GN0/RainbowDQN/Rainbow/checkpoints/astral-haze-209/11/checkpoint_18294144.pt","gao")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    plot_cnn_explaination(model,cnn_example,optimizer,cnn_policy,nsg)
    exit()


    ex_batch = Batch.from_data_list([example])
    save_path = "GN0/RainbowDQN/Rainbow/checkpoints/rainbow_gnn_11x11/11/checkpoint_44085888.pt"
    stuff = torch.load(save_path)
    model = get_pre_defined("modern_two_headed",args=stuff["args"])
    model.load_state_dict(stuff["state_dict"])
    # model = get_current_model(net_type="SAGE")
    # weight_path = "model_save/gnn_imitate/model/weights-0.97080-1.84247-0.674-0.511-3658.pt"
    # stuff = torch.load(weight_path)
    # model.load_state_dict(stuff["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    plot_explaination(model,ex_batch,optimizer,nsg=nsg)
    exit()
    TrainConfig = namedtuple("TrainConfig","q_value_ratio batch_size cpu_count")
    tc = TrainConfig(q_value_ratio=0,batch_size=1,cpu_count=1)
    loader = get_loader(train_config=tc,dataset_type="train")
    for data in loader:
        plot_explaination(model,data,optimizer)
        exit()
