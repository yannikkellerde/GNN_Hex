from torch.multiprocessing import Queue,Process,spawn
import time
import psutil
from queue import Empty
import torch.multiprocessing as multiprocessing
from graph_game.graph_tools_games import Hex_game
from typing import Tuple
from torch_geometric.data import Data,Batch
import torch
from GN0.convert_graph import convert_node_switching_game
from graph_game.utils import tempered_geometric_softmax, approximately_equal_split, approximately_equal_numbers
from torch.distributions.categorical import Categorical
from GN0.models import get_pre_defined
from torch_geometric.nn.models import GraphSAGE
from collections import deque
from torch_geometric.loader import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
import torch.nn.functional as F
from alive_progress import alive_bar,alive_it
import itertools
import numpy as np
import random
import os
from torch.utils.tensorboard import SummaryWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"
torch.multiprocessing.set_sharing_strategy('file_system')

def print_model_stats(model,hex_size):
    model.eval()
    letters = "abcdefghijklmnopqrstuvwxyz"
    game = Hex_game(hex_size)
    data = convert_node_switching_game(game.view,global_input_properties=[int(game.view.gp["m"])],need_backmap=True).to(device)
    res = model(data.x,data.edge_index).squeeze()
    something = {letters[game.board.vertex_to_board_index[game.view.vertex(data.backmap[i].item())]%hex_size]+str(game.board.vertex_to_board_index[game.view.vertex(data.backmap[i].item())]//hex_size+1):value.item() for i,value in enumerate(res) if i>1}
    for key,value in something.items():
        print(f"{key}:{value:3f}")
    model.train()



def repeated_self_training(load_model=True):
    buffer = deque(maxlen=80000)
    writer = SummaryWriter("run/MCPI")
    model = get_pre_defined("action_value").to(device)
    optimizer = Adam(model.parameters(),lr=4e-6)
    epoch = 0
    hex_size = 11
    starting_temperature = 0.5
    training_batch_size = 512
    action_batch_size = 128
    games_per_it = 300
    
    min_loss = np.inf
    temperature = starting_temperature
    if load_model:
        stuff = torch.load(os.path.join("model",f"MCPI_curriculum_{hex_size}.pt"))
        model.load_state_dict(stuff["model_state_dict"])
        model.import_norm_cache(*stuff["cache"])
        optimizer.load_state_dict(stuff["optimizer_state_dict"])
        min_loss = stuff["loss"]
        epoch = stuff["epoch"]
        temperature = stuff["temperature"].item()
    policy = get_nn_tempered_policy(model,temperature)
    print(f"Loaded model temperature:{temperature}, loss:{min_loss}, epoch:{epoch}")
    # policy = get_nn_tempered_policy(model,1)
    last_loss_improvement = 0
    while 1:
        epoch+=1
        last_loss_improvement+=1
        model.eval()
        training_data = single_process_actor(policy,action_batch_size,games_per_it,hex_size)
        print("generated",len(training_data),"new samples")
        buffer.extend(training_data)
        model.train()
        loss = train(buffer,model,optimizer,training_batch_size,1)
        writer.add_scalar("Train/loss",loss,epoch)
        print("avg loss",loss)
        print_model_stats(model,hex_size)
        if loss < min_loss:
            last_loss_improvement = 0
            print(f"Saving new best model {loss}<{min_loss}")
            min_loss = loss
            temperature = min(min_loss*0.5,temperature)
            policy = get_nn_tempered_policy(model,temperature)
            save_dic = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'temperature':temperature
            }
            if hasattr(model,"supports_cache") and model.supports_cache:
                save_dic["cache"] = model.export_norm_cache()
            torch.save(save_dic, os.path.join("model",f"MCPI_curriculum_{hex_size}.pt"))
            if loss < 0.04:
                hex_size+=1
                print(f"Increasing hex size to {hex_size}")
                min_loss = np.inf
                temperature = starting_temperature
                policy = get_nn_tempered_policy(model,temperature)
        if last_loss_improvement%20 == 19:
            temperature *= 0.95
            print("Reducing temperature to",temperature)
            policy = get_nn_tempered_policy(model,temperature)


def train(data,model,optimizer,batch_size,epochs):
    criterion = MSELoss()
    random.shuffle(data)
    train_data = list(itertools.islice(data,0,int(len(data)*0.9)))
    val_data = list(itertools.islice(data,int(len(data)*0.9),len(data)))
    train_loader = DataLoader(train_data,batch_size,shuffle=True)
    val_loader = DataLoader(val_data,batch_size)
    model.eval()
    val_losses = []
    with torch.no_grad():
        with alive_bar(len(val_loader),title="eval") as bar:
            for batch in val_loader:
                action_values = model(batch.x,batch.edge_index).squeeze()
                mask = batch.y!=0
                loss = criterion(action_values[mask],batch.y[mask])
                val_losses.append(loss)
                bar()
    print("starting val loss",sum(val_losses)/len(val_losses))

    model.train()
    losses = []
    for epoch in range(epochs):
        some_losses = []
        with alive_bar(len(train_loader),title="train") as bar:
            for i,batch in enumerate(train_loader):
                if i==len(train_loader)-1 and hasattr(model,"supports_cache") and model.supports_cache:
                    action_values = model(batch.x,batch.edge_index,set_cache=True).squeeze()
                else:
                    action_values = model(batch.x,batch.edge_index).squeeze()
                mask = batch.y!=0
                loss = criterion(action_values[mask],batch.y[mask])
                some_losses.append(loss)
                loss.backward()
                optimizer.step()
                bar()
        print(sum(some_losses)/len(some_losses))
        losses.extend(some_losses)

    model.eval()
    val_losses = []
    with torch.no_grad():
        with alive_bar(len(val_loader),title="eval") as bar:
            for batch in val_loader:
                action_values = model(batch.x,batch.edge_index).squeeze()
                mask = batch.y!=0
                loss = criterion(action_values[mask],batch.y[mask])
                val_losses.append(loss)
                bar()
    print("later val loss",sum(val_losses)/len(val_losses))
    model.train()

    return sum(losses)/len(losses)


def single_process_actor(policy,total_game_instances,total_games_to_play,hex_size,gamma=1):
    batch_size = total_game_instances
    games = [Hex_game(hex_size) for _ in range(total_game_instances)]
    game_data_history = [[] for _ in range(total_game_instances)]
    collection = []
    all_training_data = []

    for i,game in enumerate(games):
        data = convert_node_switching_game(game.view, global_input_properties=[int(game.view.gp["m"])],need_backmap=True)
        game_data_history[i].append([data])
        collection.append(data)


    games_started = len(games)
    games_finished = 0
    if len(collection) == batch_size:
        batch = Batch.from_data_list(collection)
        # actions = policy(batch)
        actions = torch.zeros(len(collection)).long()
        for r in range(len(collection)):
            actions[r] = random.randint(2,len(collection[r].x)-1)
        collection = []
    else:
        raise Exception("what?")
    
    with alive_bar(total_games_to_play) as bar:
        while games_finished<total_games_to_play:
            to_del = []
            randoms = []
            for game_id,action in enumerate(actions):
                game_data_history[game_id][-1].append(action)
                action = game_data_history[game_id][-1][0].backmap[action].item()
                game = games[game_id]
                game.make_move(action,remove_dead_and_captured=True)
                winner = game.who_won()
                if winner is not None:
                    multiplicator = -1 if winner=="b" else 1
                    for i in range(len(game_data_history[game_id])-1,-1,-1):
                        dat = game_data_history[game_id][i][0]
                        tens = torch.zeros(dat.x.size(0))
                        tens[game_data_history[game_id][i][1]] = multiplicator*gamma**i
                        setattr(dat,"y",tens)
                        all_training_data.append(dat.to(device))
                    game_data_history[game_id]=[]
                    bar()
                    games_finished+=1
                    if games_started<total_games_to_play:
                        games_started+=1
                        game = games[game_id] = Hex_game(hex_size)
                        randoms.append(game_id)
                    else:
                        to_del.append(game_id)
                        continue

                data = convert_node_switching_game(game.view, global_input_properties=[int(game.view.gp["m"])],need_backmap=True)
                game_data_history[game_id].append([data])
                collection.append(data)
            for d in reversed(to_del):
                del games[d]
                del game_data_history[d]

            batch_size = min(batch_size,len(games))
            if batch_size==0:
                break

            if len(collection) == batch_size:
                batch = Batch.from_data_list(collection)
                actions = policy(batch)
                for r in randoms:
                    actions[r] = random.randint(2,len(collection[r].x)-1)
                collection = []
            else:
                raise Exception("what?")
    return all_training_data



def evaluater(policy,batch_size,total_game_instances,eval_queue:Queue,action_queues:Tuple[Queue]):
    """
    eval_queue contains tuples with 
        0. output_queue id
        1. game id
        2. data.x
        3. data.edge_index
        4. data.backmap
    """
    collection = []
    metadata = []
    all_transitions = []
    closed_games = 0
    while 1:
        eval_stuff = eval_queue.get(block=True)
        if eval_stuff is None:
            closed_games+=1
            if closed_games==total_game_instances:
                print("Done eval")
                break
            else:
                batch_size = min(batch_size,total_game_instances-closed_games)
                print("Found closed game, changing batch size to",batch_size)
        else:   
            if len(eval_stuff) == 3:
                all_transitions.append(eval_stuff)
                continue
            data = Data(x=eval_stuff[2],edge_index=eval_stuff[3],backmap=eval_stuff[4])
            metadata.append(eval_stuff[:2])
            collection.append(data)
        if len(collection) == batch_size:
            batch = Batch.from_data_list(collection)
            actions = policy(batch)
            for (action_queue_id,game_id),action in zip(metadata,actions):
                action_queues[action_queue_id].put((game_id,action))
                # print("putting into output queue",output_queue_id)
            collection = []
            metadata = []
    eval_queue.close()
    return all_transitions

def game_player(eval_queue, action_queue, num_game_instances, total_games_to_play, hex_size, process_id, gamma=1):
    games = [Hex_game(hex_size) for _ in range(num_game_instances)]
    game_data_history = [[] for _ in range(num_game_instances)]
    for i,game in enumerate(games):
        data = convert_node_switching_game(game.view, global_input_properties=[int(game.view.gp["m"])],need_backmap=True)
        game_data_history[i].append([data])
        eval_queue.put((process_id,i,data.x,data.edge_index,data.backmap))

    games_started = len(games)
    games_finished = 0
    
    while games_finished<total_games_to_play:
        game_id,action = action_queue.get(block=True,timeout=2)
        game_data_history[game_id][-1].append(action)
        game = games[game_id]
        game.make_move(action,remove_dead_and_captured=True)
        winner = game.who_won()
        if winner is not None:
            multiplicator = -1 if winner=="b" else 1
            for i,entry in enumerate(reversed(game_data_history[game_id])):
                entry.append(multiplicator*gamma**i)
            for entry in game_data_history[game_id]:
                eval_queue.put(entry)
            game_data_history[game_id]=[]
            games_finished+=1
            print("game finished",process_id,games_finished,total_games_to_play)
            if games_started<total_games_to_play:
                games_started+=1
                game = games[game_id] = Hex_game(hex_size)
            else:
                eval_queue.put(None)
                continue

        data = convert_node_switching_game(game.view, global_input_properties=[int(game.view.gp["m"])],need_backmap=True)
        game_data_history[game_id].append([data])
        eval_queue.put((process_id,game_id,data.x,data.edge_index,data.backmap))

    action_queue.close()
    print("Done",process_id)

def get_nn_tempered_policy(nn:torch.nn.Module,temperature):
    def tempered_policy(batch):
        with torch.no_grad():
            batch = batch.to(device)
            if type(batch)==Data:
                batch.ptr = [0,len(batch.x)]
            action_values = nn(batch.x,batch.edge_index).squeeze()
            action_values *= (batch.x[:,2]*2-1)
            actions = []
            for start,fin in zip(batch.ptr,batch.ptr[1:]):  # This isn't great, but I didn't find any method for sampling in pytorch_scatter. Maybe need to implement myself at some point.
                action_part = action_values[start+2:fin]
                if len(action_part)==1:
                    actions.append(torch.tensor([2]))
                    continue
                if temperature==0:
                    action = torch.argmax(action_part)+2
                    actions.append(action)
                    continue
                prob_part = F.softmax(action_part/temperature, dim=0)
                try:
                    distrib = Categorical(prob_part.squeeze())
                except ValueError:
                    print(prob_part)
                    print(action_part)
                    print(action_part/temperature)
                    print(temperature)
                    print(F.softmax(action_part/temperature))
                    print(len(action_values))
                    print(start,fin)
                    raise ValueError
                sample = distrib.sample()
                action = sample+2
                actions.append(action.item())
        return actions
    return tempered_policy

def collect_transitions(policy,num_game_instances,batch_size,total_games):
    cpus = multiprocessing.cpu_count()
    procs = cpus-2
    print(procs,"number of processes")
    eval_queue = Queue()
    # output_queue = Queue()
    action_queues = [Queue() for _ in range(procs)]
    # eval_proc = Process(target=evaluater,args=[policy,batch_size,num_game_instances,eval_queue,action_queues])
    proc_game_instances = approximately_equal_numbers(num_game_instances,procs)
    proc_total_games = approximately_equal_numbers(total_games,procs)
    actor_processes = [Process(target=game_player,args=[eval_queue,action_queue,proc_game_instances[i],proc_total_games[i],11,i]) for i,action_queue in enumerate(action_queues)]
    actor_processes = spawn(fn=game_player)
    for actor_process in actor_processes:
        actor_process.start()
    all_transitions = evaluater(policy,batch_size,num_game_instances,eval_queue,action_queues)
    # eval_proc.start()
    # all_transitions = []
    while 1:
        try:
            all_transitions.append(output_queue.get(block=False))
        except Empty:
            eval_proc.join(0.01)
            if eval_proc.exitcode is not None:
                break
    # eval_proc.join()
    # print("joined eval proc")
    # eval_proc.close()
    for actor_process in actor_processes:
        actor_process.join()
        print("joined actor proc")
        actor_process.close()
    return all_transitions
    
if __name__=="__main__":
    repeated_self_training(load_model=True)
    # model = get_pre_defined("action_value").to(device)
    # policy = get_nn_tempered_policy(model,1)
    # start = time.perf_counter()
    # print(single_process_actor(policy,128,256,11))
    # print(time.perf_counter()-start)
    # print(collect_transitions(policy,256,128,512))


