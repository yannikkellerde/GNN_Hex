import json
import os

table_path = "/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/Rainbow/wandb/run-20220924_093459-23kjabmr/files/media/table/ev/rating_table_38100_87ac68f0e7970015c449.table.json"
starting_frame = 16000000
cleaning_path = "/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/Rainbow/checkpoints/azure-snowball-157"

with open(table_path,"r") as f:
    table = json.load(f)

to_spare = [int(x[0].split("_")[0]) for x in table["data"][:30]]

would_del = []
for fname in os.listdir(cleaning_path):
    number = int(fname.split("_")[1].split(".")[0])
    if number > starting_frame and number not in to_spare:
        os.remove(os.path.join(cleaning_path,fname))


