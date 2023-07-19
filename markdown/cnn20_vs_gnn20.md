# Oh well 2.0
* Trained gao cnn and gnn with same amount of params on 20x20 hex with rainbowdqn.
* Training the Gao CNN seems to work pretty well. Beating random agent with 99.9% after 3 minutes.
* Training the GNN is difficult. Takes like 10h to beat random agent with > 99.5%.
* After 3 days of training, the 1vs1 on various board sizes is a crushing win for cnn on all board sizes > 90%.
* On the long-range dependency problems, the CNN still fails badly.
    + CNN trained 3 days on 20x20: 35 mistakes on board sizes 6 to 25, starting to make mistakes at size 8
    + GNN trained 3 days on 20x20: 6 mistakes on board sizes 6 to 25
    + (old) GNN trained 21 hours on 11x11: 2 mistakes on board sizes 6 to 25
![](/images/wandb_plots/elo_20.svg)
![](/images/wandb_plots/random_winrate.svg)

* In Imitation learning, the Gao CNN architecture still overfits significantly more than the GNN, but less than the old CNN architecture.
