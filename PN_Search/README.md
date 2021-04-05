<p align="center">
  <a href="#installation">Installation</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="#usage">Usage</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="#why-graph-based">Why Graph Based?</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="#algorithm">Algorithm</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="#games-solved-using-gabor">Games solved using GABOR</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="/sharing_solutions.md">Sharing solutions</a>
</p>

# GABOR
## What is GABOR?
**G**r**A**ph **B**ased b**O**ard game solve**R** (GABOR) is a solving algorithm/program for a class of board games. That class is board games, where 2 players take alternating turns occupying squares. A player wins, if he manages to occupy all squares of a specific pattern. Real-world examples for this type of game are [tic tac toe](https://en.wikipedia.org/wiki/Tic-Tac-Toe), [Qango](http://qango.de/index.html?page=spiel&language=englisch) or [Go-Moku](https://en.wikipedia.org/wiki/Gomoku). GABOR can, without modifications, deal with any game of this class, no matter the shape of the board or the layout of the winpatterns.

## Why would you want to use GABOR?
On [http://python.yannikkeller.de/solver/](http://python.yannikkeller.de/solver/), you can find a community platform for sharing and exploring games and solutions created by GABOR. If you just want to see some of GABORs findings without installing anything, you should probably head over there. However, to start solving your own games, you will have to install GABOR on your computer. You can then head back to [http://python.yannikkeller.de/solver/](http://python.yannikkeller.de/solver/) and share your computed solutions with the world.

## Installation
Installation instructions are for linux systems. You should also be able to get it to work on Windows or MacOS using the [installation instructions](https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions#windows) for graph-tool.

Requirements: `python3.8+, graph-tool, flask, flask-socketio, psutil`  
GABOR depends on [graph-tool](https://graph-tool.skewed.de/), which is easiest installed via conda. If you have conda ready, you can install from [environment.yml](/environment.yml) via `conda env create -f environment.yml`. Activate the env via `conda activate GABOR`

If the install from `environment.yml` fails, you can try installing manually with the commands  
`conda install -c conda-forge graph-tool`  
`conda activate gt`   
`pip install flask flask_socketio psutil`

## Usage
Run `export FLASK_APP=flask_server.py;flask run` from the projects root. Head over to [localhost:5000](http://localhost:5000/) in your favorite browser. To start solving a game, click the on a `Analyze` link and then `Solve game for black/white`. GABOR performs binary game evaluation, so when solving for black, you are checking if the game is won for black or not (White wins or draw). To fully solve a game, you will need to try and solve for black as well as for white.

## Why Graph Based?
GABOR transforms any board game into a graph. Each winpattern will be a vertex and each square will be a vertex. Squares that are part of a winpattern share an edge with that winpattern. When a player occupies a square, that square will be removed from the graph and any connected winpattern vertex will be colored in that players color. If the winpattern vertex was already in the players color and has no edges left, the player wins. If the winpattern was already colored in the opponents player color, the winpattern vertex is removed. Any square vertex left without edges as a result is removed.

That way of representing the board game has the following advantages:
* Uninteresting squares (Ones that correspond to no more winpatterns) are automatically removed
* Heuristic move sorting is very easy (prefer square verticies, that are connected to to many winpattern verticies of low degree)
* We can exploit graph isomorphisms to avoid computing theoretically equivalent positions multiple times. GABOR uses Weisfeiler-Lehman based graph hashing for that. We can even recognize and exploit inter-game isomorphisms. There might be endgames in one game, that are isomorphic to endgames in another game.
* We can implement a general [threat-space search](https://www.researchgate.net/publication/2252447_Go-Moku_and_Threat-Space_Search) algorithm (Threats are square verticies that are neighbours to a winpattern vertex of degree 2)

## Algorithm
Inspired by the paper [Go-Moku and Threat-Space Search](https://www.researchgate.net/publication/2252447_Go-Moku_and_Threat-Space_Search), GABOR uses a combination of [Proof-Number search](https://doi.org/10.1016/0004-3702(94)90004-3) and Threat-Space Search to solve games. A big advantage of GABORs graph representation are the transpositions/symetries/graph-isomorphisms. Thus, GABORs search structure is not a tree, as in the original Proof-Number search, but instead a directed acyclic graph (DAG). I used the practical algorithm described [here](https://pdfs.semanticscholar.org/86f5/1429a19cfc76e9d42f28b93c62e978c816a0.pdf) to make PN-search work on my DAG. Also, GABOR's implementation of the threat-space is not specifically built for a single game, as in the original papers implementation, but exploits the graph structure to know what a threat is and what not.

## Games solved using GABOR
To the best of my knowledge, GABOR is the first program to solve [Qango](https://www.yucata.de/en/Rules/QANGO), in all it's shapes and forms. You can view all proofs online at [http://python.yannikkeller.de/solver/](http://python.yannikkeller.de/solver/)

## Author
Yannik Keller
