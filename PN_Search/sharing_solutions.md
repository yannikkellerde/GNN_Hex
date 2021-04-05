# Sharing solutions on yannikkeller.de
## Sharing games
If you have created a game on your local GABOR installation, you will find a .json file with the games name in [/json_games](/json_games). Click the `Upload game` button on [http://python.yannikkeller.de/solver/](http://python.yannikkeller.de/solver/) (You may need to create an accout first) and upload that json file.

## Sharing proofsets
GABORs solutions to games are stored in *proofsets*. These are python sets that store the hashes of all proven/disproven positions. If you have solved a game using GABOR, you should find your a folder named after your proofset at [/proofsets](/proofsets). It contains 4 files: `bd.pkl`, `bp.pkl`, `wd.pkl`, `wp.pkl`. On [http://python.yannikkeller.de/solver/](http://python.yannikkeller.de/solver/), click on `Upload Proofset` and select those 4 files.

## Solving games
To solve a game, you will need to install GABOR from this repository. Check [README.md](/README.md) for instructions. In the browser interface, you can then analyze a game and click `Solve game for black/white`.

## Only one proofset per account?
You can only upload one proofset to [http://python.yannikkeller.de/solver/](http://python.yannikkeller.de/solver/) at a time. This does not mean that you can only upload the solution to one game though. GABORs proofsets are non-interfering. You can store the solution to multiple games in a single proofset. This will even be benificial, as GABOR will be able to recognize if it get's into an endgame, that is equivalent to an endgame it has already solved while solving another game.