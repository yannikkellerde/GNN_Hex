# Maybe final big implementation part
## Before:
1 cpu thread, only one game at once, batch size 8 to prevent exploding virtual loss  
Experiment: 100 Nodes per MCTS, Hex size 5
|        statistic         |       value      |
| ------------------------ | ---------------- |
|            avg batch size|            5.7729|
|             games per sec|            1.6964|
|           samples per sec|            15.776|


|      task      |  total time (μs) |
| -------------- | ---------------- |
|         collate|            129001|
|   convert graph|            495468|
|       make move|           1178008|
|      nn predict|           3522885|

+ nn prediction is biggest bottleneck
+ Only 20% GPU util

## Now:
10 CPU threads that each play 128 games at once so we can get arbitrary batch sizes without exploding virtual loss.  
Experiment: 100 Nodes per MCTS, Hex size 5

|        statistic         |       value      |
| ------------------------ | ---------------- |
|            avg batch size|            208.27|
|             games per sec|            17.707|
|           samples per sec|            150.96|

|      task      |  total time (μs) |
| -------------- | ---------------- |
|         collate|          94125998|
|   convert graph|         211531510|
|       make move|         193284499|
|      nn predict|          99449282|

+ My graph env is biggest bottleneck
+ 10x speed improvement to previous version
+ Still only 30% gpu util... Not sure why

## Scaling up to hex size 11
10 CPU threads that each play 64 games at once with **800 Nodes** per MCTS and **Hex size 11**

|        statistic         |       value      |
| ------------------------ | ---------------- |
|            avg batch size|            317.29|
|             games per sec|           0.32724|
|           samples per sec|            11.966|

|      task      |  total time (μs) |
| -------------- | ---------------- |
|         collate|         981363271|
|   convert graph|        3422286115|
|       make move|       11382729470|
|      nn predict|         630034026|

+ On 11x11 Hex, making moves and converting the graph are by far the biggest bottleneck. Factor 20 !!
	- It is surprising that nn prediction takes so few time. I guess large batch sizes are processed very efficiently.
	- I could try to save some make-move time by caching the graphs at each node. However, with 10\*64 MCTS running in parallel, that might be too much on the ram.
+ Is 1 game every 3 seconds fast enough to get enough training data? I am not sure...
