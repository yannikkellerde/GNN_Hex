# NN prediction is the bottleneck

      task      |  total time (μs)
----------------+------------------
         collate|           2298880
   convert\_graph|          13527483
      file_write|              2522
       make move|          16847482
      nn predict|          46287049
    save samples|             11337
        selfplay|          88857962
