#include <string>
#include "nn_api.h"
#include "shannon_node_switching_game.h"
#include "agents/mctsagent.h"
#include "agents/rawnetagent.h"

void playmode(MCTSAgent * mctsAgent, RawNetAgent * rawAgent, SearchLimits * searchLimits, EvalInfo * evalInfo);
