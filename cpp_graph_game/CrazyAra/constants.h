#include <string>

#define LOSS_VALUE -1
#define DRAW_VALUE 0
#define WIN_VALUE 1
#define PRESERVED_ITEMS 8
// Pre-initialized index when no forced win was found: 2^16 - 1
#define NO_CHECKMATE 65535
#define Q_VALUE_DIFF 0.1f
#define Q_INIT -1.0f
#define DEPTH_INIT 64
#define Q_TRANSPOS_DIFF 0.01
#define MAX_HASH_SIZE 100000000
#ifdef MODE_CHESS
#define VALUE_TO_CENTI_PARAM 1.4f
#else
#define VALUE_TO_CENTI_PARAM 1.2f
#endif
#define TIME_EXPECT_GAME_LENGTH 38
#define TIME_THRESH_MOVE_PROP_SYSTEM 35
#define TIME_PROP_MOVES_TO_GO 14
#define TIME_INCREMENT_FACTOR 0.7f
#define TIME_BUFFER_FACTOR 30
#define NONE_IDX uint16_t(-1)

#ifndef MODE_POMMERMAN
#define TERMINAL_NODE_CACHE 8192
#else
#define TERMINAL_NODE_CACHE 1
#endif

const std::string result[] = {"1/2-1/2", "1-0", "0-1"};
