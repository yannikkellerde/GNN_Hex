#include <torch/script.h>
#include <ATen/ATen.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include "../shannon_node_switching_game.h"
#include "../util.h"
#include "../nn_api.h"
#include <blaze/Math.h>

#include <iostream>
#include <memory>

void test_torch_script(string fname);
