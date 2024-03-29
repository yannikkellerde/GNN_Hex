## Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or    implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Source (modified) https://github.com/deepmind/graph_nets/blob/master/graph_nets/demos_tf2/models.py

"""Model architectures for the demos in TensorFlow 2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import modules
from graph_nets import utils_tf
from graph_nets.graphs import GraphsTuple
from six.moves import range
import sonnet as snt
from functools import partial
import numpy as np
from graph_tool.all import Graph
from GNZero.convert_graph import convert_graph

def make_mlp_model(latent_size=16,num_layers=2):
    """Instantiates a new MLP, followed by LayerNorm.

    The parameters of each new MLP are not shared with others generated by
    this function.

    Returns:
        A Sonnet module which contains the MLP and LayerNorm.
    """
    return snt.Sequential([
            snt.nets.MLP([latent_size] * num_layers, activate_final=True),
            snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
    ])


class MLPGraphIndependent(snt.Module):
    """GraphIndependent with MLP edge, node, and global models."""

    def __init__(self, settings, name="MLPGraphIndependent"):
        super(MLPGraphIndependent, self).__init__(name=name)
        self._network = modules.GraphIndependent(
            edge_model_fn=partial(make_mlp_model,**settings["edge_model"]),
            node_model_fn=partial(make_mlp_model,**settings["node_model"]),
            global_model_fn=partial(make_mlp_model,**settings["global_model"])
        )

    def __call__(self, inputs):
        return self._network(inputs)


class MLPGraphNetwork(snt.Module):
    """GraphNetwork with MLP edge, node, and global models."""

    def __init__(self, settings, name="MLPGraphNetwork"):
        super(MLPGraphNetwork, self).__init__(name=name)
        self._network = modules.GraphNetwork(
            edge_model_fn=partial(make_mlp_model,**settings["edge_model"]),
            node_model_fn=partial(make_mlp_model,**settings["node_model"]),
            global_model_fn=partial(make_mlp_model,**settings["global_model"])
        )

    def __call__(self, inputs):
        return self._network(inputs)


class EncodeProcessDecode(snt.Module):
    """Full encode-process-decode model.

    The model we explore includes three components:
    - An "Encoder" graph net, which independently encodes the edge, node, and
        global attributes (does not compute relations etc.).
    - A "Core" graph net, which performs N rounds of processing (message-passing)
        steps. The input to the Core is the concatenation of the Encoder's output
        and the previous output of the Core (labeled "Hidden(t)" below, where "t" is
        the processing step).
    - A "Decoder" graph net, which independently decodes the edge, node, and
        global attributes (does not compute relations etc.), on each message-passing
        step.
    """

    def __init__(self,
                 encoder_settings,
                 core_settings,
                 decoder_settings,
                 edge_fn=None,
                 node_fn=None,
                 global_fn=None,
                 name="EncodeProcessDecode"):
        super(EncodeProcessDecode, self).__init__(name=name)
        self._encoder = MLPGraphIndependent(encoder_settings)
        self._core = MLPGraphNetwork(core_settings)
        self._decoder = MLPGraphIndependent(decoder_settings)
        # Transforms the outputs into the appropriate shapes.
        self._output_transform = modules.GraphIndependent(
                edge_fn, node_fn, global_fn)

    def __call__(self, input_op, num_processing_steps):
        latent = self._encoder(input_op)
        latent0 = latent
        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)
            decoded_op = self._decoder(latent)
            output_ops.append(self._output_transform(decoded_op))
        return output_ops

class NN_interface():
    def __init__(self,settings,num_processing_steps):
        self.model = EncodeProcessDecode(**settings)
        self.num_processing_steps = num_processing_steps

    def do_policy_and_value(self,graph:Graph) -> (np.ndarray,np.ndarray,float):
        graph_tup,vertexmap = convert_graph([graph])
        output = self.model(graph_tup,self.num_processing_steps)[-1]
        moves = []
        probs = []
        for node_features,vertex_ind in zip(output.nodes,vertexmap):
            if vertex_ind!=-1:
                moves.append(vertex_ind)
                probs.append(node_features[0])
        probs = np.array(probs)
        moves = np.array(moves,dtype=int)
        probs = probs/np.sum(probs)
        value = output.globals[0]
        return moves,probs,value
