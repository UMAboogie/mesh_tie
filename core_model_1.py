# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Core learned graph net model."""

import collections
from math import ceil
import math
from collections import OrderedDict
import functools
import torch
from torch import nn as nn
import torch_scatter
from torch_scatter.composite import scatter_softmax
import torch.nn.functional as F

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders',
                                             'recievers'])
#MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'reciever_features', 'sender_features', 'adj'])

device = torch.device('cuda')


class LazyMLP(nn.Module):
    def __init__(self, output_sizes):
        super().__init__()
        num_layers = len(output_sizes)
        self._layers_ordered_dict = OrderedDict()
        for index, output_size in enumerate(output_sizes):
            self._layers_ordered_dict["linear_" + str(index)] = nn.LazyLinear(output_size)
            if index < (num_layers - 1):
                self._layers_ordered_dict["relu_" + str(index)] = nn.ReLU()
        self.layers = nn.Sequential(self._layers_ordered_dict)

    def forward(self, input):
        input = input.to(device)
        y = self.layers(input)
        return y


class GraphNetBlock(nn.Module):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, model_fn, output_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = output_size // num_heads
        self.output_size = output_size
        
        """
        self.sender_model = model_fn(output_size)
        self.reciever_model = model_fn(output_size)
        self.node_model = model_fn(output_size)
        """

        self.query_linear = nn.Linear(self.output_size, self.output_size, bias=False)

        self.reciever_linear = nn.Linear(self.output_size, self.output_size, bias=True)
        self.sender_linear = nn.Linear(self.output_size, self.output_size, bias=True)
        self.memory_linear = nn.Linear(self.output_size, self.output_size, bias=True)
        """
        self.r_proj = model_fn(output_size)
        self.s_proj = model_fn(output_size)
        self.proj = model_fn(output_size)
        """

        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(p=0.2)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.node_norm = nn.LayerNorm(normalized_shape=output_size)
        self.sender_norm = nn.LayerNorm(normalized_shape=output_size)
        self.reciever_norm = nn.LayerNorm(normalized_shape=output_size)

    def _update_edge_features(self, node_features, reciever_features, sender_features):
        """Aggregrates node features, and applies edge function."""
        next_reciever_features = self.reciever_linear(node_features) + self.memory_linear(reciever_features)
        next_sender_features = self.sender_linear(node_features) + self.memory_linear(sender_features)
        next_reciever_features = self.leaky_relu(next_reciever_features)
        next_sender_features = self.leaky_relu(next_sender_features)

        next_reciever_features = self.reciever_norm(next_reciever_features)
        next_sender_features = self.sender_norm(next_sender_features)
        return next_reciever_features, next_sender_features

    def unsorted_segment_operation(self, data, segment_ids, num_segments):
        """
        Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

        :param data: A tensor whose segments are to be summed.
        :param segment_ids: The segment indices tensor.
        :param num_segments: The number of segments.
        :return: A tensor of same data type as the data argument.
        """
        assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

        # segment_ids is a 1-D tensor repeat it to have the same shape as data
        data = data.to(device)
        segment_ids = segment_ids.to(device)
        if len(segment_ids.shape) == 1:
            s = torch.prod(torch.tensor(data.shape[1:])).long().to(device)
            segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:]).to(device)

        assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

        shape = [num_segments] + list(data.shape[1:])
        result = torch.zeros(*shape).to(device)

        result = torch_scatter.scatter_add(data.float(), segment_ids, dim=0, dim_size=num_segments)
        result = result.type(data.dtype)
        return result

    def _update_node_features(self, node_features, reciever_features, sender_features, adj):
        """Aggregrates edge features, and applies node function."""
        num_nodes = node_features.shape[0]

        query_features = self.query_linear(node_features)
        query_features_head = query_features.view(num_nodes, self.num_heads, self.hidden_size)
        reciever_features_head = reciever_features.view(num_nodes, self.num_heads, self.hidden_size)
        sender_features_head = sender_features.view(num_nodes, self.num_heads, self.hidden_size)

        qr = torch.einsum('ijk,ijk->ji', query_features_head, reciever_features_head)
        qr = qr.view(self.num_heads, num_nodes, 1).repeat(1, 1, num_nodes)
        qs = torch.einsum('ijk,ljk->jil', query_features_head, sender_features_head)
        
        attn_before_sm = qr + qs #num_heads * num_nodes * num_nodes

        mask = adj.view(1, num_nodes, num_nodes).repeat(self.num_heads, 1, 1)
        attn_before_sm = attn_before_sm.masked_fill(mask == 0, float('-inf'))
        attn_before_sm = attn_before_sm / math.sqrt(self.hidden_size)

        attn_score = self.softmax(attn_before_sm)
        attn_score = self.dropout(attn_score)

        aggr_sender_head = torch.einsum('ijk,kil->jil', attn_score, sender_features_head)
        next_node_features = reciever_features + aggr_sender_head.contiguous().view(num_nodes, self.num_heads*self.hidden_size)
        next_node_features = self.leaky_relu(next_node_features)
        next_node_features = self.node_norm(next_node_features)
        return next_node_features

    def forward(self, graph):
        """Applies GraphNetBlock and returns updated MultiGraph."""
        # apply edge functions  
        new_reciever_features, new_sender_features = self._update_edge_features(graph.node_features, graph.reciever_features, graph.sender_features)

        # apply node function
        new_node_features = self._update_node_features(graph.node_features, graph.reciever_features, graph.sender_features, graph.adj)

        # add residual connections
        new_node_features += graph.node_features
        new_reciever_features += graph.reciever_features
        new_sender_features += graph.sender_features
        return MultiGraph(new_node_features, new_reciever_features, new_sender_features, graph.adj)


class Encoder(nn.Module):
    """Encodes node and edge features into latent features."""

    def __init__(self, make_mlp, latent_size):
        super().__init__()
        self._make_mlp = make_mlp
        self._latent_size = latent_size
        self.node_model = self._make_mlp(latent_size)
        self.reciever_model = self._make_mlp(latent_size)
        self.sender_model = self._make_mlp(latent_size)

    def forward(self, graph):
        node_latents = self.node_model(graph.node_features)
        sender_latents = self.sender_model(graph.reciever_features)
        reciever_latents = self.reciever_model(graph.sender_features)
        return MultiGraph(node_latents, reciever_latents, sender_latents, graph.adj)


class Decoder(nn.Module):
    """Decodes node features from graph."""

    """Encodes node and edge features into latent features."""

    def __init__(self, make_mlp, output_size):
        super().__init__()
        self.model = make_mlp(output_size)

    def forward(self, graph):
        return self.model(graph.node_features)

class Processor(nn.Module):
    '''
    This class takes the nodes with the most influential feature (sum of square)
    The the chosen numbers of nodes in each ripple will establish connection(features and distances) with the most influential nodes and this connection will be learned
    Then the result is add to output latent graph of encoder and the modified latent graph will be feed into original processor

    Option: choose whether to normalize the high rank node connection
    '''

    def __init__(self, make_mlp, output_size, message_passing_steps, num_heads):
        super().__init__()
        self.graphnet_blocks = nn.ModuleList()
        for index in range(message_passing_steps):
            self.graphnet_blocks.append(GraphNetBlock(model_fn=make_mlp, output_size=output_size, num_heads=num_heads))

    def forward(self, latent_graph, mask=None):
        for graphnet_block in self.graphnet_blocks:
            latent_graph = graphnet_block(latent_graph)
        return latent_graph

class EncodeProcessDecode(nn.Module):
    """Encode-Process-Decode GraphNet model."""

    def __init__(self,
                 output_size,
                 latent_size,
                 num_layers,
                 message_passing_steps,
                 num_heads):
        super().__init__()
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._message_passing_steps = message_passing_steps
        self._num_heads = num_heads

        self.encoder = Encoder(make_mlp=self._make_mlp, latent_size=self._latent_size)
        self.processor = Processor(make_mlp=self._make_slp, output_size=self._latent_size,
                                   message_passing_steps=self._message_passing_steps, num_heads=self._num_heads)
        self.decoder = Decoder(make_mlp=functools.partial(self._make_mlp, layer_norm=False),
                               output_size=self._output_size)

    def _make_mlp(self, output_size, layer_norm=True):
        """Builds an MLP."""
        widths = [self._latent_size] * self._num_layers + [output_size]
        network = LazyMLP(widths)
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(normalized_shape=widths[-1]))
        return network

    def _make_slp(self, output_size, layer_norm=True):
        """Builds an MLP."""
        widths = [self._latent_size, output_size]
        network = LazyMLP(widths)
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(normalized_shape=widths[-1]))
        return network

    def forward(self, graph, is_training, world_edge_normalizer=None):
        """Encodes and processes a multigraph, and returns node features."""
        #adj = self._make_adj(graph)
        latent_graph = self.encoder(graph)
        latent_graph = self.processor(latent_graph)
        return self.decoder(latent_graph)

    """
    def _make_adj(self, graph):
        node_num = len(graph.node_features)
        adj = torch.zeros((node_num, node_num), dtype=torch.int64, requires_grad=False).to(device)
        
        for edge_set in graph.edge_sets:
            recievers = graph.edge_sets[0].recievers
            senders = graph.edge_sets[0].senders
    """


        