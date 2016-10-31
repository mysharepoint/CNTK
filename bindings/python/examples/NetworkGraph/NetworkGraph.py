# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
from cntk import Trainer
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, FULL_DATA_SWEEP
from cntk.device import cpu, set_default_device
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error, relu, element_times, constant
from examples.common.nn import fully_connected_classifier_net, print_training_progress


def graph_to_png(node,path):
    '''
    Generic function that walks through every node of the graph starting at ``node``,
    creates a network graph, and saves it under ``path`` in PNG format. 
    
    
    Requirements: Pydot and Graphviz

    conda install pydot-ng
    conda install graphviz
    setx PATH "[cntk conda env path]\Library\bin\graphviz;%PATH%"

    Args:
        node (graph node): the node to start the journey from
        path (`str`): destination folder

    Returns:
        Pydot object containing all nodes and edges
    '''
    import pydot_ng as pydot
    
    # walk every node
    visitor = lambda x: True

    # initialize a dot object to store vertices and edges
    dot_object = pydot.Dot(graph_name="network_graph",rankdir='TB')
    dot_object.set_node_defaults(shape='rectangle', fixedsize='false',
                             height=.85, width=.85, fontsize=12)
    dot_object.set_edge_defaults(fontsize=10)

    # walk the graph iteratively
    stack = [node]
    accum = []
    visited = set()

    while stack:
        node = stack.pop()
        
        if node in visited:
            continue

        try:
            # Function node
            node = node.root_function
            stack.extend(node.inputs)

            # add current node
            cur_node = pydot.Node(node.op_name+' '+node.uid, label=node.op_name,shape='circle',
                                    fixedsize='true', height=1, width=1)
            dot_object.add_node(cur_node)

            # add node's inputs
            for child in node.inputs:
                child_node = pydot.Node(child.uid)#,shape="rectangle")#,label=child.name)
                dot_object.add_node(child_node)
                dot_object.add_edge(pydot.Edge(child_node, cur_node,label=str(child.shape)))

            # ad node's output
            out_node = pydot.Node(node.outputs[0].uid)#,shape="rectangle")#,label=node.outputs[0].name)
            dot_object.add_node(out_node)
            dot_object.add_edge(pydot.Edge(cur_node,out_node,label=str(node.outputs[0].shape)))

        except AttributeError:
            # OutputVariable node
            try:
                if node.is_output:
                    stack.append(node.owner)
            except AttributeError:
                pass

    if visitor(node):
        accum.append(node)

    # save to png
    dot_object.write_png(path + '\\network_graph.png', prog='dot')

    return dot_object


def graph_to_string(node):
    '''
    Generic function that walks through every node of the graph starting at ``node``
    and dumps the network structure to `str` 
    Args:
        node (graph node): the node to start the journey from

    Returns:
        `str` describing network structure where each line is
        in the following format:
        OperatorName(Input1.uid, Input2.uid, ...) -> Output.uid
    '''

    # walk every node
    visitor = lambda x: True
    
    model = ''

    stack = [node]
    accum = []
    visited = set()

    while stack:
        node = stack.pop()
        if node in visited:
            continue

        try:
            # Function node
            stack.extend(node.root_function.inputs)

            # add current node operator
            model += node.root_function.op_name + '('

            # add inputs
            for i in range(len(node.root_function.inputs)):
                
                model += node.root_function.inputs[i].uid
                if (i != len(node.root_function.inputs) - 1):
                    model += ", "

            # add output
            model += ") -> " + node.outputs[0].uid +'\n'
        except AttributeError:
            
            # OutputVariable node
            try:
                if node.is_output:
                    stack.append(node.owner)
            except AttributeError:
                pass

        if visitor(node):
            accum.append(node)

        visited.add(node)

    # return lines in reversed order
    return "\n".join(model.split("\n")[::-1])


abs_path = os.path.dirname(os.path.abspath(__file__))

input_dim = 784
num_output_classes = 10
num_hidden_layers = 1
hidden_layers_dim = 200

# input variables denoting the features and label data
input = input_variable(input_dim, np.float32)
label = input_variable(num_output_classes, np.float32)

# instantiate the feedforward classification model
scaled_input = element_times(constant(0.00390625), input)
netout = fully_connected_classifier_net(
    scaled_input, num_output_classes, hidden_layers_dim, num_hidden_layers, relu)

# save network graph in the PNG format 
graph_to_png(netout, abs_path)

# get network structure as a string
model = graph_to_string(netout)

print(model)