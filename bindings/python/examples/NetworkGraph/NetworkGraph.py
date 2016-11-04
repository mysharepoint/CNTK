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


def output_function_graph(node,png_flag=0,path=None):
    '''
    Generic function that walks through every node of the graph starting at ``node``,
    creates a network graph, and saves it as a string. If pydot_ng module is 
    installed the graph can be saved under ``path`` in the PNG format. 
    
    
    Requirements for PNG output: Pydot and Graphviz

    conda install pydot-ng
    conda install graphviz
    setx PATH "[cntk conda env path]\Library\bin\graphviz;%PATH%"

    Args:
        node (graph node): the node to start the journey from
        png_flag (`bool`, optional): saves to PNG if `True`
        path (`str`, optional): destination folder

    Returns:
        `str` containing all nodes and edges
    '''
   
    # walk every node
    visitor = lambda x: True

    if (png_flag):

        try:
            import pydot_ng as pydot
        except ImportError:
            raise ImportError("PNG format requires pydot_ng package. Unable to import pydot_ng.")

        if (path==None):
            raise ValueError("Destination folder is not specified. Expected arguments: output_function_graph(node,png_flag,path)")

        # initialize a dot object to store vertices and edges
        dot_object = pydot.Dot(graph_name="network_graph",rankdir='TB')
        dot_object.set_node_defaults(shape='rectangle', fixedsize='false',
                                 height=.85, width=.85, fontsize=12)
        dot_object.set_edge_defaults(fontsize=10)
    
    # string to store model 
    model = ''

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
            model += node.op_name + '('
            if (png_flag):
                cur_node = pydot.Node(node.op_name+' '+node.uid,label=node.op_name,shape='circle',
                                        fixedsize='true', height=1, width=1)
                dot_object.add_node(cur_node)

            # add node's inputs
            for i in range(len(node.inputs)):
                child = node.inputs[i]
                
                model += child.uid
                if (i != len(node.inputs) - 1):
                    model += ", "

                if (png_flag):
                    child_node = pydot.Node(child.uid)
                    dot_object.add_node(child_node)
                    dot_object.add_edge(pydot.Edge(child_node, cur_node,label=str(child.shape)))

            # ad node's output
            model += ") -> " + node.outputs[0].uid +'\n'

            if (png_flag):
                out_node = pydot.Node(node.outputs[0].uid)
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
    if (png_flag):
        dot_object.write_png(path + '\\network_graph.png', prog='dot')

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
model = output_function_graph(netout,1,abs_path)

print(model)