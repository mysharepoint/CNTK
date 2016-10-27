# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

def dfs_walk(node, visitor, accum, visited):
    '''
    Generic function to walk the graph.

    Args:
        node (graph node): the node to start the journey from
        visitor (Python function or lambda): function that takes a node as
         argument and returns `True` if that node should be returned.
        accum (`list`): accumulator of nodes while traversing the graph
        visited (`set`): set of nodes that have already been visited.
         Initialize with empty set.
    '''
    if node in visited:
        return
    visited.add(node)
    if hasattr(node, 'root_function'):
        node = node.root_function
        for child in node.inputs:
            dfs_walk(child, visitor, accum, visited)
    elif hasattr(node, 'is_output') and node.is_output:
        dfs_walk(node.owner, visitor, accum, visited)

    if visitor(node):
        accum.append(node)


def build_graph(node, visitor, accum, visited, dot_object):
    '''
    Generic function to build the graph.

    Args:
        node (graph node): the node to start the journey from
        visitor (Python function or lambda): function that takes a node as
         argument and returns `True` if that node should be returned.
        accum (`list`): accumulator of nodes while traversing the graph
        visited (`set`): set of nodes that have already been visited.
         Initialize with empty set.
        dot_object(`Pydot.Dot`): contains the graph description in 
         dot format
    '''
    import pydot
    if node in visited:
        return
    visited.add(node)

    if hasattr(node, 'root_function'):
        node = node.root_function
        cur_node = pydot.Node(node.op_name+' '+node.uid, label=node.op_name)
        dot_object.add_node(cur_node)
        out_node = pydot.Node(node.outputs[0].uid, label=node.outputs[0].uid 
                                + '\nshape:\n' + str(node.outputs[0].shape))
        dot_object.add_node(out_node)
        dot_object.add_edge(pydot.Edge(cur_node,out_node))
        for child in node.inputs:
            child_node = pydot.Node(child.uid, label=child.uid + '\nshape:\n' + str(child.shape))
            dot_object.add_node(child_node)
            dot_object.add_edge(pydot.Edge(child_node, cur_node))
            build_graph(child, visitor, accum, visited, dot_object)

    elif hasattr(node, 'is_output') and node.is_output:
        build_graph(node.owner, visitor, accum, visited, dot_object)

    if visitor(node):
        accum.append(node)
 

def png_graph(model, path):
    '''
    Saves the network graph to the file

    Args:
        model(`cntk.ops.functions.Function`): model to plot
        path(`str`): path to the save directory
    '''
    import pydot
    dot_object = pydot.Dot(graph_name="network_graph",rankdir='LR')
    dot_object.set_node_defaults(shape='circle', fixedsize='false',
                             height=.85, width=.85, fontsize=10)

    accum = []
    build_graph(model, lambda x: True, accum, set(), dot_object)
    dot_object.write_png(path + '\\network_graph.png', prog='dot')



def visit(node, visitor):
    '''
    Generic function that walks through the graph starting at `node` and
    applies function `visitor` on each of those.

    Args:
        node (graph node): the node to start the journey from
        visitor (Python function or lambda): function that takes a node as
         argument and returns `True` if that node should be returned.

    Returns:
        List of nodes, for which `visitor` was `True`
    '''
    nodes = []
    dfs_walk(node, visitor, nodes, set())
    return nodes

def find_nodes_by_name(node, node_name):
    '''
    Finds nodes in the graph starting from `node` and doing a depth-first
    search.

    Args:
        node (graph node): the node to start the journey from
        node_name (`str`): name for which we are search nodes

    Returns:
        List of nodes having the specified name
    '''
    return visit(node, lambda x: x.name == node_name)

def build_graph(node,visitor,path):
    import pydot_ng as pydot

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
            cur_node = pydot.Node(node.op_name+' '+node.uid, label=node.op_name,shape='circle',
                                    fixedsize='true', height=1, width=1)
            dot_object.add_node(cur_node)
            out_node = pydot.Node(node.outputs[0].uid)#,shape="rectangle")#,label=node.outputs[0].name)
            dot_object.add_node(out_node)
            dot_object.add_edge(pydot.Edge(cur_node,out_node,label=str(node.outputs[0].shape)))
            for child in node.inputs:
                child_node = pydot.Node(child.uid)#,shape="rectangle")#,label=child.name)
                dot_object.add_node(child_node)
                dot_object.add_edge(pydot.Edge(child_node, cur_node,label=str(child.shape)))

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

