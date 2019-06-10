# -*- coding: utf-8 -*-

import queue
from collections import OrderedDict
import pandas as pd
import torch.nn as nn

from summary.model_hook import CModelHook
from summary.summary_tree import CSummaryTree, CSummaryNode

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 10000)


def get_parent_node(root_node, summary_node_name):
    assert isinstance(root_node, CSummaryNode)

    node = root_node
    names = summary_node_name.split('.')
    for i in range(len(names) - 1):
        node_name = '.'.join(names[0:i+1])
        child_index = node.find_child_index(node_name)
        assert child_index != -1
        node = node.children[child_index]
    return node


def convert_leaf_modules_to_summary_tree(leaf_modules):
    assert isinstance(leaf_modules, OrderedDict)

    create_index = 1
    root_node = CSummaryNode(name='root', parent=None)
    for leaf_module_name, leaf_module in leaf_modules.items():
        names = leaf_module_name.split('.')
        for i in range(len(names)):
            create_index += 1
            summary_node_name = '.'.join(names[0:i+1])
            parent_node = get_parent_node(root_node, summary_node_name)
            node = CSummaryNode(name=summary_node_name, parent=parent_node)
            parent_node.add_child(node)
            if i == len(names) - 1:  # leaf module itself
                input_shape = leaf_module.input_shape.numpy().tolist()
                output_shape = leaf_module.output_shape.numpy().tolist()
                node.input_shape = input_shape
                node.output_shape = output_shape
                node.parameter_quantity = leaf_module.parameter_quantity.numpy()[0]
                node.inference_memory = leaf_module.inference_memory.numpy()[0]
                node.MAdd = leaf_module.MAdd.numpy()[0]
                node.duration = leaf_module.duration.numpy()[0]
    return CSummaryTree(root_node)


def get_collected_summary_nodes(root_node, query_granularity):
    assert isinstance(root_node, CSummaryNode)

    collected_nodes = []
    q = queue.Queue()
    q.put(root_node)
    while not q.empty():
        node = q.get()
        for child in node.children:
            q.put(child)
        if node.depth == query_granularity:
            collected_nodes.append(node)
        if node.depth < query_granularity <= node.granularity:
            collected_nodes.append(node)
    collected_nodes = sorted(collected_nodes, key=lambda x: x.create_index)
    return collected_nodes


def pretty_format(collected_nodes):
    data = list()
    for node in collected_nodes:
        name = node.name
        input_shape = ' '.join(['{:>3d}'] * len(node.input_shape)).format(
            *[e for e in node.input_shape])
        output_shape = ' '.join(['{:>3d}'] * len(node.output_shape)).format(
            *[e for e in node.output_shape])
        parameter_quantity = node.parameter_quantity
        inference_memory = node.inference_memory
        MAdd = node.MAdd
        duration = node.duration
        data.append([name, input_shape, output_shape,
                     parameter_quantity, inference_memory, MAdd, duration])
    df = pd.DataFrame(data)
    df.columns = ['module name', 'input shape', 'output shape',
                  'parameter quantity', 'inference memory(MB)',
                  'MAdd', 'duration']
    df['duration percent'] = df['duration'] / df['duration'].sum()
    total_parameters_quantity = df['parameter quantity'].sum()
    total_memory = df['inference memory(MB)'].sum()
    total_operation_quantity = df['MAdd'].sum()
    del df['duration']
    df = df.fillna(' ')
    df['inference memory(MB)'] = df['inference memory(MB)'].apply(
        lambda x: '{:.2f}MB'.format(x))
    df['duration percent'] = df['duration percent'].apply(lambda x: '{:.2%}'.format(x))
    df['MAdd'] = df['MAdd'].apply(lambda x: '{:,}'.format(x))

    summary = str(df) + '\n'
    summary += "=" * len(str(df).split('\n')[0])
    summary += '\n'
    summary += "total parameters quantity: {:,}\n".format(total_parameters_quantity)
    summary += "total memory: {:.2f}MB\n".format(total_memory)
    summary += "total MAdd: {:,}\n".format(total_operation_quantity)
    print(summary)
    return summary


def model_summary(model, input_size, query_granularity=1):
    assert isinstance(model, nn.Module)
    assert isinstance(input_size, (list, tuple)) and len(input_size) == 3

    model_hook = CModelHook(model, input_size)
    leaf_modules = model_hook.retrieve_leaf_modules()
    summary_tree = convert_leaf_modules_to_summary_tree(leaf_modules)
    collected_nodes = summary_tree.get_collected_summary_nodes(query_granularity)
    summary = pretty_format(collected_nodes)
    return summary
