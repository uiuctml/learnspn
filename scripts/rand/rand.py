#!/usr/bin/env python3

"""
@file   random.py
@author Simon Yu
@date   10/29/2025
@brief  Script for random PCs.
"""

import itertools
import json
import matplotlib.pyplot
import natsort
import networkx
import os
import queue
import rat_torch
import region_graph

dataset_prefix = "awa2"
file_path_dataset_config = os.path.join("../../../npc-dataset-utils/configs/npc-dataset-utils", dataset_prefix + ".json")
pc_count_leaf_nodes_per_region = 100
pc_count_root_nodes = 1
pc_count_sum_nodes_per_region = 2
pc_plot = False
region_graph_plot = False
region_graph_split_parts = 2
region_graph_split_depth = 2
region_graph_split_repetitions = 8

def assignRegionDepths(pc):
    def recurseDepths(depth, regions_recurse, regions_leaf):
        for region in regions_recurse:
            region.depth = depth

            if not isinstance(region, rat_torch.GaussVector):
                recurseDepths(depth + 1, region.inputs, regions_leaf)
            else:
                regions_leaf.add(region)

    depth_leaf = -1
    regions_leaf = set()

    recurseDepths(0, [pc.output_vector], regions_leaf)

    for region_leaf in regions_leaf:
        if region_leaf.depth > depth_leaf:
            depth_leaf = region_leaf.depth

    for region_leaf in regions_leaf:
        region_leaf.depth = depth_leaf

    return

def assignRegionIDs(pc):
    id = 0
    regions = queue.Queue()
    regions_visited = set()

    regions.put(pc.output_vector)
    regions_visited.add(pc.output_vector)

    while not regions.empty():
        region_parent = regions.get()

        region_parent.id = id
        id += 1

        if isinstance(region_parent, rat_torch.SumVector):
            region_parent.type = 'S'
        elif isinstance(region_parent, rat_torch.ProductVector):
            region_parent.type = 'P'
        elif isinstance(region_parent, rat_torch.GaussVector):
            region_parent.type = 'L'
        else:
            print("[FATAL]: Unknown region type. Quit.")
            exit(-1)

        if not isinstance(region_parent, rat_torch.GaussVector):
            for region_child in region_parent.inputs:
                if region_child not in regions_visited:
                    regions.put(region_child)
                    regions_visited.add(region_child)

    return

def computeScopeSizes(labels_attribute, labels_class):
    scope = 0
    sizes_scope = {}

    for (scope, attribute) in enumerate(labels_attribute.keys()):
        sizes_scope[scope] = len(labels_attribute[attribute])

    sizes_scope[scope + 1] = len(labels_class)

    return sizes_scope

def createPC(labels_attribute):
    count_variables = len(labels_attribute) + 1
    graph = region_graph.RegionGraph(range(count_variables))

    for _ in range(region_graph_split_repetitions):
        graph.random_split(region_graph_split_parts, region_graph_split_depth)

    arguments = rat_torch.SpnArgs()

    arguments.num_gauss = pc_count_leaf_nodes_per_region
    arguments.num_sums = pc_count_sum_nodes_per_region

    return rat_torch.RatSpn(pc_count_root_nodes, region_graph = graph, args = arguments)

def createPCEdges(pc):
    edges = set()
    regions = queue.Queue()

    regions.put(pc.output_vector)

    while not regions.empty():
        region_parent = regions.get()

        if isinstance(region_parent, rat_torch.SumVector):
            for region_child in region_parent.inputs:
                regions.put(region_child)

                for id_node_sum in region_parent.nodes:
                    for id_node_child in region_child.nodes:
                        edges.add((id_node_sum, id_node_child))
        elif isinstance(region_parent, rat_torch.ProductVector):
            for region_child in region_parent.inputs:
                regions.put(region_child)

            region_child_1 = region_parent.inputs[0]
            region_child_2 = region_parent.inputs[1]
            id_node_child_pairs = itertools.product(region_child_1.nodes, region_child_2.nodes)

            for (id_node_product, id_node_child_pair) in zip(region_parent.nodes, id_node_child_pairs):
                edges.add((id_node_product, id_node_child_pair[0]))
                edges.add((id_node_product, id_node_child_pair[1]))

    return edges

def createPCNodes(pc):
    id_node = 0
    ids_region_visited = set()
    nodes = {}
    regions = queue.Queue()

    regions.put(pc.output_vector)
    ids_region_visited.add(pc.output_vector.id)

    while not regions.empty():
        region_parent = regions.get()
        region_parent.nodes = set()

        for _ in range(region_parent.size):
            if id_node in nodes:
                print("[FATAL]: Duplicated PC node. Quit.")
                exit(-1)

            nodes[id_node] = (region_parent.type, region_parent.depth, region_parent.scope)
            region_parent.nodes.add(id_node)
            id_node += 1

        if not isinstance(region_parent, rat_torch.GaussVector):
            for region_child in region_parent.inputs:
                if region_child.id not in ids_region_visited:
                    regions.put(region_child)
                    ids_region_visited.add(region_child.id)

    return nodes

def expandPCLeafNodes(pc_nodes, pc_edges):
    node_id_leaf = max(pc_nodes.keys()) + 1
    pc_nodes_leaf = {}

    for node_id in pc_nodes.keys():
        node_type = pc_nodes[node_id][0]

        if node_type != 'L':
            continue

        node_depth = pc_nodes[node_id][1]
        node_scopes = pc_nodes[node_id][2]
        pc_nodes[node_id] = ('P', node_depth, node_scopes)

        for node_scope in node_scopes:
            pc_nodes_leaf[node_id_leaf] = ('L', node_depth + 1, node_scope)
            pc_edges.add((node_id, node_id_leaf))
            node_id_leaf += 1

    pc_nodes.update(pc_nodes_leaf)

    return

def getLabelsAttribute(dataset_config):
    labels_attribute = {}

    for attribute in dataset_config["attributes"]:
        if "" in attribute["labels"]:
            attribute["labels"].remove("")

        labels_attribute[attribute["name"]] = attribute["labels"]

    return labels_attribute

def getLabelsClass(dataset_config):
    if "instance_wise" in dataset_config and dataset_config["instance_wise"]:
        labels_class = []
        labels_class_set = set()

        for image_name in dataset_config["mappings"].keys():
            class_name = image_name.split('/')[0]

            if class_name not in labels_class_set:
                labels_class.append(class_name)
                labels_class_set.add(class_name)

        return natsort.natsorted(labels_class)
    else:
        return list(dataset_config["mappings"].keys())

def plotPC(pc_nodes, pc_edges):
    graph = networkx.DiGraph()

    for node_id in pc_nodes.keys():
        node_type = pc_nodes[node_id][0]
        node_depth = pc_nodes[node_id][1]
        node_name = str(node_id) + str(node_type)

        graph.add_node(node_name, layer = node_depth)

    for pc_edge in pc_edges:
        node_id_1 = pc_edge[0]
        node_type_1 = pc_nodes[node_id_1][0]
        node_name_1 = str(node_id_1) + str(node_type_1)
        node_id_2 = pc_edge[1]
        node_type_2 = pc_nodes[node_id_2][0]
        node_name_2 = str(node_id_2) + str(node_type_2)

        graph.add_edge(node_name_1, node_name_2)

    graph_position = networkx.multipartite_layout(graph, subset_key = "layer")

    networkx.draw_networkx(graph, pos = graph_position, node_size = 1000, font_size = 7)
    matplotlib.pyplot.title("Probabilistic Circuit")
    matplotlib.pyplot.show()

    return

def plotRegionGraph(pc):
    graph = networkx.DiGraph()
    regions = queue.Queue()

    regions.put(pc.output_vector)

    while not regions.empty():
        region_parent = regions.get()
        region_parent_name = str(region_parent.id) + str(region_parent.type)
        graph.add_node(region_parent_name, layer = region_parent.depth)

        if not isinstance(region_parent, rat_torch.GaussVector):
            for region_child in region_parent.inputs:
                regions.put(region_child)

                region_child_name = str(region_child.id) + str(region_child.type)
                graph.add_node(region_child_name, layer = region_child.depth)
                graph.add_edge(region_parent_name, region_child_name)

    graph_position = networkx.multipartite_layout(graph, subset_key = "layer")

    networkx.draw_networkx(graph, pos = graph_position, node_size = 1000, font_size = 7)
    matplotlib.pyplot.title("Region Graph")
    matplotlib.pyplot.show()

    return

def validateRegionGraph(pc):
    regions = queue.Queue()

    regions.put(pc.output_vector)

    while not regions.empty():
        region_parent = regions.get()

        if isinstance(region_parent, rat_torch.SumVector):
            for region_child in region_parent.inputs:
                if isinstance(region_child, rat_torch.SumVector):
                    print("[FATAL]: Invalid region graph: sum regions have sum children. Quit.")
                    exit(-1)
                elif isinstance(region_child, rat_torch.ProductVector) or isinstance(region_child, rat_torch.GaussVector):
                    continue
                else:
                    print("[FATAL]: Unknown region type. Quit.")
                    exit(-1)

                regions.put(region_child)
        elif isinstance(region_parent, rat_torch.ProductVector):
            if len(region_parent.inputs) != 2:
                print("[FATAL]: Invalid region graph: product regions have other than 2 children. Quit.")
                exit(-1)

            region_child_1 = region_parent.inputs[0]
            region_child_2 = region_parent.inputs[1]

            if region_parent.size != region_child_1.size * region_child_2.size:
                print("[FATAL]: Invalid region graph: incorrectly sized product regions. Quit.")
                exit(-1)

            id_node_child_pairs = itertools.product(range(region_child_1.size), range(region_child_2.size))

            if region_parent.size != len(id_node_child_pairs):
                print("[FATAL]: Invalid region graph: incorrectly sized product regions. Quit.")
                exit(-1)

            for region_child in region_parent.inputs:
                if isinstance(region_child, rat_torch.SumVector) or isinstance(region_child, rat_torch.GaussVector):
                    continue
                elif isinstance(region_child, rat_torch.ProductVector):
                    print("[FATAL]: Invalid region graph: product regions have product children. Quit.")
                    exit(-1)
                else:
                    print("[FATAL]: Unknown region type. Quit.")
                    exit(-1)

                regions.put(region_child)
        elif isinstance(region_parent, rat_torch.GaussVector):
            continue
        else:
            print("[FATAL]: Unknown region type. Quit.")
            exit(-1)

    return

def main():
    file_config_dataset = open(file_path_dataset_config, "r")
    config_dataset = json.load(file_config_dataset)
    file_config_dataset.close()

    labels_attribute = getLabelsAttribute(config_dataset)
    labels_class = getLabelsClass(config_dataset)
    pc = createPC(labels_attribute)

    validateRegionGraph(pc)
    assignRegionIDs(pc)
    assignRegionDepths(pc)

    if region_graph_plot:
        plotRegionGraph(pc)

    pc_nodes = createPCNodes(pc)
    pc_edges = createPCEdges(pc)
    expandPCLeafNodes(pc_nodes, pc_edges)

    if pc_plot:
        plotPC(pc_nodes, pc_edges)

    return

if __name__ == "__main__":
    main()
