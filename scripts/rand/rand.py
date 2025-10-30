#!/usr/bin/env python3

"""
@file   random.py
@author Simon Yu
@date   10/29/2025
@brief  Script for random PCs.
"""

import json
import matplotlib.pyplot
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
plot = False
region_graph_split_parts = 2
region_graph_split_depth = 2
region_graph_split_repetitions = 8

def createPC(count_variables):
    graph = region_graph.RegionGraph(range(count_variables))

    for _ in range(region_graph_split_repetitions):
        graph.random_split(region_graph_split_parts, region_graph_split_depth)

    arguments = rat_torch.SpnArgs()

    arguments.num_gauss = pc_count_leaf_nodes_per_region
    arguments.num_sums = pc_count_sum_nodes_per_region

    return rat_torch.RatSpn(pc_count_root_nodes, region_graph = graph, args = arguments)

def getLabelsAttribute(dataset_config):
    labels_attribute = {}

    for attribute in dataset_config["attributes"]:
        if "" in attribute["labels"]:
            attribute["labels"].remove("")

        labels_attribute[attribute["name"]] = attribute["labels"]

    return labels_attribute

def assignRegionIDs(pc):
    id = 0
    regions = queue.Queue()

    regions.put(pc.output_vector)

    while not regions.empty():
        region_parent = regions.get()

        region_parent.id = id
        id += 1

        if isinstance(region_parent, rat_torch.SumVector):
            region_parent.type = "SUM"
        elif isinstance(region_parent, rat_torch.ProductVector):
            region_parent.type = "PRD"
        elif isinstance(region_parent, rat_torch.GaussVector):
            region_parent.type = "LEA"
        else:
            region_parent.type = "???"

        if not isinstance(region_parent, rat_torch.GaussVector):
            for region_child in region_parent.inputs:
                regions.put(region_child)

    return

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

def plotRegionGraph(pc):
    graph = networkx.DiGraph()
    regions = queue.Queue()

    regions.put(pc.output_vector)

    while not regions.empty():
        region_parent = regions.get()
        region_parent_name = str(region_parent.type) + " " + str(region_parent.id)
        graph.add_node(region_parent_name, layer = region_parent.depth)

        if not isinstance(region_parent, rat_torch.GaussVector):
            for region_child in region_parent.inputs:
                region_child_name = str(region_child.type) + " " + str(region_child.id)
                regions.put(region_child)
                graph.add_node(region_child_name, layer = region_child.depth)
                graph.add_edge(region_parent_name, region_child_name)

    graph_position = networkx.multipartite_layout(graph, subset_key = "layer")

    networkx.draw_networkx(graph, pos = graph_position, node_size = 1000, font_size = 7)
    matplotlib.pyplot.title("Region Graph")
    matplotlib.pyplot.show()

    return

def main():
    file_config_dataset = open(file_path_dataset_config, "r")
    config_dataset = json.load(file_config_dataset)
    file_config_dataset.close()

    count_variables = len(getLabelsAttribute(config_dataset)) + 1
    pc = createPC(count_variables)

    assignRegionIDs(pc)
    assignRegionDepths(pc)

    if plot:
        plotRegionGraph(pc)

    return

if __name__ == "__main__":
    main()
