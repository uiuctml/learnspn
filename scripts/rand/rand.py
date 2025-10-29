#!/usr/bin/env python3

"""
@file   random.py
@author Simon Yu
@date   10/29/2025
@brief  Script for random PCs.
"""

import rat_torch
import region_graph

def traverse_layer_recursive(layers, layer_index, layer_nodes):
    if layer_index not in layers:
        layers[layer_index] = []

    layers[layer_index] += layer_nodes

    for layer_node in layer_nodes:
        if isinstance(layer_node, rat_torch.GaussVector):
            continue

        traverse_layer_recursive(layers, layer_index + 1, layer_node.inputs)

    return

def main():
    count_variables = 3
    count_repetitions = 20
    depth_split = 2

    rg = region_graph.RegionGraph(range(count_variables))

    for _ in range(0, count_repetitions):
        rg.random_split(2, depth_split)

    args = rat_torch.SpnArgs()

    args.num_sums = 2
    args.num_gauss = 1

    spn = rat_torch.RatSpn(1, region_graph=rg, name="spn", args=args)

    layers = {}

    traverse_layer_recursive(layers, 0, [spn.output_vector])

    for i in range(len(layers)):
        print(str(i) + ": ", end = "")

        for node in layers[i]:
            print(node, end = "")

            if isinstance(node, rat_torch.GaussVector):
                print(": ", end = "")
                print(node.scope, end = "")

            print("; ", end = "")

        print()

    return

if __name__ == "__main__":
    main()
