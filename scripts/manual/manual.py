#!/usr/bin/env python3

"""
@file   manual.py
@author Simon Yu
@date   12/10/2024
@brief  Script for manual PCs.
"""

import json
import natsort
import os

dataset_prefix = "awa2"
dir_outputs = "../../outputs/manual"
file_name_pc_manual = dataset_prefix + ".spn.txt"
file_path_dataset = os.path.join("../../data", dataset_prefix + ".ts.data")
file_path_dataset_config = os.path.join("../../../npc-dataset-utils/configs/npc-dataset-utils", dataset_prefix + ".json")

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

def getIndicesFromLabelsAttribute(labels_attribute):
    indices = {}

    for attribute in labels_attribute.keys():
        labels_to_indices = {}

        for i in range(len(labels_attribute[attribute])):
            labels_to_indices[labels_attribute[attribute][i]] = i

        indices[attribute] = labels_to_indices

    return indices

def loadRules():
    file_dataset = open(file_path_dataset, "r")
    lines = file_dataset.readlines()
    file_dataset.close()

    count_instances = len(lines)
    rules = {}

    for line in lines:
        indices_category = line.split(',')
        indices_category = tuple([int(index_category) for index_category in indices_category])

        if indices_category not in rules:
            rules[indices_category] = 1
        else:
            rules[indices_category] += 1

    for indices_category in rules.keys():
        rules[indices_category] /= count_instances

    return rules

def main():
    file_config_dataset = open(file_path_dataset_config, "r")
    config_dataset = json.load(file_config_dataset)
    file_config_dataset.close()

    attribute_index_task = len(config_dataset["attributes"])
    cat_node_dict = {}
    edge_count_sum_prd = 0
    edge_count_prd_leaf = 0
    labels_attribute = getLabelsAttribute(config_dataset)
    labels_attribute_indices = getIndicesFromLabelsAttribute(labels_attribute)
    labels_class = getLabelsClass(config_dataset)
    lines_edges = "##EDGES##\n"
    lines_nodes = "##NODES##\n"
    node_count_leaf = 0
    node_count_prd = 0
    node_count_sum = 0
    node_sequence = 0
    node_sequence_root = node_sequence
    rules = loadRules()

    # Add sum root node
    lines_nodes += str(node_sequence_root) + ",SUM\n"
    node_sequence += 1
    node_count_sum += 1

    # Log attribute statistics
    for attribute_name in labels_attribute.keys():
        print("[INFO]: Total categories for attribute \"" + attribute_name + "\": " + str(len(labels_attribute[attribute_name])) + ".")

    # Add attribute leaf nodes
    for (attribute_index, attribute) in enumerate(config_dataset["attributes"]):
        for category_index in labels_attribute_indices[attribute["name"]].values():
            line_cat_node = "CATNODEPRD," + str(attribute_index) + "," + str(category_index)
            cat_node_dict[line_cat_node] = node_sequence
            lines_nodes += str(node_sequence) + "," + line_cat_node + "\n"
            node_sequence += 1
            node_count_leaf += 1

    # Add task leaf nodes
    for category_index in range(0, len(labels_class)):
        line_cat_node = "CATNODEPRD," + str(attribute_index_task) + "," + str(category_index)
        cat_node_dict[line_cat_node] = node_sequence
        lines_nodes += str(node_sequence) + "," + line_cat_node + "\n"
        node_sequence += 1
        node_count_leaf += 1

    # Add product nodes and edges
    for indices_category in rules.keys():
        frequency = rules[indices_category]

        # Add product node
        node_sequence_prd = node_sequence
        lines_nodes += str(node_sequence_prd) + ",PRD\n"
        node_sequence += 1
        node_count_prd += 1

        # Add root-to-product edge
        lines_edges += str(node_sequence_root) + "," + str(node_sequence_prd) + "," + str(frequency) + "\n"
        edge_count_sum_prd += 1

        # Add product-to-leaf edges
        for (attribute_index, category_index) in enumerate(indices_category):
            line_cat_node = "CATNODEPRD," + str(attribute_index) + "," + str(category_index)
            node_sequence_cat = cat_node_dict[line_cat_node]
            lines_edges += str(node_sequence_prd) + "," + str(node_sequence_cat) + "\n"
            edge_count_prd_leaf += 1

    print("[INFO]: Total PC sum nodes: " + str(node_count_sum) + ".")
    print("[INFO]: Total PC product nodes: " + str(node_count_prd) + ".")
    print("[INFO]: Total PC leaf nodes: " + str(node_count_leaf) + ".")
    print("[INFO]: Total PC sum-to-product edges: " + str(edge_count_sum_prd) + ".")
    print("[INFO]: Total PC product-to-leaf edges: " + str(edge_count_prd_leaf) + ".")

    lines = lines_nodes + lines_edges

    if not os.path.isdir(dir_outputs):
        os.makedirs(dir_outputs, exist_ok = True)

    file_path_pc_manual = os.path.join(dir_outputs, file_name_pc_manual)

    with open(file_path_pc_manual, "w") as file_pc_manual:
        file_pc_manual.writelines(lines)

    print("[INFO]: Wrote to \"" + file_path_pc_manual + "\".")

    return

if __name__ == "__main__":
    main()
