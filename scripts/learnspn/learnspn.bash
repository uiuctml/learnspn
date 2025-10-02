#!/usr/bin/env bash

##
# @file   learnspn.bash
# @author Simon Yu
# @date   02/13/2024
# @brief  Script for LearnSPN PCs.
##

# Go to script directory
cd "$(dirname $0)"

# Go to project root directory
cd "../.."

function main()
{
	declare -A datasets=(["awa2"]="29" ["celeba"]="30" ["gtsrb"]="31" ["mnist"]="32")

	local dataset_prefix="$1"

	if [ "$dataset_prefix" = "" ]
	then
		echo "[INFO]: Usage: ./$(basename $0) <dataset prefix>"
		exit 1
	elif [ "${datasets[$dataset_prefix]}" = "" ]
	then
		echo "[FATAL]: Unknown dataset prefix. Quit."
		exit 1
	fi

	local project_name="learnspn"
	local dir_build="build/$project_name"
	local dir_output="output/$project_name"
	local dir_source="src"
	local file_name_spn="$dataset_prefix.spn"
	local file_path_class_main="$dir_source/exp/RunSLSPN.java"

	# Build project
	echo "[INFO]: Building project..."
	if [ ! -d "$dir_build" ]
	then
		mkdir -p "$dir_build"
	fi
	javac -cp "$dir_source" -d "$dir_build" "$file_path_class_main"

	# Run project
	echo "[INFO]: Running project..."
	if [ ! -d "$dir_output" ]
	then
		mkdir -p "$dir_output"
	fi
	java -cp "$dir_build" "exp.RunSLSPN" "DATA" "${datasets[$dataset_prefix]}" "N" "$dir_output/$file_name_spn" "CP" "1.6" "GF" "10"
}

# Call main function
main "$@"
