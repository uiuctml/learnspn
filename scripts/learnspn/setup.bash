#!/usr/bin/env bash

# Go to script directory
cd "$(dirname $0)"

# Go to project root directory
cd "../.."

function main()
{
	local project_name="learnspn"
	local dir_build="build/$project_name"
	local dir_output="output/$project_name"
	local dir_source="src"
	local file_name_spn="gtsrb.spn"
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
	if [ ! -d "$dir_" ]
	then
		mkdir -p "$dir_output"
	fi
	java -cp "$dir_build" "exp.RunSLSPN" "DATA" "29" "N" "$dir_output/$file_name_spn" "CP" "1.6" "GF" "10"
}

# Call main function
main "$@"
