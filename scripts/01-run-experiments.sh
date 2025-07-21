#!/bin/bash

# List of process counts to try
procs_list=(2 4 8 16 32 64)

# Root directory containing config files
CONFIG_ROOT="../experiments"

# Output directory
OUTPUT_ROOT="../output"
mkdir -p "$OUTPUT_ROOT/naive"
mkdir -p "$OUTPUT_ROOT/numba_aot"
mkdir -p "$OUTPUT_ROOT/numba_jit"
mkdir -p "$OUTPUT_ROOT/numpy"

# Find all .yaml files recursively

find "$CONFIG_ROOT" -type f -name "*.yaml" | while read -r config_path; do
    config_filename=$(basename "$config_path")
    backend=$(basename "$(dirname "$config_path")")
    problem_size=$(echo "$config_filename" | grep -o '[0-9]\+x[0-9]\+')
    echo "Debug:" $config_filename $backend $problem_size
    if [[ -z "$problem_size" ]]; then
        echo "Skipping $config_path (no problem size found)"
        continue
    fi
    for np in "${procs_list[@]}"; do
        echo "Running $config_filename with $np processes..."

        output_file="output/${backend}_${problem_size}_np${np}.txt"
        mpirun -np "$np" python run.py --config-file "$config_path" > "$output_file" 2>&1

        echo "Saved output to $output_file"
    done
done
