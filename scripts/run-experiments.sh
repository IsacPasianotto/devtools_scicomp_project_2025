#!/bin/bash

process_config() {
    local config_path="$1"
    local config_filename=$(basename "$config_path")
    local backend=$(basename "$(dirname "$config_path")")
    local problem_size=$(echo "$config_filename" | grep -o '[0-9]\+x[0-9]\+')

    echo "> Running: $config_filename"

    for np in "${procs_list[@]}"; do
        echo "> Running with $np processes..."
        output_dir="$OUTPUT_ROOT/${backend}/${np}"
        mkdir -p "$output_dir"
        output_file="$output_dir/${problem_size}.txt"

        if [[ "$config_path" == *null* ]]; then
            mpirun -np "$np" python run.py --config-file "$config_path" 2>&1 | tee "$output_file"
            echo "> Saved output to $output_file"
        else
            echo "Profiling.. no STDOUT provided"
            mpirun -np "$np" python run.py --config-file "$config_path" > /dev/null
        fi
        # Move the ouput of the memory profiler
        mkdir -p "$output_dir/memory_profile/$problem_size"
        for file in memory_profile-*; do
            [[ -f "$file" ]] || continue
            mv "$file" "$output_dir/memory_profile/$problem_size/${file}"
        done
        # Move the ouput of the line profiler
	mkdir -p "$output_dir/line_profile/$problem_size"
        for file in line_profile-*; do
            [[ -f "$file" ]] || continue
            mv "$file" "$output_dir/line_profile/$problem_size/${file}"
        done
        # Move the ouput of memray
       	mkdir -p "$output_dir/memray/$problem_size"
        for file in memray-*; do
            [[ -f "$file" ]] || continue
            mv "$file" "$output_dir/memray/$problem_size/${file}"
       	done
    done

    echo "> Finished: $config_filename"
}





# How many procs
procs_list=(2 4 8 16 32 64 96 128)
# Config files and output folder
CONFIG_ROOT="../experiments"
OUTPUT_ROOT="../output"


# Necessary to use the stdin trick
tmpfile=$(mktemp)
find "$CONFIG_ROOT" -type f -name "*.yaml" > "$tmpfile"

# Why mapfile ? 
#We cannot do the classical while loop over STDIN since it seems that mpi4py or something is consuming the buffer inside the loop.
mapfile -t yaml_files < "$tmpfile"
#Iterate over yaml, now is an array
for config_path in "${yaml_files[@]}"; do
    ( process_config "$config_path" )
done

rm "$tmpfile"
