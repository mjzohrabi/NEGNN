#!/bin/bash
# Get the list of all processes using the GPU
process_list=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits)

# Check if the process list is empty
if [[ -z "$process_list" ]]; then
    echo "No processes found using the GPU."
    exit 1
fi

# Loop through each line in the process list
echo "$process_list" | while IFS=',' read -r pid pname; do
    # Remove leading/trailing whitespaces from PID and process name
    pid=$(echo $pid | tr -d '[:space:]')
    pname=$(echo $pname | tr -d '[:space:]')
    
    # Check if the process name contains 'python'
    if [[ $pname == *"python"* ]]; then
        # Kill the process
        kill -9 $pid
        echo "Killed Python process with PID $pid"
    fi
done
