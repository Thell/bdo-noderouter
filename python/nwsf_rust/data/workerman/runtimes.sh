#!/bin/bash

# List files sorted by modification time (oldest first)
files=( $(ls -1t --time=mod) )

# Iterate through files to compute deltas
for ((i=${#files[@]}-1; i>0; i--)); do
    older=${files[i]}
    newer=${files[i-1]}
    
    time_older=$(stat -c %Y "$older")
    time_newer=$(stat -c %Y "$newer")
    
    delta=$((time_newer - time_older))
    
    echo "$newer $delta"
done
