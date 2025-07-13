#!/bin/bash
set -e

# Move to the libs directory
cd ./libs

# If abeja-toolkit exists, get the directory name under abeja-toolkit.
if [ -d abeja-toolkit ]; then
    cd abeja-toolkit
    existing_dirs=()
    for d in */ ; do
        # d is in the form "dirname/", so exclude the trailing slash
        dir_name="${d%/}"

        # If dir_name is not *, add to array
        if [ "$dir_name" != "*" ]; then
            existing_dirs+=("$dir_name")
        fi
    done
    cd ../
else
    existing_dirs=()
fi

# Remove abeja-toolkit if it exists
if [ -d abeja-toolkit ]; then
    rm -rf abeja-toolkit
fi

# Clone abeja-toolkit
if [ ! -d abeja-toolkit ]; then
    git clone --filter=blob:none --no-checkout git@github.com:abeja-inc/abeja-toolkit.git
fi

# Go to the abeja-toolkit directory
cd ./abeja-toolkit

# Combine arguments with existing directory name
final_args=("$@" "${existing_dirs[@]}")

# Execute sparse-checkout only if final_args is non-empty
if [ ${#final_args[@]} -gt 1 ]; then
    # sparse-checkout only the tools you need
    git sparse-checkout set "${final_args[@]}"
    git checkout

    # To avoid leaking out our assets through the git history of the abeja-toolkit
    rm -rf .git 
    cd ../../

    echo "Sync completed successfully."
    echo "The following tools have been synced:"
    for dir in "${final_args[@]}"; do
        echo "- $dir"
    done
else
    echo "No tools specified for sync."
    cd ../../
fi
