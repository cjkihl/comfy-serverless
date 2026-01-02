#!/bin/bash

# Change the current working directory to the project root (parent of scripts/)
pushd "$(dirname "$0")/.." >/dev/null || exit 1

# Check if requirements.txt exists in the current directory
if [ ! -f requirements.txt ]; then
    echo "Error: requirements.txt not found in the current directory."
    exit 1
fi

# Install requirements first (needed by install_extensions.py)
# Use python -m pip to ensure we use the correct pip from the Pixi environment
echo "Installing requirements from requirements.txt"
python -m pip install -r ./requirements.txt

# Install extensions (requires GitPython from requirements.txt)
python ./scripts/install_extensions.py

# Install local CJ nodes
python ./scripts/install_local_extensions.py

# Install all requirements for custom nodes
if [ -d custom_nodes ]; then
    find custom_nodes -name 'requirements.txt' -type f | while read -r file
    do
        echo "Installing requirements from $file"
        python -m pip install -r "$file"
    done
else
    echo "Warning: custom_nodes directory not found. Skipping installation of custom node requirements."
fi

popd >/dev/null