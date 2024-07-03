#!/bin/bash

# Check if pip or pip3 is available
PIP=$(command -v pip || command -v pip3) || { echo "pip is not installed. Exiting."; exit 1; }

# Change the current working directory to the directory where the script is located
pushd "$(dirname "$0")" >/dev/null || exit 1

# Install extensions
python ./install_extensions.py

# Check if requirements.txt exists in the current directory
if [ ! -f requirements.txt ]; then
    echo "Error: requirements.txt not found in the current directory."
    exit 1
fi

# Install requirements
echo "Installing requirements from requirements.txt"
$PIP install -r ./requirements.txt

# Install all requirements for custom nodes
if [ -d custom_nodes ]; then
    find custom_nodes -name 'requirements.txt' -type f | while read -r file
    do
        echo "Installing requirements from $file"
        $PIP install -r "$file"
    done
else
    echo "Warning: custom_nodes directory not found. Skipping installation of custom node requirements."
fi

popd >/dev/null