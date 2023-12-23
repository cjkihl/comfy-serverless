#!/bin/bash

pip install -r requirements.txt

# Install all requirements for custom nodes
find custom_nodes -name 'requirements.txt' -type f | while read file
do
    echo "Installing requirements from $file"
    pip install -r "$file"
done