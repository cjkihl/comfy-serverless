#!/bin/bash

PIP=pip

if ! command -v $PIP &> /dev/null
then
    PIP=pip3
fi

$PIP install -r requirements.txt

# Install all requirements for custom nodes
find custom_nodes -name 'requirements.txt' -type f | while read file
do
    echo "Installing requirements from $file"
    $PIP install -r "$file"
done