#!/bin/bash

# Read the contents of workflow.json into a variable
data=$(cat workflow.json)

# Make the POST request and process the response line by line
curl -X POST -H "Content-Type: application/json" --no-buffer -d "$data" http://127.0.0.1:8188/v1/execute | while read -r line
do
  echo "Received: $line"
done