# Before executing this file Start serverless with debugpy
# pip install -r requirements.txt
# python -m debugpy --listen 0.0.0.0:5678 serverless.py --listen --port 3001
# Attach debugger in vscode
# Run this file

# Load file workflow.json into a variable
workflow=$(cat debug.json)

# python3 handler.py --test_input "{\"input\": $workflow }"
# python3 handler.py --rp_serve_api

#curl -X POST -H "Content-Type: application/json" -d "{\"input\": $workflow }" http://localhost:8000/runsync

curl -X POST -H "Content-Type: application/json" -d "$workflow" http://127.0.0.1:3001/v1/execute
# curl -X POST -H "Content-Type: application/json" -d "$workflow" https://uhziyoae8u7280-3000.proxy.runpod.net/v1/execute

# curl -N -X POST http://localhost:8000/stream/test-dc23c8da-bfe8-4e5d-8581-4ff5864b3959 | while read line; do
#     echo $line
# done


#curl -X POST http://localhost:8000/status/test-eca5b60c-b7af-43cd-8d6b-7cd76c4595c4