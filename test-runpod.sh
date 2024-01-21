# Load file workflow.json into a variable
workflow=$(cat workflow.json)

# python3 handler.py --test_input "{\"input\": $workflow }"
# python3 handler.py --rp_serve_api

#curl -X POST -H "Content-Type: application/json" -d "{\"input\": $workflow }" http://localhost:8000/run

curl -N -X POST http://localhost:8000/stream/test-68c421ac-0b6f-4519-a5e4-147b0749b1c7 | while read line; do
    echo $line
done


#curl -X POST http://localhost:8000/status/test-eca5b60c-b7af-43cd-8d6b-7cd76c4595c4