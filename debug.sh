# python3 -m debugpy --listen 0.0.0.0:5678 "$@"
python -m debugpy --listen 5678 ./main.py

# To debug serverless remote
# python -m debugpy --listen 0.0.0.0:5678 serverless.py --listen --port 3000