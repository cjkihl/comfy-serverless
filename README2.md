run
python3 -m venv venv
source venv/bin/activate
python ./install_extensions.py
python ./sync_models.py --folder xl

# To start
python main.py --listen --port 4000


# To start in prod with supervisor
supervisorctl start comfy

# TO update from upstream
git fetch upstream
 git rebase upstream/master


add-detail-xl.safetensors


 # Download model from CivitAI
```sh 
model_name="ponyRealism_V22MainVAE.safetensors"
url="https://civitai.com/api/download/models/914390?type=Model&format=SafeTensor&size=full&fp=fp16"
bearer_token="96883c9da9f368d8b1d00554fe2b1c3a"
curl -L -o ./${model_name} -H "Authorization: Bearer ${bearer_token}" -X GET ${url}
```

```sh 
model_name="easynegative.safetensors"
url="https://civitai.com/api/download/models/9208?type=Model&format=SafeTensor&size=full&fp=fp16"
bearer_token="96883c9da9f368d8b1d00554fe2b1c3a"
curl -L -o ./${model_name} -H "Authorization: Bearer ${bearer_token}" -X GET ${url}
```


Upload to R2

```sh
S3_ENDPOINT=https://8afb1da9709a0795ed70199046e27446.r2.cloudflarestorage.com
FILE=shape_predictor_81_face_landmarks.dat
aws s3 cp "$FILE" "s3://stable-diffusion/xl/dlib/$FILE" --endpoint-url $S3_ENDPOINT
```


tmux new-session -s ui

tmux ls

tmux attach-session -t session_name