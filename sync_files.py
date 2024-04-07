import os
import boto3
from botocore.exceptions import NoCredentialsError


def download_dir(prefix, local, bucket, client):
    keys = []
    next_token = ""
    base_kwargs = {
        "Bucket": bucket,
        "Prefix": prefix,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != "":
            kwargs.update({"ContinuationToken": next_token})
        results = client.list_objects_v2(**kwargs)
        contents = results.get("Contents")
        if contents is None:
            raise FileNotFoundError("Folder does not exist", prefix)
        for i in contents:
            k = i.get("Key")
            if k[-1] != "/":
                keys.append(k)
        next_token = results.get("NextContinuationToken")
    for k in keys:
        dest_pathname = os.path.join(local, k[len(prefix) :])
        print("key", dest_pathname)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        try:
            if (
                not os.path.exists(dest_pathname)
                or os.path.getmtime(dest_pathname) < i.get("LastModified").timestamp()
            ):
                print(f"Downloading {dest_pathname}")
                client.download_file(bucket, k, dest_pathname)
        except NoCredentialsError:
            print("Credentials not available")
            return False
    return True


s3 = boto3.client(
    service_name="s3",
    aws_access_key_id="a5b3414299cdab6a8d81ecd9e5095b14",
    aws_secret_access_key="77465d67e7a9ac19dd84400dc5d9fcd5b300facb98703ca964a2624745a9beef",
    endpoint_url="https://8afb1da9709a0795ed70199046e27446.r2.cloudflarestorage.com",
    region_name="auto",  # Must be one of: wnam, enam, weur, eeur, apac, auto
)

download_dir("1.5/vae/", "models/vae/", "stable-diffusion", s3)
download_dir("1.5/models/", "models/checkpoints/", "stable-diffusion", s3)
download_dir("1.5/embeddings/", "models/embeddings/", "stable-diffusion", s3)
download_dir("1.5/loras/", "models/loras/", "stable-diffusion", s3)
download_dir("1.5/upscale_models/", "models/upscale_models/", "stable-diffusion", s3)
download_dir("1.5/controlnet/", "models/controlnet/", "stable-diffusion", s3)
download_dir("1.5/clip_vision/", "models/clip_vision/", "stable-diffusion", s3)
download_dir("1.5/ipadapter/", "models/ipadapter/", "stable-diffusion", s3)
