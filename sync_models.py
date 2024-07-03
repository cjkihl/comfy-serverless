import os
import subprocess
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Sync models from S3')
parser.add_argument('--folder', type=str, help='Sync folder')
args = parser.parse_args()

# Expects S3_ENDPOINT_URL as environment variable
S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
if not S3_ENDPOINT_URL:
    raise ValueError("S3_ENDPOINT_URL is not set")

# Use --folder parameter if provided, else fall back to SYNC_FOLDER environment variable
SYNC_FOLDER = args.folder if args.folder else os.getenv('SYNC_FOLDER' ,'1.5')

logging.info("Syncing models from S3")
model_names = ["checkpoints", "embeddings", "loras", "upscale_models", "controlnet", "vae", "vae_approx", "clip_vision", "ipadapter", "instantid","insightface"]
failed_syncs = []
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

try:
    dir = models_dir
    s3path = f"s3://stable-diffusion/{SYNC_FOLDER}"
    logging.info(f"Syncing {s3path} to {dir}")
    subprocess.check_call(['aws', 's3', 'sync', s3path, dir, '--endpoint-url', S3_ENDPOINT_URL])
except subprocess.CalledProcessError:
    logging.error(f"Sync of root failed")
    failed_syncs.append('root')

# for model_name in model_names:
#     try:
#         dir = os.path.join(models_dir, model_name)
#         s3path = f"s3://stable-diffusion/{SYNC_FOLDER}/{model_name}"
#         logging.info(f"Syncing {s3path} to {dir}")
#         subprocess.check_call(['aws', 's3', 'sync', s3path, dir, '--endpoint-url', S3_ENDPOINT_URL])
#     except subprocess.CalledProcessError:
#         logging.error(f"Sync of {model_name} failed")
#         failed_syncs.append(model_name)

if failed_syncs:
    logging.error(f"Failed to sync the following models: {', '.join(failed_syncs)}")
    exit(1)