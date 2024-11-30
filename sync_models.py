import os
import subprocess
import logging
import argparse
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up argument parser
parser = argparse.ArgumentParser(description="Sync models from S3")
parser.add_argument('--sync-folder', type=str, help='The sync folder name')
args = parser.parse_args()

# Expects S3_ENDPOINT_URL as environment variable
S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
if not S3_ENDPOINT_URL:
    raise ValueError("S3_ENDPOINT_URL is not set")

# Use the provided folder name from argument, else fall back to SYNC_FOLDER environment variable
SYNC_FOLDER = args.sync_folder or os.getenv('SYNC_FOLDER')

# If SYNC_FOLDER is still not set, ask the user for the folder name
if not SYNC_FOLDER:
    SYNC_FOLDER = input("Please enter the sync folder name: ")

logging.info("Syncing models from S3")
failed_syncs = []
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

try:
    dir = models_dir
    s3path = f"s3://stable-diffusion/{SYNC_FOLDER}"
    logging.info(f"Syncing {s3path} to {dir}")
    subprocess.check_call(['aws', 's3', 'sync', s3path, dir, '--endpoint-url', S3_ENDPOINT_URL])
except subprocess.CalledProcessError:
    logging.error("Sync of root failed")
    failed_syncs.append('root')

if failed_syncs:
    logging.error(f"Failed to sync the following models: {', '.join(failed_syncs)}")
    sys.exit(1)