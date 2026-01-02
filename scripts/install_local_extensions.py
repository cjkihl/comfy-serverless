import os
import subprocess
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)

# Get paths (parent directory of scripts/)
script_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
cj_nodes_dir = os.path.join(script_dir, "cj_nodes")
custom_nodes_dir = os.path.join(script_dir, "custom_nodes")

# Ensure custom_nodes exists
os.makedirs(custom_nodes_dir, exist_ok=True)

# First, cleanup any stale symlinks
logging.info("Cleaning up stale symlinks...")
cleanup_script = os.path.join(script_dir, "scripts", "cleanup_stale_nodes.py")
if os.path.exists(cleanup_script):
    try:
        subprocess.run([sys.executable, cleanup_script], check=True)
    except subprocess.CalledProcessError as e:
        logging.warning(f"Cleanup script failed: {e}")

# Check if cj_nodes directory exists
if not os.path.exists(cj_nodes_dir):
    logging.info("cj_nodes directory not found, skipping local extensions installation")
    exit(0)

# Get all packages in cj_nodes
packages = [
    d
    for d in os.listdir(cj_nodes_dir)
    if os.path.isdir(os.path.join(cj_nodes_dir, d)) and not d.startswith("__")
]

if not packages:
    logging.info("No packages found in cj_nodes directory")
    exit(0)

logging.info(f"Found {len(packages)} local extension packages: {', '.join(packages)}")

for package in packages:
    source = os.path.join(cj_nodes_dir, package)
    target = os.path.join(custom_nodes_dir, package)

    # Remove existing symlink/dir if exists
    if os.path.islink(target):
        logging.info(f"Removing existing symlink: {target}")
        os.unlink(target)
    elif os.path.exists(target):
        logging.warning(
            f"Target exists and is not a symlink: {target}. Skipping {package}."
        )
        continue

    # Create symlink using relative path
    relative_source = os.path.relpath(source, custom_nodes_dir)
    os.symlink(relative_source, target)
    logging.info(f"✓ Symlinked {package} into custom_nodes/")

    # Install requirements
    req_file = os.path.join(source, "requirements.txt")
    if os.path.exists(req_file):
        # Check if requirements file is not empty or just comments
        with open(req_file, "r") as f:
            content = f.read().strip()
            # Skip if file is empty or only contains comments
            if not content or all(
                line.startswith("#") or not line.strip() for line in content.split("\n")
            ):
                logging.info(f"  No requirements to install for {package}")
                continue

        logging.info(f"  Installing requirements for {package}...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", req_file],
                capture_output=True,
                text=True,
                check=True,
            )
            logging.info(f"  ✓ Requirements installed for {package}")
        except subprocess.CalledProcessError as e:
            logging.error(f"  ✗ Failed to install requirements for {package}: {e}")
            if e.stdout:
                logging.error(f"  stdout: {e.stdout}")
            if e.stderr:
                logging.error(f"  stderr: {e.stderr}")
    else:
        logging.info(f"  No requirements.txt found for {package}")

logging.info("Local extensions installation complete!")

