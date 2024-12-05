import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

from get_repo import get_repo

# Define the directory for custom nodes
custom_nodes_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "custom_nodes"
)

# Create the custom nodes directory if it doesn't exist
os.makedirs(custom_nodes_dir, exist_ok=True)

# Change directory to custom_nodes_dir
os.chdir(custom_nodes_dir)

# Setup each plugin by cloning or updating its repository and running its install script
# get_repo("https://github.com/ltdrdata/ComfyUI-Impact-Pack.git")
# get_repo("https://github.com/ltdrdata/ComfyUI-Inspire-Pack.git")
# get_repo("https://github.com/Fannovel16/comfyui_controlnet_aux.git")
# get_repo("https://github.com/kijai/ComfyUI-SUPIR.git")
# get_repo("https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git")
get_repo("https://github.com/cubiq/ComfyUI_IPAdapter_plus.git")
# get_repo(repo_url="https://github.com/cubiq/ComfyUI_InstantID.git")
# get_repo("https://github.com/cubiq/ComfyUI_essentials.git")
# get_repo("https://github.com/cubiq/ComfyUI_FaceAnalysis.git")
# get_repo("https://github.com/ZHO-ZHO-ZHO/ComfyUI-BRIA_AI-RMBG.git")
# get_repo("https://github.com/huchenlei/ComfyUI-IC-Light-Native.git")
# get_repo("https://github.com/WASasquatch/was-node-suite-comfyui.git")

# Special dir symlinks. This is a workaround for the fact that the custom nodes are not in the same directory as the models.
# import subprocess
# try:
#     # Remove the directory
#     subprocess.run(['rm', '-rf', '/comfy/custom_nodes/ComfyUI_FaceAnalysis/dlib'], check=True)
#     # Create a symbolic link
#     subprocess.run(['ln', '-s', '/comfy/models/dlib', '/comfy/custom_nodes/ComfyUI_FaceAnalysis/dlib'], check=True)
#     print("Commands executed successfully.")
# except subprocess.CalledProcessError as e:
#     print(f"An error occurred: {e}")
