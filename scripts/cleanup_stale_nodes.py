import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Get paths
script_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
custom_nodes_dir = os.path.join(script_dir, "custom_nodes")

if not os.path.exists(custom_nodes_dir):
    logging.info("custom_nodes directory not found, nothing to clean up")
    exit(0)

# Find and remove broken symlinks
removed_count = 0
for item in os.listdir(custom_nodes_dir):
    item_path = os.path.join(custom_nodes_dir, item)

    # Check if it's a symlink
    if os.path.islink(item_path):
        # Check if the symlink is broken (target doesn't exist)
        if not os.path.exists(item_path):
            logging.info(f"Removing broken symlink: {item}")
            try:
                os.unlink(item_path)
                removed_count += 1
            except OSError as e:
                logging.error(f"Failed to remove {item}: {e}")

if removed_count > 0:
    logging.info(f"Cleanup complete! Removed {removed_count} stale symlink(s)")
else:
    logging.info("No stale symlinks found")

