import os
import git
import logging
import subprocess
import argparse
import sys

logger = logging.getLogger()


def run_command(command, cwd=None):
    try:
        subprocess.run(command, check=True, cwd=cwd)
    except subprocess.CalledProcessError:
        logger.error(f"Failed to run command {command} in {cwd}")
        raise


def get_repo(repo_url: str, hash: str = ""):
    # Get the name of the repo and remove the .git extension
    repo_dir = repo_url.split("/")[-1].replace(".git", "")
    logger.info(f"Cloning or updating {repo_url} to {repo_dir}")
    repo_dir = os.path.abspath(repo_dir)
    if os.path.isdir(repo_dir):
        logger.info(f"{repo_dir} already exists. Updating...")
        repo = git.Repo(repo_dir)
        origin = repo.remotes.origin
        try:
            origin.pull()
            # Update submodules as well
            repo.git.submodule("update", "--init", "--recursive")
        except git.GitCommandError as e:
            logger.error(
                f"Failed to pull changes due to error: {e}. Resetting local branch to match remote..."
            )
            default_branch = repo.remotes.origin.refs[0].remote_head
            origin.fetch()
            repo.git.reset("--hard", f"origin/{default_branch}")
            repo.git.clean("-df")
        # Checkout specific hash if provided
        if hash:
            logger.info(f"Checking out commit {hash} in {repo_dir}")
            repo.git.checkout(hash)
    else:
        try:
            # Clone the repo including submodules
            # If hash is specified, don't use shallow clone as the hash might not be in recent history
            if hash:
                logger.info(f"Cloning full repo to checkout specific commit {hash}")
                git.Repo.clone_from(repo_url, repo_dir, recursive=True)
            else:
                git.Repo.clone_from(repo_url, repo_dir, depth=1, recursive=True)

            logger.info(f"Cloned {repo_dir} successfully.")
            repo = git.Repo(repo_dir)

            # Checkout specific hash if provided
            if hash:
                logger.info(f"Checking out commit {hash} in {repo_dir}")
                repo.git.checkout(hash)
        except git.GitCommandError as e:
            logger.error(f"Failed to clone {repo_dir} due to error: {e}")
            raise
    # If there is an install script, run it
    install_script = os.path.join(repo_dir, "install.py")
    if os.path.isfile(install_script):
        run_command(["python3", install_script], cwd=repo_dir)


if __name__ == "__main__":
    # Setup logging for CLI usage
    logging.basicConfig(
        level=logging.INFO,
        format='[get_repo] %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="Clone or update a git repository")
    parser.add_argument("repo_url", help="Git repository URL")
    parser.add_argument("--hash", default="", help="Specific commit hash to checkout")
    parser.add_argument("--target-dir", help="Target directory (default: derived from repo URL)")

    args = parser.parse_args()

    # Change to target directory if specified
    if args.target_dir:
        os.chdir(args.target_dir)

    try:
        get_repo(args.repo_url, args.hash)
        logger.info("Repository operation completed successfully")
    except Exception as e:
        logger.error(f"Failed to get repository: {e}")
        sys.exit(1)
