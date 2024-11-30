import os
import git
import logging
import subprocess

logger = logging.getLogger()

def run_command(command, cwd=None):
    try:
        subprocess.run(command, check=True, cwd=cwd)
    except subprocess.CalledProcessError:
        logger.error(f"Failed to run command {command} in {cwd}")
        raise

def get_repo(repo_url:str):
    # Get the name of the repo and remove the .git extension
    repo_dir = repo_url.split('/')[-1].replace('.git', '')
    logger.info(f"Cloning or updating {repo_url} to {repo_dir}")
    repo_dir = os.path.abspath(repo_dir)
    if os.path.isdir(repo_dir):
        logger.info(f"{repo_dir} already exists. Updating...")
        repo = git.Repo(repo_dir)
        origin = repo.remotes.origin
        try:
            origin.pull()
            # Update submodules as well
            repo.git.submodule('update', '--init', '--recursive')
        except git.GitCommandError as e:
            logger.error(f"Failed to pull changes due to error: {e}. Resetting local branch to match remote...")
            default_branch = repo.remotes.origin.refs[0].remote_head
            origin.fetch()
            repo.git.reset('--hard', f"origin/{default_branch}")
            repo.git.clean('-df')
    else:
        try:
            # Clone the repo including submodules
            git.Repo.clone_from(repo_url, repo_dir, depth=1, recursive=True)
            logger.info(f"Cloned {repo_dir} successfully.")
        except git.GitCommandError as e:
            logger.error(f"Failed to clone {repo_dir} due to error: {e}")
            raise
    # If there is an install script, run it
    install_script = os.path.join(repo_dir, "install.py")
    if os.path.isfile(install_script):
        run_command(["python3", install_script], cwd=repo_dir)
