import os
import shutil
import subprocess

def push_to_github():
    gh_token = os.environ.get('GH_TOKEN')
    if not gh_token:
        print("Error: GH_TOKEN environment variable is not set.")
        return

    username = os.environ.get('GITHUB_USERNAME', 'your_username')
    repo_name = os.environ.get('GITHUB_REPO', 'trajectory-prediction')
    branch = os.environ.get('GITHUB_BRANCH', 'main')

    repo_url = f"https://{gh_token}@github.com/{username}/{repo_name}.git"
    local_repo_path = f"/tmp/{repo_name}"

    if os.path.exists(local_repo_path):
        print(f"Repository already cloned at {local_repo_path}. Pulling latest changes...")
        subprocess.run(["git", "-C", local_repo_path, "pull", "origin", branch])
    else:
        print(f"Cloning repository into {local_repo_path}...")
        subprocess.run(["git", "clone", repo_url, local_repo_path])
        subprocess.run(["git", "-C", local_repo_path, "checkout", branch])

    source_model = 'checkpoints/best_model.pt'
    dest_model = os.path.join(local_repo_path, 'best_model.pt')
    
    if os.path.exists(source_model):
        shutil.copy(source_model, dest_model)
        print(f"Copied {source_model} to {dest_model}")
    else:
        print(f"Warning: {source_model} not found.")

    subprocess.run(["git", "-C", local_repo_path, "config", "user.name", username])
    subprocess.run(["git", "-C", local_repo_path, "config", "user.email", f"{username}@example.com"])
    
    subprocess.run(["git", "-C", local_repo_path, "add", "best_model.pt"])
    subprocess.run(["git", "-C", local_repo_path, "commit", "-m", "Add trained PyTorch best_model.pt"])
    subprocess.run(["git", "-C", local_repo_path, "push", repo_url, f"HEAD:{branch}", "--force-with-lease"])
    print("GitHub push process completed.")

if __name__ == "__main__":
    push_to_github()
