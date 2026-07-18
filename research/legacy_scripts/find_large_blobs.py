import subprocess
import os
import sys

def run_cmd(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    if result.returncode != 0:
        print(f"Error running: {cmd}\n{result.stderr}")
        return ""
    return result.stdout

def main():
    # Find all commits from origin/loso_new to HEAD
    commits_str = run_cmd("git rev-list origin/loso_new..HEAD")
    commits = [c.strip() for c in commits_str.strip().split('\n') if c.strip()]
    
    print(f"Checking {len(commits)} commits for files > 50MB...")
    
    large_files = {}
    for commit in commits:
        # Get list of files and their sizes/objects in this commit
        files_str = run_cmd(f"git ls-tree -r -l {commit}")
        for line in files_str.strip().split('\n'):
            if not line:
                continue
            # Format is: <mode> <type> <object> <size>\t<file>
            parts = line.split(maxsplit=4)
            if len(parts) < 5:
                continue
            mode, obj_type, sha, size_str, filepath = parts[0], parts[1], parts[2], parts[3], parts[4]
            # Some entries might not have size (e.g. submodules or directories, though -r should expand directories)
            if size_str == '-':
                continue
            try:
                size = int(size_str)
            except ValueError:
                continue
            
            if size > 50 * 1024 * 1024: # 50 MB
                if filepath not in large_files:
                    large_files[filepath] = []
                large_files[filepath].append((commit, size))
                
    if not large_files:
        print("No files > 50MB found in the commits ahead of origin/loso_new.")
    else:
        print("Found the following large files in commits:")
        for filepath, occurrences in large_files.items():
            print(f"\nFile: {filepath}")
            for commit, size in occurrences:
                commit_info = run_cmd(f"git log -1 --oneline {commit}").strip()
                print(f"  Commit: {commit_info}")
                print(f"  Size: {size / (1024 * 1024):.2f} MB")

if __name__ == '__main__':
    main()
