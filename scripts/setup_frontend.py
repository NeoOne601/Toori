#!/usr/bin/env python3
"""Install frontend dependencies, build the desktop shell, and launch Electron."""

import sys
import shutil
import subprocess
from pathlib import Path

FRONTEND_DIR = Path("desktop/electron")

def check_node_npm():
    """Ensure that both `node` and `npm` are in the PATH."""
    for cmd in ["node", "npm"]:
        if shutil.which(cmd) is None:
            print(f"Error: '{cmd}' is not installed or not in PATH. Please install Node.js (which provides both).")
            sys.exit(1)
    print("Node.js and npm are available.")

def npm_install():
    """Run `npm install` in the frontend directory."""
    if not (FRONTEND_DIR / "package.json").exists():
        print(f"No package.json found in {FRONTEND_DIR}. Cannot install dependencies.")
        sys.exit(1)
    print(f"Running 'npm install' in {FRONTEND_DIR} ...")
    subprocess.check_call(["npm", "install"], cwd=FRONTEND_DIR)
    print("npm dependencies installed.")

def launch_electron():
    print("Building desktop shell ...")
    subprocess.check_call(["npm", "run", "build"], cwd=FRONTEND_DIR)
    print("Launching Electron via 'npm start' ...")
    subprocess.Popen(["npm", "start"], cwd=FRONTEND_DIR)

def main():
    print("=== Frontend (Electron) setup script ===")
    check_node_npm()
    npm_install()
    launch_electron()
    print("Electron app launched. Use Ctrl‑C in this terminal to stop the script (the app will keep running until closed).")

if __name__ == "__main__":
    main()
