"""Setup helper: create a virtual environment and install requirements.

Usage:
    python scripts/setup_env.py [--core]

--core: install only the minimal core requirements (skips optional/legacy
packages like qutip, cvxpy, cvxopt, amazon-braket-sdk). Use this on Windows
when pip builds fail for heavy compiled packages.
"""
import os
import sys
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--core",
        action="store_true",
        help="Install only core requirements (skip optional/legacy packages)",
    )
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    venv_dir = os.path.join(root, ".venv")
    print("Creating virtual environment in", venv_dir)
    subprocess.check_call([sys.executable, "-m", "venv", venv_dir])

    if os.name == "nt":
        py = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        py = os.path.join(venv_dir, "bin", "python")

    reqs = os.path.join(root, "requirements-core.txt" if args.core else "requirements.txt")

    print("Installing requirements using", py)
    # Upgrade pip/setuptools/wheel first to improve build reliability
    subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # Try installing the requested requirements file. If it fails (common on
    # Windows when optional packages require compiled wheels), fall back to the
    # minimal core requirements so the project remains usable.
    try:
        subprocess.check_call([py, "-m", "pip", "install", "-r", reqs])
    except subprocess.CalledProcessError:
        print("\nFull requirements installation failed. This often happens when")
        print("some optional packages require compiled wheels (qutip/cvxpy).")
        print("Falling back to minimal/core requirements to ensure the project")
        print("has the essential runtime dependencies (including python-dotenv).")
        core_reqs = os.path.join(root, "requirements-core.txt")
        subprocess.check_call([py, "-m", "pip", "install", "-r", core_reqs])
        # ensure dotenv is available
        try:
            subprocess.check_call([py, "-m", "pip", "install", "python-dotenv"])
        except subprocess.CalledProcessError:
            print("Warning: failed to install python-dotenv. You may need to")
            print("install it manually in the venv: pip install python-dotenv")

    print("\nSetup complete. Activate the environment:")
    if os.name == "nt":
        print(f"{venv_dir}\\Scripts\\Activate.ps1  # PowerShell")
        print(f"{venv_dir}\\Scripts\\activate.bat   # cmd.exe")
    else:
        print(f"source {venv_dir}/bin/activate")


if __name__ == "__main__":
    main()
