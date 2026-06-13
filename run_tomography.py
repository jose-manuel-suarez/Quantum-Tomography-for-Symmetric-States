"""Run tomography over an existing Measurements folder (configured via .env)."""
import os
import sys
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    print("Missing dependency 'python-dotenv'. Install it with:")
    print("  python -m pip install --user python-dotenv")
    print("Or create the project venv and install core requirements:")
    print("  python scripts/setup_env.py --core")
    sys.exit(1)
from quantum_tomography import tomography


def main():
    load_dotenv()
    measurements_root = os.getenv('MEASUREMENTS_ROOT', 'Measurements')
    state = os.getenv('STATE', '')
    symmetry = os.getenv('SYMMETRY', 'Permutational')
    if state:
        measurements_folder = os.path.join(measurements_root, f"{state}_{symmetry}")
    else:
        # Find first matching folder
        candidates = [p for p in os.listdir(measurements_root) if p.endswith(symmetry)]
        if not candidates:
            raise FileNotFoundError('No measurements folder found; run measurements first or set MEASUREMENTS_ROOT/STATE in .env')
        measurements_folder = os.path.join(measurements_root, candidates[0])

    results_folder = os.getenv('RESULTS_ROOT', 'Results')
    tomography.run_tomography(measurements_folder, results_folder)


if __name__ == '__main__':
    main()
