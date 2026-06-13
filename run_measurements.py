"""Run measurement simulation using configuration from .env (or defaults)."""
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
from quantum_tomography import measurements


def parse_list(env_val, cast=int):
    if not env_val:
        return None
    return [cast(x.strip()) for x in env_val.split(',') if x.strip()]


def main():
    load_dotenv()
    state_type = os.getenv('STATE_TYPE', 'GHZ')
    N = int(os.getenv('N', '2'))
    s = int(os.getenv('S', '25'))
    symmetry = os.getenv('SYMMETRY', 'Permutational')
    c = int(os.getenv('C', '30'))
    shots = parse_list(os.getenv('SHOTS', '100,300,500'))
    noise_types = os.getenv('NOISE_TYPES', 'Depolarizing,BitFlip,AmplitudeDamping').split(',')
    noise_list_str = os.getenv('NOISE_LIST', '')
    noise_list = None
    if noise_list_str:
        noise_list = [float(x) for x in noise_list_str.split(',') if x.strip()]

    measurements.simulate_measurements(state_type=state_type, N=N, s=s, symmetry=symmetry, c=c, shots=shots, noise_types=noise_types, noise_list=noise_list)


if __name__ == '__main__':
    main()
