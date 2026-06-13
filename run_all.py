"""Run the full workflow: measurements -> tomography -> tests."""
import subprocess
import sys
import os


def main():
    # Run measurements
    print('Running measurements...')
    subprocess.check_call([sys.executable, 'run_measurements.py'])

    # Run tomography
    print('Running tomography...')
    subprocess.check_call([sys.executable, 'run_tomography.py'])

    # Run tests (post-tomography checks)
    print('Running tests...')
    subprocess.check_call([sys.executable, '-m', 'unittest', 'discover', '-v'])


if __name__ == '__main__':
    main()
