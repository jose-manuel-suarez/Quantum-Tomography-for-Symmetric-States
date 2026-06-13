"""Tomography routines: reconstruct density matrices from measurement data.

This module implements a robust least-squares reconstruction over an
orthogonal basis and projects the result to the PSD cone.
"""
from typing import List
import numpy as np
from . import utils
import os
import shutil
from datetime import datetime
import matplotlib.pyplot as plt


def load_basis(symmetry: str, N: int) -> List[np.ndarray]:
    path = f"Orthogonal_Basis/BasisOrt_{symmetry}_{N}.npz"
    data = np.load(path, allow_pickle=True)
    # preserve original ordering used in legacy code
    BasisOrt = [data[key] for key in data]
    return BasisOrt


def reconstruct_rho_ls(BasisOrt: List[np.ndarray], measured_obs: List[np.ndarray], means: List[float]) -> np.ndarray:
    # Build matrix M where M[j, i] = trace(Basis_i @ Obs_j)
    num_obs = len(measured_obs)
    num_basis = len(BasisOrt)
    M = np.zeros((num_obs, num_basis), dtype=complex)
    for j in range(num_obs):
        for i in range(num_basis):
            M[j, i] = np.trace(BasisOrt[i] @ measured_obs[j])
    m = np.array(means, dtype=complex)
    # Solve least squares M a = m
    try:
        a, *_ = np.linalg.lstsq(M, m, rcond=None)
    except Exception:
        a = np.linalg.pinv(M) @ m
    # Reconstruct
    RM0 = sum(a[i] * BasisOrt[i] for i in range(num_basis))
    RM = (RM0 + RM0.conj().T) / 2
    # Project to PSD and renormalize
    RM_psd = utils.project_to_psd(RM)
    return RM_psd


def _observables_from_keys(keys: List[str]) -> List[np.ndarray]:
    # Map characters to Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    MAP = {"I": I, "X": X, "Y": Y, "Z": Z}

    obs = []
    for k in keys:
        mats = [MAP[ch] for ch in k]
        res = mats[0]
        for m in mats[1:]:
            res = np.kron(res, m)
        obs.append(res)
    return obs


def run_tomography(measurements_folder: str, results_folder: str = None):
    """Run tomography over the data saved in `measurements_folder` and
    write results into `results_folder`.
    The `measurements_folder` is expected to match the structure produced
    by `quantum_tomography.measurements.simulate_measurements`.
    """
    # Base results directory for this measurements set
    if results_folder is None:
        results_base = os.path.join("Results", os.path.basename(measurements_folder))
    else:
        results_base = results_folder

    # Ensure base exists
    utils.ensure_dir(results_base)

    # Create a timestamped subdirectory for this run: shot_DD_MM_YY_HH_MM_SS
    timestamp = datetime.now().strftime("shot_%d_%m_%y_%H_%M_%S")
    results_folder = os.path.join(results_base, timestamp)

    # Clean up any legacy (flat) results directly under results_base so structure
    # remains consistent (preserve any existing timestamped runs prefixed with 'shot_')
    for child in os.listdir(results_base):
        if child.startswith("shot_"):
            continue
        child_path = os.path.join(results_base, child)
        # Skip the new timestamp folder if it somehow exists
        if os.path.abspath(child_path) == os.path.abspath(results_folder):
            continue
        try:
            if os.path.isdir(child_path):
                shutil.rmtree(child_path)
            else:
                os.remove(child_path)
        except Exception:
            print(f"Warning: failed to remove legacy results item: {child_path}")

    # Make the timestamped results folder
    utils.ensure_dir(results_folder)
    print(f"Results will be saved to: {os.path.abspath(results_folder)}")

    noise_types = utils.load_json(os.path.join(measurements_folder, "noise_types.txt"))
    noise_levels = utils.load_json(os.path.join(measurements_folder, "noise_levels.txt"))
    shots = utils.load_json(os.path.join(measurements_folder, "shots.txt"))

    # Determine state name and symmetry from folder name
    base = os.path.basename(measurements_folder)
    # Try to parse N and symmetry from parent folder name
    parts = base.split("_")
    # guess symmetry is last part
    symmetry = parts[-1]
    # Try to infer N from base
    N = 2
    # load basis
    BasisOrt = load_basis(symmetry, N)

    for noise in noise_types:
        print(f"Processing tomography for noise: {noise}")
        meas_folder = os.path.join(measurements_folder, noise)
        results_noise = os.path.join(results_folder, noise)
        utils.ensure_dir(results_noise)
        DM_ideal = utils.load_np(os.path.join(meas_folder, "DM_ideal.npy"))

        # load noise levels list for metadata
        noise_level = utils.load_json(os.path.join(meas_folder, "noise_level.txt"))

        for shot in shots:
            shot_file = os.path.join(meas_folder, f"shots{shot}.txt")
            measurements_data = utils.load_json(shot_file)
            # measurements_data: list of c elements, each is list over noise_levels of dicts

            # extract observable keys from first measurement
            data_keys = list(measurements_data[0][0].keys())
            measured_observables = _observables_from_keys(data_keys)

            Fid_shots = []
            Fid_shots_R = []
            Fid_shots_SD = []
            Fid_shots_R_SD = []

            # For each noise level index
            num_noise_levels = len(noise_level)
            # For each noise level we'll aggregate reconstructed RMs
            for n in range(num_noise_levels):
                reconstructed_list = []
                fidelities = []
                fidelities_R = []
                # For each measurement repetition
                for meas in measurements_data:
                    mean_values = meas[n]
                    means_vec = [mean_values[k] for k in data_keys]
                    RM = reconstruct_rho_ls(BasisOrt, measured_observables, means_vec)
                    reconstructed_list.append(RM)
                    # Fidelity with DM_ideal
                    fid = utils.fidelity(DM_ideal, RM)
                    fidelities.append(fid)
                    # If there is a saved noisy DM file available, try to load it
                    noisy_dm_path = os.path.join(meas_folder, f"DM_{n}.npy")
                    if os.path.exists(noisy_dm_path):
                        RHO_Real = utils.load_np(noisy_dm_path)
                        fidR = utils.fidelity(RHO_Real, RM)
                        fidelities_R.append(fidR)

                # Aggregate
                if len(fidelities) > 0:
                    Fidelity = float(np.mean(fidelities))
                    Fidelity_SD = float(np.std(fidelities, ddof=0))
                else:
                    Fidelity = 0.0
                    Fidelity_SD = 0.0
                if len(fidelities_R) > 0:
                    Fidelity_R = float(np.mean(fidelities_R))
                    Fidelity_R_SD = float(np.std(fidelities_R, ddof=0))
                else:
                    Fidelity_R = 0.0
                    Fidelity_R_SD = 0.0

                Fid_shots.append(Fidelity)
                Fid_shots_R.append(Fidelity_R)
                Fid_shots_SD.append(Fidelity_SD)
                Fid_shots_R_SD.append(Fidelity_R_SD)

                # Save reconstructed list for this noise level
                utils.save_np(os.path.join(results_noise, f"DMtomo_noise{n}_shots{shot}.npy"), np.array(reconstructed_list, dtype=object))

            # Save summarized fidelities similar to original
            final_result = {
                "Fidelities": Fid_shots,
                "Fidelities_SD": Fid_shots_SD,
                "Fidelities_R": Fid_shots_R,
                "Fidelities_R_SD": Fid_shots_R_SD,
            }
            utils.save_json(final_result, os.path.join(results_noise, f"Fidelities_shots{shot}.txt"))

            # Plotting: simple plot of fidelities vs noise parameter if possible
            try:
                noise_axis = [nl.get(list(nl.keys())[0], 0) if isinstance(nl, dict) else nl for nl in noise_level]
                plt.errorbar(noise_axis, Fid_shots_R, Fid_shots_R_SD, marker='o', linestyle='None', label=noise)
                plt.errorbar(noise_axis, Fid_shots, Fid_shots_SD, marker='o', linestyle='None', label=noise + ' Ideal')
                plt.xlabel('Noise level')
                plt.ylabel('Fidelity')
                plt.legend()
                plt.title(f"Tomography results shots={shot} noise={noise}")
                plt.savefig(os.path.join(results_noise, f"tomo_shots{shot}_{noise}.png"), dpi=200)
                plt.close()
            except Exception:
                pass

        print(f"Finished tomography for noise {noise}")

    print("Tomography run completed.")
    print(f"Final results folder: {os.path.abspath(results_folder)}")
