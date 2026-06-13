"""Measurement simulation utilities (backend-agnostic).

These functions simulate state preparation and measurements and save data
in the same folder layout expected by the tomography code:
  Measurements/<state>_<symmetry>/<noise_type>/shots{shot}.txt

The simulation is lightweight and does not require Amazon Braket.
"""
from typing import List, Tuple
import numpy as np
from . import utils

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = {"I": I, "X": X, "Y": Y, "Z": Z}


def pauli_kron(pattern: str) -> np.ndarray:
    mats = [PAULIS[ch] for ch in pattern]
    res = mats[0]
    for m in mats[1:]:
        res = np.kron(res, m)
    return res


def observables(N: int, symmetry: str = "Permutational") -> List[str]:
    import itertools

    if symmetry == "Complete":
        return ["".join(p) for p in itertools.product("IXYZ", repeat=N)]
    else:
        # Permutational: combinations with replacement
        return ["".join(p) for p in itertools.combinations_with_replacement("IXYZ", N)]


def ghz_density(N: int) -> np.ndarray:
    dim = 2 ** N
    vec = np.zeros((dim,), dtype=complex)
    vec[0] = 1
    vec[-1] = 1
    vec = vec / np.linalg.norm(vec)
    return np.outer(vec, vec.conj())


def werner_density(s: int) -> np.ndarray:
    # s indexes into 50 possible p values (0..49)
    ps = np.linspace(0, 1, 50)
    p = float(ps[s])
    # singlet for two qubits: (|01> - |10>)/sqrt(2)
    psi = np.zeros((4,), dtype=complex)
    psi[1] = 1
    psi[2] = -1
    psi = psi / np.linalg.norm(psi)
    proj = np.outer(psi, psi.conj())
    return p * proj + (1 - p) * np.eye(4) / 4


def apply_depolarizing(rho: np.ndarray, p: float) -> np.ndarray:
    d = rho.shape[0]
    return (1 - p) * rho + p * np.eye(d) / d


def apply_bitflip(rho: np.ndarray, p: float, N: int) -> np.ndarray:
    # Apply bitflip independently on each qubit
    out = rho.copy()
    for i in range(N):
        # build X_i
        ops = [I] * N
        ops[i] = X
        X_i = ops[0]
        for op in ops[1:]:
            X_i = np.kron(X_i, op)
        out = (1 - p) * out + p * (X_i @ out @ X_i)
    return out


def apply_amplitude_damping(rho: np.ndarray, gamma: float, N: int) -> np.ndarray:
    # Apply amplitude damping channel on each qubit sequentially
    def single_qubit_amplitude_map(rho_in, target_qubit):
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
        ops0 = [I] * N
        ops1 = [I] * N
        ops0[target_qubit] = K0
        ops1[target_qubit] = K1
        K0_full = ops0[0]
        for op in ops0[1:]:
            K0_full = np.kron(K0_full, op)
        K1_full = ops1[0]
        for op in ops1[1:]:
            K1_full = np.kron(K1_full, op)
        return K0_full @ rho_in @ K0_full.conj().T + K1_full @ rho_in @ K1_full.conj().T

    out = rho.copy()
    for q in range(N):
        out = single_qubit_amplitude_map(out, q)
    return out


def simulate_measurements(
    state_type: str = "GHZ",
    N: int = 2,
    s: int = 25,
    symmetry: str = "Permutational",
    c: int = 30,
    shots: List[int] = [100, 300],
    noise_types: List[str] = None,
    noise_list: List[float] = None,
    save_root: str = "Measurements",
):
    """Simulate measurements and save them in the Measurements/ folder.

    The saved format mirrors the original scripts so the tomography
    module can consume the results unchanged.
    """
    if noise_types is None:
        noise_types = ["Depolarizing", "BitFlip", "AmplitudeDamping"]
    if noise_list is None:
        noise_list = np.linspace(0, 0.15, 7).tolist()

    # Prepare state
    if state_type == "GHZ":
        DM_ideal = ghz_density(N)
        state_name = f"GHZ_{N}qubits"
    else:
        N = 2
        DM_ideal = werner_density(s)
        state_name = f"Werner_s{s}_{N}qubits"

    L0 = observables(N, symmetry)

    mother_folder = f"{save_root}/{state_name}_{symmetry}"
    utils.ensure_dir(mother_folder)
    utils.save_json(noise_types, f"{mother_folder}/noise_types.txt")
    utils.save_json(noise_list, f"{mother_folder}/noise_levels.txt")
    utils.save_json(shots, f"{mother_folder}/shots.txt")

    for noise_type in noise_types:
        folder = f"{mother_folder}/{noise_type}"
        utils.ensure_dir(folder)
        # Save DM_ideal
        utils.save_np(f"{folder}/DM_ideal.npy", DM_ideal)

        # Build noise levels (dictionaries) similar to original layout
        noise_levels = []
        for p in noise_list:
            nd = {
                "Depolarizing_gateH": p / 10,
                "Depolarizing_gateCNot": p,
                "Depolarizing_gateRy": p / 10,
                "BitFlip_gateH": 0.0,
                "BitFlip_gateCNot": 0.0,
                "BitFlip_gateRy": 0.0,
                "AmplitudeDamping_gateH": 0.0,
                "AmplitudeDamping_gateCNot": 0.0,
                "AmplitudeDamping_gateRy": 0.0,
            }
            if noise_type == "BitFlip":
                nd = {k: 0.0 for k in nd}
                nd["BitFlip_gateH"] = p / 10
                nd["BitFlip_gateCNot"] = p
                nd["BitFlip_gateRy"] = p / 10
            if noise_type == "AmplitudeDamping":
                nd = {k: 0.0 for k in nd}
                nd["AmplitudeDamping_gateH"] = p / 10
                nd["AmplitudeDamping_gateCNot"] = p
                nd["AmplitudeDamping_gateRy"] = p / 10
            noise_levels.append(nd)

        utils.save_json(noise_levels, f"{folder}/noise_level.txt")

        for shot in shots:
            total_data = []
            for _ in range(c):
                per_noise = []
                for nd in noise_levels:
                    # Apply noise to DM_ideal
                    if noise_type == "Depolarizing":
                        p = nd["Depolarizing_gateCNot"]
                        DM_noisy = apply_depolarizing(DM_ideal, p)
                    elif noise_type == "BitFlip":
                        p = nd["BitFlip_gateCNot"]
                        DM_noisy = apply_bitflip(DM_ideal, p, N)
                    else:
                        p = nd["AmplitudeDamping_gateCNot"]
                        DM_noisy = apply_amplitude_damping(DM_ideal, p, N)

                    # Compute mean values for all observables in L0
                    Mean_Values = {}
                    for m in L0:
                        O = pauli_kron(m)
                        val = np.real(np.trace(DM_noisy @ O))
                        # Add sampling noise according to shots (approx Gaussian)
                        std = np.sqrt(max(0, 1 - val ** 2) / max(1, shot))
                        noisy_val = float(np.clip(np.random.normal(val, std), -1, 1))
                        Mean_Values[m] = noisy_val
                    per_noise.append(Mean_Values)
                total_data.append(per_noise)
            utils.save_json(total_data, f"{folder}/shots{shot}.txt")
