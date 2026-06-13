import json
import os
import numpy as np
from scipy.linalg import sqrtm, eigvals


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_json(obj, filename):
    ensure_dir(os.path.dirname(filename) or '.')
    with open(filename, 'w') as f:
        json.dump(obj, f)


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def save_np(filename, arr):
    ensure_dir(os.path.dirname(filename) or '.')
    np.save(filename, arr)


def load_np(filename):
    return np.load(filename, allow_pickle=True)


def fidelity(A, B):
    """Uhlmann fidelity for density matrices A and B."""
    A = np.array(A, dtype=complex)
    B = np.array(B, dtype=complex)
    # Numerical safeguard
    try:
        root = sqrtm(A)
        inter = sqrtm(root @ B @ root)
        val = np.real(np.trace(inter))
        return float((val) ** 2)
    except Exception:
        # Fallback using eigenvalues of sqrt(A)B sqrt(A)
        vals = eigvals(A)
        vals = np.real(vals)
        return float(np.sum(np.sqrt(np.abs(vals))) ** 2)


def project_to_psd(rho):
    """Project a Hermitian matrix onto the PSD cone and renormalize trace to 1."""
    rho = np.array(rho, dtype=complex)
    rho = (rho + rho.conj().T) / 2
    w, v = np.linalg.eigh(rho)
    w_clipped = np.clip(w, 0, None)
    rho_psd = (v @ np.diag(w_clipped) @ v.conj().T)
    tr = np.trace(rho_psd)
    if tr == 0:
        # fallback to maximally mixed
        d = rho.shape[0]
        return np.eye(d) / d
    return rho_psd / tr
