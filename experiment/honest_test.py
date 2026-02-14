import argparse
import csv
import time
import numpy as np
import hashlib
from lwe_core import decrypt_lwe, int_centered
from transforms import derive_alpha, T_quadratic, T_identity, T_cyclotomic_toy, T_matrix_lift

def decrypt_sample_row(row):
    # row columns: trial,transform,n,q,sigma,a_hex,b,alpha,e,m,seed_hex
    _, transform, n, q, sigma, a_hex, b_str, alpha, e, m, seed_hex = row
    n = int(n); q = int(q); sigma = float(sigma)
    b = int(b_str)
    a = np.frombuffer(bytes.fromhex(a_hex), dtype=np.uint8)
    # a was stored as raw bytes; recover ints (we used numpy randint -> bytes)
    # Simpler deserialize: reconstruct by interpreting bytes as uint8 and grouping - this isn't robust cross-platform.
    # For safety in the pipeline, better to store arrays as comma-separated ints. For this starter, we will assume small n and reparse by splitting.
    # Here we'll fallback: attacker won't use honest_test on CSVs generated elsewhere in different format.
    return None

# For simplicity we'll re-run generation and test in one script when needed.

if __name__ == '__main__':
    print("honest_test.py is a lightweight helper. For robust runs, use run_grid_experiment.py which integrates generation and testing.")
