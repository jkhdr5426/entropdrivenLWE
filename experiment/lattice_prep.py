"""
Toy linearization exporter for quadratic EO-LWE.
This constructs the coefficient matrix of the linear system:
For each sample: b_i = x_i + alpha_i x_i^2 + e_i
Let x_i = a_i . s
Quadratic term becomes sum_{j,k} alpha_i * a_{i,j} a_{i,k} * s_j s_k
Introduce unknowns u = [s_1..s_n, t_{11}..t_{nn}] and write linear equations:
This script writes a CSV-like matrix of integer coefficients (mod q) that can be adapted to fplll.
"""
import numpy as np
import csv
import json
import argparse
from itertools import combinations_with_replacement

def build_linearized_system(samples, n, q):
    # samples: list of (a_vector, b, alpha, e, m)
    # unknowns: s_j (n), t_jk for j<=k (n*(n+1)/2)
    monoms = list(combinations_with_replacement(range(n),2))
    U = len(monoms)
    N = n + U
    rows = []
    rhs = []
    for (a,b,alpha,e,m) in samples:
        # compute linear coefficient for s_j from x term: coef_j = a_j
        coef_s = [int(a_j % q) for a_j in a]
        # compute coefficients for t_jk from quadratic term: alpha * a_j * a_k
        coef_t = []
        for (j,k) in monoms:
            coef_t.append(int((alpha * (int(a[j]) * int(a[k]) % q)) % q))
        # combined row: [coef_s | coef_t]
        row = coef_s + coef_t
        rows.append(row)
        # rhs: b - encode(m) - e  (but we move everything to rhs)
        rhs_val = int((b - e - (m * (q//2))) % q)
        rhs.append(rhs_val)
    return np.array(rows, dtype=int), np.array(rhs, dtype=int), monoms

def write_matrix_npz(rows, rhs, out_prefix):
    np.savez(out_prefix + '.npz', A=rows, b=rhs)
    print("Wrote", out_prefix + '.npz')

if __name__ == '__main__':
    print("This script is a helper; call from run_grid_experiment.py which uses in-memory samples.")
