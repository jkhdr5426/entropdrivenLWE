import argparse
import numpy as np
import csv
import hashlib
from lwe_core import sample_secret, generate_lwe_sample
from transforms import derive_alpha, T_quadratic, T_identity, T_cyclotomic_toy, T_matrix_lift

def generate_samples(transform_name, n, q, sigma, trials, alpha_range, out_csv):
    rng_seed = hashlib.sha256(b'EO-LWE-experiment-seed').digest()
    s = sample_secret(n,q,small=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trial','transform','n','q','sigma','a_hex','b','alpha','e','m','seed_hex'])
        for t in range(trials):
            m = int(np.random.randint(0,2))
            a = np.random.randint(0, q, size=n)
            x = int(np.dot(a, s) % q)
            alpha = derive_alpha(rng_seed, t, alpha_range)
            omega = {'alpha': alpha, 'a': alpha+1, 'b': alpha}
            if transform_name == 'identity':
                T = T_identity(x, omega, q)
            elif transform_name == 'quadratic':
                T = T_quadratic(x, omega, q)
            elif transform_name == 'cyclotomic_toy':
                T = T_cyclotomic_toy(x, omega, q)
            elif transform_name == 'matrix_lift':
                T = T_matrix_lift(x, omega, q)
            else:
                raise ValueError("Unknown transform")
            e = int(round(np.random.normal(0, sigma)))
            b = (T + e + (m * (q//2))) % q
            writer.writerow([t, transform_name, n, q, sigma, a.tobytes().hex(), b, alpha, e, m, rng_seed.hex()])
    print(f"Samples written to {out_csv}")
    return s

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transform', default='quadratic', choices=['identity','quadratic','cyclotomic_toy','matrix_lift'])
    parser.add_argument('--n', type=int, default=32)
    parser.add_argument('--q', type=int, default=3329)
    parser.add_argument('--sigma', type=float, default=2.0)
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--alpha_range', type=int, default=8)
    parser.add_argument('--out', default='samples.csv')
    args = parser.parse_args()
    generate_samples(args.transform, args.n, args.q, args.sigma, args.trials, args.alpha_range, args.out)
