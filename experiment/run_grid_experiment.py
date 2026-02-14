import argparse
import time
import numpy as np
import csv
import os
from lwe_core import sample_secret, sample_error, encode_message_bit, int_centered
from transforms import derive_alpha, T_quadratic, T_identity, T_cyclotomic_toy, T_matrix_lift, T_cubic, T_eo_lwe
from attack_proxies import naive_linear_attack, attacker_success_with_shat, linearization_dimension_quadratic, simple_bkw_proxy
from lattice_prep import build_linearized_system, write_matrix_npz

def run_cell(transform, n, q, sigma, trials, alpha_range, out_dir):
    seed_bytes = hashlib = __import__('hashlib').sha256(b'EO-LWE-experiment-seed').digest()
    s = sample_secret(n,q,small=True)
    os.makedirs(out_dir, exist_ok=True)
    samples = []
    honest_success = 0
    t0 = time.time()
    for t in range(trials):
        m = int(np.random.randint(0,2))
        a = np.random.randint(0, q, size=n)
        x = int(np.dot(a,s) % q)
        alpha = derive_alpha(seed_bytes, t, alpha_range)
        omega = {'alpha': alpha, 'a': alpha+1, 'b': alpha}
        if transform == 'identity':
            T = T_identity(x, omega, q)
        elif transform == 'quadratic':
            T = T_quadratic(x, omega, q)
        elif transform == 'cyclotomic_toy':
            T = T_cyclotomic_toy(x, omega, q)
        elif transform == 'matrix_lift':
            T = T_matrix_lift(x, omega, q)
        elif transform == 'cubic':
            T = T_cubic(x,omega,q)
        elif transform == 'eo_lwe':
            from transforms import T_eo_lwe
            T = T_eo_lwe(x, omega, q)

        else:
            raise ValueError("Unknown transform")
        e = sample_error(sigma)
        b = (T + e + (m * (q//2))) % q
        # authorized decryption
        # derive alpha again
        alpha_r = derive_alpha(seed_bytes, t, alpha_range)
        omega_r = {'alpha': alpha_r}
        if transform == 'identity':
            T_est = T_identity(x, omega_r, q)
        elif transform == 'quadratic':
            T_est = T_quadratic(x, omega_r, q)
        elif transform == 'cyclotomic_toy':
            T_est = T_cyclotomic_toy(x, omega_r, q)
        elif transform == 'matrix_lift':
            T_est = T_matrix_lift(x, omega_r, q)
        elif transform == 'cubic':
            T_est = T_cubic(x,omega_r,q)
        elif transform == 'eo_lwe':
            T_est = T_eo_lwe(x, omega_r, q)
        diff = int_centered(b - T_est, q)
        m_hat = 1 if diff > q//4 else 0
        if m_hat == m:
            honest_success += 1
        samples.append((a, int(b), m, int(alpha), int(e)))
    honest_dsr = honest_success / trials
    # Attack proxies: naive regression
    # Build samples for attacker: list of (a, b, m)
    # attacker sees only the first two items: a, b
    atk_samples = [(s[0], s[1]) for s in samples]

    # For regression we need arrays
    regression_input = [(a,b,m) for (a,b,m,alpha,e) in [(a,b,m,alpha,e) for (a,b,m,alpha,e) in samples]]
    # reuse code from attack_proxies: naive_linear_attack expects (a,b) pairs
    ab_pairs = [(smp[0], smp[1]) for smp in regression_input]
    s_hat = naive_linear_attack(ab_pairs)
    from attack_proxies import attacker_success_with_shat
    attacker_success = attacker_success_with_shat(regression_input, s_hat, q)
    bkw_score = simple_bkw_proxy(regression_input, q)
    lin_dim = n*4 if transform=='eo_lwe' else (linearization_dimension_quadratic(n) if transform == 'quadratic' else (n*3 if transform=='cubic' else n*2))
    elapsed = time.time() - t0
    # Save summary CSV
    summary_csv = os.path.join(out_dir, f"summary_{transform}_n{n}.csv")
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['transform','n','q','sigma','trials','honest_dsr','attacker_success_regression','bkw_score','linearization_dim','time_s'])
        writer.writerow([transform,n,q,sigma,trials,honest_dsr,attacker_success,bkw_score,lin_dim,elapsed])
    # Save samples CSV
    samples_csv = os.path.join(out_dir, f"samples_{transform}_n{n}.csv")
    with open(samples_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trial','a_hex','b','m','alpha','e'])
        for idx,(a,b,m,alpha,e) in enumerate(samples):
            writer.writerow([idx, a.tobytes().hex(), b, m, alpha, e])
    # If quadratic, prepare linearized lattice export (toy)
    if transform == 'quadratic':
        # build samples list in format expected by lattice_prep
        lp_samples = [(a,b,alpha,e,m) for (a,b,m,alpha,e) in [(row[0], row[1], row[3], row[4], row[2]) for row in samples]]
        Arows, rhs = build_linearized_system(lp_samples, n, q)[:2]
        write_matrix_npz(Arows, rhs, os.path.join(out_dir, f"lin_{transform}_n{n}"))
    print("Wrote results to", out_dir)
    return summary_csv, samples_csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transform', default='quadratic')
    parser.add_argument('--n', type=int, default=32)
    parser.add_argument('--q', type=int, default=3329)
    parser.add_argument('--sigma', type=float, default=2.0)
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--alpha_range', type=int, default=8)
    parser.add_argument('--out_dir', default='results')
    args = parser.parse_args()
    run_cell(args.transform, args.n, args.q, args.sigma, args.trials, args.alpha_range, args.out_dir)
