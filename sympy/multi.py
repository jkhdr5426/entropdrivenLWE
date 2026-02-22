import csv
from sympy import symbols, expand, Poly, groebner
from math import comb
from itertools import combinations
import time

# ======================================================
# Parameters
# ======================================================
n_list = [2, 4, 8, 16, 64, 128]  # lattice dimensions
grb_limit = 128  # only attempt Gröbner basis for small n

# ======================================================
# Utilities
# ======================================================
def poly_stats(expr, vars):
    p = Poly(expand(expr), *vars)
    return {
        "degree": p.total_degree(),
        "monomials": len(p.monoms())
    }

def linearization_dimension(n, d):
    return sum(comb(n + i - 1, i) for i in range(1, d + 1))

# ======================================================
# Transform Generators
# ======================================================
def T_quadratic(vars, alpha_list):
    return sum(x + alpha*x**2 for x, alpha in zip(vars, alpha_list))

def T_cubic(vars, alpha_list):
    return sum(x**3 + alpha for x, alpha in zip(vars, alpha_list))

def T_eo_lwe(vars, alpha_list, beta_list, cross_degree=2):
    expr = sum(x**3 + alpha + beta for x, alpha, beta in zip(vars, alpha_list, beta_list))
    for deg in range(2, cross_degree+1):
        for combo in combinations(vars, deg):
            expr += sum(xi*combo[0] for xi in combo[1:])
    return expr

# ======================================================
# Prepare CSV
# ======================================================
csv_file = "transform_analysis.csv"
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n", "Transform", "Degree", "Monomials", "Linearization_dim", "Grb_time_s"])

    # ======================================================
    # Experiment
    # ======================================================
    for n in n_list:
        vars = symbols(f"x1:{n+1}")
        alpha_list = symbols(f"alpha1:{n+1}")
        beta_list = symbols(f"beta1:{n+1}")
        
        transforms = {
            "Quadratic": T_quadratic(vars, alpha_list),
            "Cubic": T_cubic(vars, alpha_list),
            "EO-LWE": T_eo_lwe(vars, alpha_list, beta_list, cross_degree=2)
        }
        
        for name, T in transforms.items():
            stats = poly_stats(T, vars)
            lin_dim = linearization_dimension(n, 3 if "cubic" in name.lower() else 2)
            
            grb_time = None
            if n <= grb_limit:
                y = symbols('y')
                eq = y - T
                try:
                    t0 = time.time()
                    G = groebner([eq], *vars, *alpha_list, *(beta_list if "eo-lwe" in name.lower() else []), y, order='lex')
                    grb_time = round(time.time() - t0, 4)
                except Exception as e:
                    grb_time = None
            # Write row to CSV
            writer.writerow([n, name, stats["degree"], stats["monomials"], lin_dim, grb_time])
            # Print to console
            print(f"n={n}, {name}: degree={stats['degree']}, monomials={stats['monomials']}, "
                  f"linearization_dim={lin_dim}, grb_time_s={grb_time}")