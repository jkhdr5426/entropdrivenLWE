from sympy import symbols, expand, Poly, groebner
from math import comb
import time

# ======================================================
# Symbols
# ======================================================
x = symbols('x')
alpha, beta, a, b = symbols('alpha beta a b', integer=True)

# ======================================================
# Symbolic mirrors of transforms.py
# (NO mod q on purpose — structural analysis only)
# ======================================================

T_identity = x

T_quadratic = x + alpha * x**2

T_cubic = x**3 + alpha

T_eo_lwe = x**3 + alpha + beta

T_matrix_lift = a * x + b


# ======================================================
# Utilities
# ======================================================

def poly_stats(expr):
    p = Poly(expand(expr), x)
    return {
        "degree": p.degree(),
        "monomials": len(p.monoms())
    }


def linearization_dimension(n, d):
    """
    Number of monomials up to degree d in n variables
    (standard linearization bound)
    """
    return sum(comb(n + i - 1, i) for i in range(1, d + 1))


# ======================================================
# Experiment S1: Structural complexity
# ======================================================

def experiment_S1():
    print("\n=== S1: Polynomial Structure ===")
    transforms = {
        "Identity": T_identity,
        "Quadratic": T_quadratic,
        "Cubic": T_cubic,
        "EO-LWE": T_eo_lwe,
        "Matrix-lift": T_matrix_lift
    }

    for name, T in transforms.items():
        stats = poly_stats(T)
        print(f"{name}: degree={stats['degree']}, monomials={stats['monomials']}")


# ======================================================
# Experiment S2: Gröbner elimination feasibility
# ======================================================

def experiment_S2():
    print("\n=== S2: Gröbner Elimination ===")
    y = symbols('y')

    systems = {
        "Identity": y - T_identity,
        "Quadratic": y - T_quadratic,
        "Cubic": y - T_cubic,
        "EO-LWE": y - T_eo_lwe
    }

    for name, eq in systems.items():
        print(f"\n{name}:")
        try:
            t0 = time.time()
            # groebner expects a sequence of polynomials; wrap eq in a list
            G = groebner([eq], x, alpha, beta, y, order='lex')
            # use G.polys for a reliable basis list
            basis = G.polys
            print("  basis size:", len(basis))
            print("  basis polys:", basis)
            print("  time (s):", round(time.time() - t0, 4))
        except Exception as e:
            print("  FAILED:", e)


# ======================================================
# Experiment S3: Linearization growth
# ======================================================

def experiment_S3():
    print("\n=== S3: Linearization Dimension ===")
    for n in [8, 16, 32]:
        print(
            f"n={n}: "
            f"quadratic={linearization_dimension(n, 2)}, "
            f"cubic={linearization_dimension(n, 3)}"
        )


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":
    experiment_S1()
    experiment_S2()
    experiment_S3()