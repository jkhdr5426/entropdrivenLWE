import hashlib
import numpy as np

def derive_alpha(seed_bytes: bytes, counter: int, alpha_range: int=8) -> int:
    h = hashlib.sha256(seed_bytes + counter.to_bytes(8,'little')).digest()
    v = int.from_bytes(h, 'big')
    return v % alpha_range

# All T_* functions operate on a scalar x (mod q) and return integer in 0..q-1.

def T_identity(x, omega, q):
    return x % q

def T_quadratic(x, omega, q):
    # omega expected to contain 'alpha' integer
    alpha = omega.get('alpha', 0)
    return (x + alpha * (x * x % q)) % q

def T_cubic(x, omega, q):
    return (x**3 + omega['alpha']) % q

def T_eo_lwe(x, omega, q):
    """
    EO-LWE mapping: computes T_omega(x) given secret x and auxiliary omega.
    Replace this placeholder with your actual EO-LWE function.
    """
    # Example: simple cubic with omega mix (replace with real EO-LWE logic)
    return (x**3 + omega['alpha'] + omega.get('beta',0)) % q

def T_cyclotomic_toy(x, omega, q, m=8):
    """
    Toy cyclotomic-like transform: simulate polynomial mixing by rotating
    bits of x and adding small multiple controlled by omega.
    This is intentionally simple: real cyclotomic transforms require ring arithmetic.
    """
    alpha = omega.get('alpha', 1)
    # rotate bits of x (toy)
    width = max(8, int(np.ceil(np.log2(q))))
    rot = (x << 1) & ((1<<width)-1)
    return (x + alpha * rot) % q

def T_matrix_lift(x, omega, q):
    """
    Toy matrix-lift: map scalar x -> 2x2 matrix [x r; 0 x], multiply by small matrix and flatten
    We'll just return (a*x + b) mod q for toy purposes where a,b come from omega.
    """
    a = omega.get('a', 1)
    b = omega.get('b', 0)
    return (a * x + b) % q
