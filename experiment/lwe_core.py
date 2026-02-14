import numpy as np
import math

def int_centered(x, q):
    x = int(x) % q
    if x > q//2:
        x -= q
    return x

def sample_secret(n, q, small=True):
    """Sample secret vector s. small=True gives entries in {-1,0,1}."""
    if small:
        return np.random.choice([-1,0,1], size=n)
    else:
        return np.random.randint(0, q, size=n)

def sample_error(sigma):
    """Discrete Gaussian approx (rounded normal)."""
    return int(round(np.random.normal(0, sigma)))

def encode_message_bit(m, q):
    return (m * (q//2)) % q

def threshold_decode(value, q):
    # value assumed centered in (-q/2, q/2]
    if value > q//4:
        return 1
    else:
        return 0

def generate_lwe_sample(s, n, q, sigma, m_bit):
    a = np.random.randint(0, q, size=n)
    x = int(np.dot(a, s) % q)
    e = sample_error(sigma)
    b = (x + e + encode_message_bit(m_bit,q)) % q
    return a, b, e

def decrypt_lwe(a, b, s, q):
    x = int(np.dot(a, s) % q)
    diff = int_centered(b - x, q)
    return threshold_decode(diff, q)
