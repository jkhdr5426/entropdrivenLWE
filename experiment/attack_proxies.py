import numpy as np
from numpy.linalg import lstsq
import math

def naive_linear_attack(samples):
    """
    samples: list of (a_vector, b_scalar)
    Returns real-valued s_hat from least squares (A*s = b)
    """
    A = np.array([s[0] for s in samples], dtype=float)
    b = np.array([s[1] for s in samples], dtype=float)
    # Solve least squares
    try:
        s_hat, *_ = lstsq(A, b, rcond=None)
    except Exception:
        s_hat = np.linalg.pinv(A).dot(b)
    return s_hat

def attacker_success_with_shat(samples, s_hat, q):
    correct = 0
    for a, b, m in samples:
        x_hat = int(round(np.dot(a, s_hat))) % q
        diff = int((b - x_hat + q) % q)
        if diff > q//2: diff -= q
        # decode same thresholds as encode
        mhat = 1 if diff > q//4 else 0
        if mhat == m:
            correct += 1
    return correct / len(samples)

def linearization_dimension_quadratic(n):
    # unknowns: s (n) + symmetric pairwise products t_{i<=j} -> n(n+1)/2
    return n + (n*(n+1))//2

def simple_bkw_proxy(samples, q, bucket_bits=4):
    """
    A very simple sample-combining heuristic (toy BKW-like).
    Not a full BKW implementation — just a proxy to see if many samples help.
    """
    # group samples by low bits of a[0]
    buckets = {}
    for a,b,m in samples:
        key = int(a[0]) & ((1<<bucket_bits)-1)
        buckets.setdefault(key, []).append((a,b,m))
    # For each bucket, average b values and try naive decode using most frequent parity
    votes = []
    for k,v in buckets.items():
        # try simplistic guess: average b mod q
        avg = sum([int(b) for _,b,_ in v]) / max(1,len(v))
        # decode
        val = int(round(avg)) % q
        if val > q//2: val -= q
        mhat = 1 if val > q//4 else 0
        votes.append(mhat)
    # return fraction of votes equal to majority (just a heuristic signal)
    if len(votes)==0:
        return 0.0
    from collections import Counter
    c = Counter(votes)
    maj = c.most_common(1)[0][0]
    score = sum(1 for v in votes if v==maj) / len(votes)
    return score
