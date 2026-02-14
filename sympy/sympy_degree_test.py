import sympy as sp

def quadratic_T(x, alpha):
    return x + alpha*x**2

def cubic_T(x, alpha):
    return x + alpha*x**3

def eo_lwe_T(x, alpha):
    return x + alpha*x**2 + alpha**2*x**3

x, alpha = sp.symbols('x alpha')

transforms = {
    "quadratic": quadratic_T(x, alpha),
    "cubic": cubic_T(x, alpha),
    "eo_lwe": eo_lwe_T(x, alpha)
}

print("Algebraic degree of T(x):")
for name, expr in transforms.items():
    deg = sp.total_degree(expr)
    print(f"{name}: degree {deg}")
