import sympy as sp

x, s, alpha, b, e = sp.symbols('x s alpha b e')

# EO-LWE toy equation
T = s*x + alpha*(s*x)**2
eq = sp.Eq(T + e, b)

# Try eliminating s
elim = sp.solve(eq, s, dict=True)

print("Solutions for s:")
print(elim)
