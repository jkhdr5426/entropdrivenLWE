import sympy as sp

n = 4
x = sp.symbols(f'x0:{n}')
alpha = sp.symbols('alpha')

# EO-LWE-style multivariate toy
T = sum(x[i] for i in range(n))
T += alpha * sum(x[i]*x[j] for i in range(n) for j in range(i, n))

# Count monomials after expansion
expanded = sp.expand(T)
monomials = expanded.as_ordered_terms()

print("Expanded T(x):")
print(expanded)
print("\nNumber of monomials:", len(monomials))
