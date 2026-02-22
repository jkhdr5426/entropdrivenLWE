import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
# CSV columns: n, transform, degree, monomials, linearization_dim, grb_time_s
df = pd.read_csv("transform_analysis.csv")

# Set styles
colors = {"Quadratic": "blue", "Cubic": "green", "EO-LWE": "red"}

# --------------------------
# Plot Gröbner time vs n
# --------------------------
plt.figure(figsize=(8,5))
for t in df["Transform"].unique():
    subset = df[df["Transform"] == t]
    plt.plot(subset["n"], subset["Grb_time_s"], marker='o', color=colors[t], label=t)

plt.xlabel("Lattice dimension n")
plt.ylabel("Gröbner time (s)")
plt.title("Gröbner Basis Computation Time vs Lattice Dimension")
plt.yscale("log")  # use log scale for large differences
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------
# Plot Linearization dimension vs n
# --------------------------
plt.figure(figsize=(8,5))
for t in df["Transform"].unique():
    subset = df[df["Transform"] == t]
    plt.plot(subset["n"], subset["linearization_dim"], marker='o', color=colors[t], label=t)

plt.xlabel("Lattice dimension n")
plt.ylabel("Linearization dimension")
plt.title("Linearization Dimension vs Lattice Dimension")
plt.yscale("log")  # log scale makes large growth easier to visualize
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------
# Optional: Monomials vs n
# --------------------------
plt.figure(figsize=(8,5))
for t in df["transform"].unique():
    subset = df[df["transform"] == t]
    plt.plot(subset["n"], subset["monomials"], marker='o', color=colors[t], label=t)

plt.xlabel("Lattice dimension n")
plt.ylabel("Number of monomials")
plt.title("Monomial Growth vs Lattice Dimension")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()