import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np

# Load CSV data
df = pd.read_csv("love_numbers.csv")  # Replace with your actual CSV file name

# Get all unique indexes
unique_indexes = sorted(df["index"].unique())

# Set up colormap (excluding 1.0 since it's plotted separately)
other_indexes = [i for i in unique_indexes if i != 1.0]
norm = colors.Normalize(vmin=min(other_indexes), vmax=max(other_indexes))
cmap = cm.coolwarm

# Create figure and axis
fig, ax = plt.subplots(constrained_layout=True)

# Labels
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$k_2$")
ax.set_title("Tidal Love Number vs Compactness for Various Polytropic Indexes")
ax.set_xlim([0, 0.35])
ax.set_ylim([0, 0.4])

# Create figure and axis
fig, ax2 = plt.subplots(constrained_layout=True)

# Labels
ax2.set_xlabel(r"$\beta$")
ax2.set_ylabel(r"$log_{10}(\lambda)$")
ax2.set_title("Log of Tidal Polarizability vs Compactness for Various Polytropic Indexes")
ax2.set_xlim([0, 0.35])
ax2.set_ylim([2, 6])

# Plotting
for n in unique_indexes:
    data = df[df["index"] == n]
    polarizability = np.log10((2 / 3) * (data['k2'] * (data['radius']) ** 5))
    if n == 1.0:
        ax.plot(data["compactness"], data["k2"], color="black", label="n = 1.0", linewidth=1.5)
        ax2.plot(data["compactness"], polarizability, color="black", label="n = 1.5", linewidth=1.5)
    else:
        color = cmap(norm(n))
        ax.plot(data["compactness"], data["k2"], color=color, label=f"n = {n}")
        ax2.plot(data["compactness"], polarizability, color=color, label=f"n = {n}")

# Show plot
plt.show()

# Create figure and axis
fig, ax3 = plt.subplots(constrained_layout=True)

# Labels
ax3.set_xlabel(r"$Radius$")
ax3.set_ylabel(r"$Mass$")
ax3.set_title("Log of Tidal Polarizability vs Compactness for Various Polytropic Indexes")

# Plotting
for n in unique_indexes:
    data = df[df["index"] == n]
    if n == 1.0:
        ax3.plot(data['radius'], data['mass'], color="black", label="n = 1.5", linewidth=1.5)
    else:
        color = cmap(norm(n))
        ax3.plot(data['radius'], data['mass'], color=color, label=f"n = {n}")

# Show plot
plt.show()