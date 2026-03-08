import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"

data = np.loadtxt("22112.dat", skiprows=0) 
x, y = data[:, 0], data[:, 1]

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(x, y, color="black", linewidth=1.5)
ax.fill(x, y, color="lightgrey", alpha=0.6)


split = np.argmin(x)

x_upper = x[:split+1]
y_upper = y[:split+1]
x_lower = x[split:]
y_lower = y[split:]

# Upper surface goes from x=1 down to x=0, so flip it
x_upper_flip = x_upper[::-1]
y_upper_flip = y_upper[::-1]

# Interpolate upper onto lower surface x points
y_upper_interp = np.interp(x_lower, x_upper_flip, y_upper_flip)
y_camber = (y_lower + y_upper_interp) / 2

print("split index:", split)
print("x_upper:", x_upper[:5])
print("x_lower:", x_lower[:5])
print("y_camber:", y_camber[:5])
ax.plot(x_lower, y_camber, color="black", linewidth=1.3, linestyle="--", label="Mean camber line")
ax.legend(fontsize=35, loc="lower center", bbox_to_anchor=(0.5, -1.2))

# Axis formatting
ax.axhline(0, color="black", linewidth=0.8, linestyle="dotted", alpha=0.4)
ax.set_xlabel("x/c", fontsize=25)
ax.set_ylabel("y/c", fontsize=25)
ax.set_aspect("equal")
ax.grid(True, linestyle="dotted", alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.tight_layout()
plt.savefig("airfoil.png", dpi=1200, bbox_inches="tight")
plt.show()
