# coding: utf-8

# Imports
from __future__ import division
import matplotlib.pyplot as plt

# Control variables
temperature1 = 0.1
temperature2 = 10.0
delta = 0.01
nAgents = 1000

# Read data from files
bin_centers = []
densities = []
densities1 = []
densities2 = []
with open("./Results/profitRateDist-delta{:.4f}-T{:.4f}-T{:.4f}.csv".format(delta, temperature1, temperature2),
          "r") as f:
    line = f.next()
    for line in f:
        bin_centers.append((float(line.split(",")[0]) + float(line.split(",")[1])) / 2.0)
        densities.append(float(line.split(",")[2]))
        densities1.append(float(line.split(",")[3]))
        densities2.append(float(line.split(",")[4]))

# Create figure and plot
plt.figure(figsize=(6, 4.5), facecolor='white')
plt.plot([x for x, a in zip(bin_centers, densities) if a > 0.0], [x for x in densities if x > 0.0], "o-",
         label="Aggregate")
plt.plot([x for x, a in zip(bin_centers, densities1) if a > 0.0], [x for x in densities1 if x > 0.0], "o-",
         label="Type 1 agents")
plt.plot([x for x, a in zip(bin_centers, densities2) if a > 0.0], [x for x in densities2 if x > 0.0], "o-",
         label="Type 2 agents")
plt.title(r"$\delta$ = {:.4f}, T1 = {:.4f}, T2 = {:.4f}".format(delta, temperature1, temperature2))
plt.xlim(-delta * 100, delta * 100)
plt.ylabel("Probability density")
plt.xlabel("Rate of profit")
plt.yscale("log")
plt.legend()

# Show and save
plt.tight_layout()
plt.show()
# plt.savefig("./RateDistribution-delta{:.4f}-T{:.4f}-T{:.4f}.pdf".format(delta, temperature1, temperature2),
#             format="pdf", dpi=300, bbox_inches='tight')
