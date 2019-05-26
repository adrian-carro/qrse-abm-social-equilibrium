# coding: utf-8

# Imports
from __future__ import division
import matplotlib.pyplot as plt

# Control variables
temperatures = [0.01, 0.1, 1.0, 10.0, 100.0]  # List of temperatures to plot
deltas = [0.01]  # List of delta values to plot
# temperatures = [1.0]  # List of temperatures to plot
# deltas = [0.01, 0.1, 1.0, 10.0]  # List of delta values to plot
nAgents = 1000

# Read data from files
bin_centers = []
densities = []
# For each value of delta...
for delta in deltas:
    # ...and for each temperature...
    for temperature in temperatures:
        # ...read results...
        bin_centers.append([])
        densities.append([])
        with open("./Results/profitRateDist-delta{:.4f}-T{:.4f}-T{:.4f}.csv".format(delta, temperature, temperature),
                  "r") as f:
            line = f.next()
            for line in f:
                if float(line.split(",")[2]) > 0.0:
                    bin_centers[-1].append((float(line.split(",")[0]) + float(line.split(",")[1])) / 2.0)
                    densities[-1].append(float(line.split(",")[2]))

# Create figure and plot
plt.figure(figsize=(6, 4.5), facecolor='white')
if len(deltas) == 1:
    for temperature, bin_centers_line, densities_line in zip(temperatures, bin_centers, densities):
        plt.plot(bin_centers_line, densities_line, "o-", label="T = {:.4f}".format(temperature))
    plt.title(r"$\delta$ = {:.4f}".format(deltas[0]))
    plt.xlim(-deltas[0] * 100, deltas[0] * 100)
elif len(temperatures) == 1:
    for delta, bin_centers_line, densities_line in zip(deltas, bin_centers, densities):
        plt.plot(bin_centers_line, densities_line, "o-", label=r"$\delta$ = {:.4f}".format(delta))
    plt.title(r"T = {:.4f}".format(temperatures[0]))
    # plt.xlim(-max(deltas) * 100, max(deltas) * 100)
else:
    print("Either temperatures or deltas must have length 1")
    exit()

plt.ylabel("Probability density")
plt.xlabel("Rate of profit")
plt.yscale("log")
plt.legend()

# Show and save
plt.tight_layout()
plt.show()
# if len(deltas) == 1:
#     plt.savefig("./RateDistribution-delta{:.4f}.pdf".format(deltas[0]), format="pdf", dpi=300, bbox_inches='tight')
# elif len(temperatures) == 1:
#     plt.savefig("./RateDistribution-T{:.4f}.pdf".format(temperatures[0]), format="pdf", dpi=300, bbox_inches='tight')
