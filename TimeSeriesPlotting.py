# coding: utf-8

# Imports
from __future__ import division
import matplotlib.pyplot as plt


# Control variables
temperature1 = 0.1
temperature2 = 10.0
fraction_of_agents1 = 0.5
delta = 0.01
n_agents = 1000
final_time = 10000

# Reading model results from files
rate = []
with open("./Results/profitRateC-delta{:.4f}-T{:.4f}-T{:.4f}.csv".format(delta, temperature1, temperature2), "r") as f:
    for line in f:
        rate.append(float(line))
n_agents_up1 = []
n_agents_up2 = []
with open("./Results/nAgentsUpC-delta{:.4f}-T{:.4f}-T{:.4f}.csv".format(delta, temperature1, temperature2), "r") as f:
    for line in f:
        n_agents_up1.append(float(line.split(",")[0]))
        n_agents_up2.append(float(line.split(",")[1]))

# Create figure
plt.figure(figsize=(6, 4.5), facecolor='white')
# Plot number of agents up
plt.plot(range(final_time + 1), [x / (n_agents * fraction_of_agents1) for x in n_agents_up1], "-b",
         label="n_agents_up1")
plt.plot(range(final_time + 1), [x / (n_agents * (1 - fraction_of_agents1)) for x in n_agents_up2], "-g",
         label="n_agents_up2")
plt.xlim(0.0, final_time)
plt.ylim(0.0, 1.0)
plt.ylabel("Frequency of Action (for each type of agent)")
plt.xlabel("Time")
plt.legend()
plt.title(r"$\delta$ = {:.4f}, T1 = {:.4f}, T2 = {:.4f}".format(delta, temperature1, temperature2))
# Plot rate of profit
ax1 = plt.gca()
ax1.twinx()  # instantiate a second axes that shares the same x-axis
plt.plot(range(final_time + 1), rate, "-r", label="rate")
plt.axhline(y=0, ls="--", c="k", lw=2)
plt.xlim(0.0, final_time)
plt.ylim(-delta * 100, delta * 100)
plt.ylabel("Rate of profit")

# Show and save
plt.tight_layout()
plt.show()
# plt.savefig("./RateTimeSeriesC-delta{:.4f}-T{:.4f}-T{:.4f}.pdf".format(delta, temperature1, temperature2),
#             format="pdf", dpi=300, bbox_inches='tight')
