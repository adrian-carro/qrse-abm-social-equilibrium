# coding: utf-8

# Imports
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np


# Control variables
temperature = 0.2425
nAgents = 1000

# Reading model results from file
with open("./Results/nAgentsUp-T{:.4f}.csv".format(temperature), 'r') as f:
    line = f.next()
    fractionAgentsUp = [[] for i in range(len(line.split(",")))]
    for column, element in zip(fractionAgentsUp, line.split(",")):
        column.append(float(element))
    for line in f:
        for column, element in zip(fractionAgentsUp, line.split(",")):
            column.append(float(element))
# Rearranging different columns (realizations) as a single one and defining fraction rather than absolute number
fractionAgentsUp = [item / nAgents for column in fractionAgentsUp for item in column]

# Creating bins for the histogram
myBins = np.linspace(0.0, 1.0, 101, endpoint=True)

# Plot
fig = plt.figure(figsize=(8, 6), facecolor='white')
plt.hist(fractionAgentsUp, bins=myBins, density=True)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 20.0)
plt.ylabel("Prob. Density")
plt.xlabel("nAgentsUp")
plt.title("T = {:.4f}".format(temperature))
# plt.legend()
plt.tight_layout()
# plt.savefig("./nAgentsUp-T{:.4f}.eps".format(temperature), format='eps', dpi=300, bbox_inches='tight')

# Show
plt.show()
