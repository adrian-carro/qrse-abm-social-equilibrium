# coding: utf-8

# Imports
from __future__ import division
import matplotlib.pyplot as plt


# Control variables
temperature = 0.24
nAgents = 1000

# Reading model results from file
with open("./Results/nAgentsUp-T{:.4f}.csv".format(temperature), 'r') as f:
    line = f.next()
    nAgentsUp = [[] for i in range(len(line.split(",")))]
    for column, element in zip(nAgentsUp, line.split(",")):
        column.append(float(element))
    for line in f:
        for column, element in zip(nAgentsUp, line.split(",")):
            column.append(float(element))

# Plot
fig = plt.figure(figsize=(8, 6), facecolor='white')
plt.plot(range(len(nAgentsUp[0])), nAgentsUp[8], "o-", label="nAgentsUp")
plt.xlim(0.0, len(nAgentsUp[0]))
plt.ylim(0.0, nAgents)
plt.ylabel("nAgentsUp")
plt.xlabel("Time")
plt.title("T = " + str(temperature))
# plt.legend()
plt.tight_layout()
# plt.savefig("./nAgentsUp-T{:.4f}.eps".format(temperature), format='eps', dpi=300, bbox_inches='tight')

# Show
plt.show()
