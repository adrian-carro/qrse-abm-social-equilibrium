# coding: utf-8

# Imports
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import os


# Control variables
temperatures = np.linspace(0.2, 0.3, 41, endpoint=True)  # List of temperatures to simulate
nAgents = 1000

# Create temporary folder for video snapshots
tempDirectory = "./tempVideoFiles"
if not os.path.exists(tempDirectory):
    os.makedirs(tempDirectory)
# else:
#     print("Folder \"" + tempDirectory + "\" already exists. Please, delete it before proceeding further.")
#     exit()

# Create bins for the histograms
myBins = np.linspace(0.0, 1.0, 101, endpoint=True)

# Open general figure
fig = plt.figure(figsize=(8, 6), facecolor="white")

# Read model results from file, plot, save figure and clear it
imageNumber = 1
for temperature in temperatures:
    with open("./Results/nAgentsUp-T{:.4f}.csv".format(temperature), 'r') as f:
        line = f.next()
        fractionAgentsUp = [[] for i in range(len(line.split(",")))]
        for column, element in zip(fractionAgentsUp, line.split(",")):
            column.append(float(element))
        for line in f:
            for column, element in zip(fractionAgentsUp, line.split(",")):
                column.append(float(element))
    # Rearrange different columns (realizations) as a single one and define fraction rather than absolute number
    fractionAgentsUp = [item / nAgents for column in fractionAgentsUp for item in column]
    # Plot
    plt.hist(fractionAgentsUp, bins=myBins, density=True)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 20.0)
    plt.ylabel("Prob. Density")
    plt.xlabel("nAgentsUp")
    plt.title("T = {:.4f}".format(temperature))
    # plt.tight_layout()
    plt.savefig(tempDirectory + "/im{:04d}.png".format(imageNumber), format="png", dpi=300, bbox_inches='tight')
    # Clear axis
    plt.cla()
    print("Done image {:d} out of {:d}".format(imageNumber, len(temperatures)))
    imageNumber += 1

# Call ffmpeg via command line for creating a video with the temporary image files, print to screen any output
os.system("ffmpeg -r 2 -i " + tempDirectory +
          "/im%04d.png -c:v libx264 -x264opts b_pyramid=0 -b 800k -vf fps=2 ./out.mp4")

# Remove temporary image files
os.system("rm -rf " + tempDirectory)
