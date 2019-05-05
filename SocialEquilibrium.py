# coding: utf-8

# Imports
from __future__ import division
import numpy as np
import random as rand
import matplotlib.pyplot as plt


def main():
    # Control variables
    temperatures = np.linspace(0.2, 0.3, 41, endpoint=True)  # List of temperatures to simulate
    mu = 0.5
    n_agents = 1000
    final_time = 10000
    n_realizations = 50  # Realizations per temperature value
    initial_frequency = 0.5
    random_numbers_seed = 1
    control_write_results = True
    control_plot_results = False

    # Confirm large number of plots with user
    check_control_variables(control_write_results, control_plot_results, len(temperatures) * n_realizations)

    # Iterate over temperatures and realizations
    i = 0
    for temperature in temperatures:
        ts_collector = []
        for realization in range(n_realizations):

            # Set seed for random number generator for this realization and temperature
            rand.seed(random_numbers_seed * i)

            # Run model
            ts_n_agents_up = social_interaction_model(temperature, mu, n_agents, final_time, initial_frequency)

            # If printing results to file, collect time series for a given temperature
            if control_write_results:
                ts_collector.append(ts_n_agents_up)

            # Plot results
            if control_plot_results:
                plot_results(final_time, n_agents, ts_n_agents_up, temperature)

            i += 1

        # Print results to file
        if control_write_results:
            write_results(temperature, ts_collector, "nAgentsUp")

    # So that plots are shown
    if control_plot_results:
        plt.show()


def social_interaction_model(temperature, mu, n_agents, final_time, initial_frequency):
    # Compute initial state of the system
    state = []
    n_agents_up = 0
    ts_n_agents_up = []
    for i in range(n_agents):
        if rand.random() <= initial_frequency:
            state.append(1)
            n_agents_up += 1
        else:
            state.append(0)
    # Store initial state in time series
    ts_n_agents_up.append(n_agents_up)

    # Start simulation
    t = 1
    while t <= final_time:
        # Update the frequency of buying for a given agent (for now, all agents have the same frequency)
        frequency = 1/(1 + np.exp((mu - n_agents_up / n_agents) / temperature))
        n_agents_up = 0
        for i, s in enumerate(state):
            # Synchronous update: all agents update their state at the same time, thus not being aware of the changes
            # of the other agents till next time step. TODO: Confirm this point with Jangho.
            if rand.random() <= frequency:
                state[i] = 1
                n_agents_up += 1
            else:
                state[i] = 0
        # Store current state in time series
        ts_n_agents_up.append(n_agents_up)
        t += 1
    return ts_n_agents_up


def write_results(temperature, time_series_collector, file_name):
    """Prints results to file"""
    with open("./Results/" + file_name + "-T{:.4f}.csv".format(temperature), 'w') as f:
        for i, line in enumerate(zip(*time_series_collector)):
            if i < len(time_series_collector[1]) - 1:
                f.write("%s\n" % ", ".join([str(element) for element in line]))
            else:
                f.write("%s" % ", ".join([str(element) for element in line]))


def plot_results(final_time, n_agents, time_series, temperature):
    """Performs basic plotting"""
    plt.figure(figsize=(8, 6), facecolor='white')
    plt.plot(range(final_time + 1), time_series, "o-")
    plt.xlim(0.0, final_time)
    plt.ylim(0.0, n_agents)
    plt.ylabel("nAgentsUp")
    plt.xlabel("Time")
    plt.title("T = " + str(temperature))
    plt.tight_layout()
    plt.draw()


def check_control_variables(control_write_results, control_plot_results, number_of_plots):
    """Checks that control parameter values make sense"""
    if not control_write_results and not control_plot_results:
        print("Neither writing results nor plotting them!\n"
              "Aborting simulation.")
        exit()
    if control_plot_results and number_of_plots > 10:
        reply = raw_input("Are you sure you want to generate {} plots?\n"
                          "To confirm, type \"Y\": ".format(number_of_plots))
        if reply != "Y":
            print("Aborting simulation.")
            exit()
        else:
            print("Continuing simulation.")


if __name__ == "__main__":
    main()
