# coding: utf-8

# Imports
from __future__ import division
import numpy as np
import random as rand
import matplotlib.pyplot as plt


def main():
    # Control variables
    # temperatures1 = [0.01, 0.1, 1.0, 10.0, 100.0]  # List of temperatures to simulate for the first type of agent
    # temperatures2 = [0.01, 0.1, 1.0, 10.0, 100.0]  # List of temperatures to simulate for the second type of agent
    temperatures1 = [1.0]  # List of temperatures to simulate for the first type of agent
    temperatures2 = [1.0]  # List of temperatures to simulate for the second type of agent
    fraction_of_agent1 = 0.5
    mu = 0.0
    n_agents = 1000
    delta = 0.01
    final_time = 10000000
    # final_time = 10000
    initial_rate = 0.0
    random_numbers_seed = 1
    control_write_results = True
    control_plot_results = False

    # Confirm large number of plots with user
    check_control_variables(control_write_results, control_plot_results, len(temperatures1))

    # Iterate over temperatures and realizations
    i = 1
    for temperature1, temperature2 in zip(temperatures1, temperatures2):

        # Set seed for random number generator for this realization and temperature
        rand.seed(random_numbers_seed * i)

        # Run model
        ts_n_agents_up, ts_rate = social_interaction_model_asynchronous(temperature1, temperature2,
                                                                        fraction_of_agent1, mu, n_agents,
                                                                        final_time, initial_rate, delta)

        # Plot results
        if control_plot_results:
            plot_time_series(final_time, n_agents, ts_n_agents_up, ts_rate[0], delta, temperature1, temperature2,
                             fraction_of_agent1)
            # plot_distribution(ts_rate, delta, temperature1, temperature2)

        i += 1

        # Print results to file
        if control_write_results:
            # write_time_series(delta, temperature1, temperature2, ts_n_agents_up, "nAgentsUp")
            # write_time_series(delta, temperature1, temperature2, ts_rate, "profitRate")
            write_distribution(delta, temperature1, temperature2, ts_rate[0], ts_n_agents_up, n_agents,
                               fraction_of_agent1, "profitRateDist")

    # So that plots are shown
    if control_plot_results:
        plt.show()


def social_interaction_model(temperature, mu, n_agents, final_time, initial_rate, delta):
    state = [0] * n_agents
    rate = initial_rate
    ts_n_agents_up = []
    ts_rate = []
    t = 0
    old_n_agents_up = int(n_agents/2)
    while t <= final_time:
        # Update the frequency of buying for a given agent (for now, all agents have the same frequency)
        probability_of_entry = 1/(1 + np.exp((mu - rate) / temperature))
        n_agents_up = 0
        for i, s in enumerate(state):
            # Synchronous update: all agents update their state at the same time, thus not being aware of the changes
            # of the other agents till next time step. TODO: Confirm this point with Jangho.
            if rand.random() <= probability_of_entry:
                state[i] = 1
                n_agents_up += 1
            else:
                state[i] = 0
        # Store current state in time series
        ts_n_agents_up.append(n_agents_up)
        ts_rate.append(rate)
        # Update the rate of profit for next time step
        rate = rate - delta * (n_agents_up - old_n_agents_up)
        old_n_agents_up = n_agents_up
        t += 1
    return ts_n_agents_up, ts_rate


def social_interaction_model_asynchronous(temperature1, temperature2, fraction_of_agent1, mu, n_agents, final_time,
                                          initial_rate, delta):
    rate = initial_rate
    ts_n_agents_up = [[], []]
    ts_rate = [[]]
    # Compute initial state of the system (with exact expected number of agents in each state)...
    state = []
    kind = []
    n_agents_up = [0, 0]
    # ...first for agents of type 1
    initial_probability_of_entry = 1 / (1 + np.exp((mu - initial_rate) / temperature1))
    for i in range(int(n_agents * fraction_of_agent1 * initial_probability_of_entry)):
        state.append(1)
        kind.append(0)
        n_agents_up[0] += 1
    for i in range(int(n_agents * fraction_of_agent1 * initial_probability_of_entry),
                   int(n_agents * fraction_of_agent1)):
        state.append(0)
        kind.append(0)
    # ...then for agents of type 2
    initial_probability_of_entry = 1 / (1 + np.exp((mu - initial_rate) / temperature2))
    for i in range(int(n_agents * (1 - fraction_of_agent1) * initial_probability_of_entry)):
        state.append(1)
        kind.append(1)
        n_agents_up[1] += 1
    for i in range(int(n_agents * (1 - fraction_of_agent1) * initial_probability_of_entry),
                   int(n_agents * (1 - fraction_of_agent1))):
        state.append(0)
        kind.append(1)
    # Store initial state in time series
    ts_n_agents_up[0].append(n_agents_up[0])
    ts_n_agents_up[1].append(n_agents_up[1])
    ts_rate[0].append(rate)
    # Store initial state as old state and set most recent equilibrium to None
    old_n_agents_up = n_agents_up[0] + n_agents_up[1]

    # Start simulation
    t = 1
    while t <= final_time:
        # Update buying frequencies for agents of both types
        probability_of_entry = [1/(1 + np.exp((mu - rate) / temperature1)), 1/(1 + np.exp((mu - rate) / temperature2))]
        # Asynchronous update: only a randomly chosen agent updates its state in a given time step, thus the system is
        # instantaneously aware of every individual change of state
        i = rand.randint(0, n_agents - 1)
        if rand.random() <= probability_of_entry[kind[i]]:
            if state[i] != 1:
                state[i] = 1
                n_agents_up[kind[i]] += 1
        else:
            if state[i] != 0:
                state[i] = 0
                n_agents_up[kind[i]] -= 1
        # Store current state in time series
        ts_n_agents_up[0].append(n_agents_up[0])
        ts_n_agents_up[1].append(n_agents_up[1])
        ts_rate[0].append(rate)
        # Update the rate of profit for next time step
        rate = rate - delta * ((n_agents_up[0] + n_agents_up[1]) - old_n_agents_up)
        old_n_agents_up = n_agents_up[0] + n_agents_up[1]
        t += 1
    return ts_n_agents_up, ts_rate


def write_time_series(delta, temperature1, temperature2, time_series, file_name):
    """Prints results to file"""
    with open("./Results/" + file_name + "-delta{:.4f}-T{:.4f}-T{:.4f}.csv".format(delta, temperature1, temperature2),
              "w") as f:
        for i, line in enumerate(zip(*time_series)):
            if i < len(time_series[0]) - 1:
                f.write("%s\n" % ", ".join([str(element) for element in line]))
            else:
                f.write("%s" % ", ".join([str(element) for element in line]))


def write_distribution(delta, temperature1, temperature2, time_series_rate, time_series_n_agents_up, n_agents,
                       fraction_of_agent1, file_name):
    """Prints rate distribution results to file for the aggregate rate and the average rate for each type of agent"""
    # First create the aggregate, general histogram
    my_bins = np.linspace(-delta * 100, delta * 100, 200, endpoint=True)
    densities, bins = np.histogram(time_series_rate, bins=my_bins, density=True)
    # First create a time series of average rate for agents of type 1, and a histogram of this average
    time_series_rate1 = []
    for rate, n_agents_up1 in zip(time_series_rate, time_series_n_agents_up[0]):
        time_series_rate1.append(rate * n_agents_up1 / (n_agents * fraction_of_agent1))
    densities1, bins1 = np.histogram(time_series_rate1, bins=my_bins, density=True)
    time_series_rate2 = []
    for rate, n_agents_up2 in zip(time_series_rate, time_series_n_agents_up[1]):
        time_series_rate2.append(rate * n_agents_up2 / (n_agents * (1 - fraction_of_agent1)))
    densities2, bins2 = np.histogram(time_series_rate2, bins=my_bins, density=True)
    # Then print to file all three distributions, with the same bins
    with open("./Results/" + file_name + "-delta{:.4f}-T{:.4f}-T{:.4f}.csv".format(delta, temperature1, temperature2),
              "w") as f:
        f.write("Lower bin edge, upper bin edge, density, density1, density2\n")
        for lower_bin_edge, upper_bin_edge, density, density1, density2 in zip(my_bins[:-1], my_bins[1:], densities,
                                                                               densities1, densities2):
            f.write("{}, {}, {}, {}, {}\n".format(lower_bin_edge, upper_bin_edge, density, density1, density2))


def plot_time_series(final_time, n_agents, ts_n_agents_up, ts_rate, delta, temperature1, temperature2,
                     fraction_of_agents1):
    """Performs basic plotting of the time series of n_agents_up and the rate of profit"""
    # Create figure
    plt.figure(figsize=(6, 4.5), facecolor='white')
    # Plot number of agents up
    plt.plot(range(final_time + 1), [x / (n_agents * fraction_of_agents1) for x in ts_n_agents_up[0]], "-b",
             label="n_agents_up1")
    plt.plot(range(final_time + 1), [x / (n_agents * (1 - fraction_of_agents1)) for x in ts_n_agents_up[1]], "-g",
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
    plt.plot(range(final_time + 1), ts_rate, "-r", label="rate")
    plt.axhline(y=0, ls="--", c="k", lw=2)
    plt.xlim(0.0, final_time)
    plt.ylim(-delta * 100, delta * 100)
    plt.ylabel("Rate of profit")
    plt.tight_layout()
    plt.draw()
    # plt.savefig("./RateTimeSeries-delta{:.4f}-T{:.4f}-T{:.4f}.pdf".format(delta, temperature1, temperature2),
    #             format="pdf", dpi=300, bbox_inches='tight')


def plot_distribution(ts_rate, delta, temperature1, temperature2):
    """Performs basic plotting of the distribution of the rate of profit"""
    # Create figure
    plt.figure(figsize=(6, 4.5), facecolor='white')
    # Plot histogram of the rate of profit
    my_bins = np.linspace(-delta * 100, delta * 100, 200, endpoint=True)
    plt.hist(ts_rate, bins=my_bins, density=True)
    plt.xlim(-delta * 100, delta * 100)
    plt.ylabel("Probability density")
    plt.xlabel("Rate of profit")
    plt.title(r"$\delta$ = {:.4f}, T1 = {:.4f}, T2 = {:.4f}".format(delta, temperature1, temperature2))
    plt.tight_layout()
    plt.draw()
    # plt.savefig("./RateDistribution-delta{:.4f}-T{:.4f}-T{:.4f}.pdf".format(delta, temperature1, temperature2),
    #             format="pdf", dpi=300, bbox_inches='tight')


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
