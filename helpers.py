# -*- coding: utf-8 -*-
'''
Some utility functions used in the scripts.
'''
import ann


def get_net(incount, mingroup,
            func=ann.geneticnetwork.FITNESS_SURV_KAPLAN_MIN):
    '''
    A network construction function.

    Parameters:
    - incount, number of input neurons = number of input columns in data
    - mingroup, minimum group size (depends on your data)
    - func, fitness function. Should be one of KAPLAN_MIN or KAPLAN_MAX

    Returns:
    - a connected neural network with training parameters set.
    '''
    hidden_count = 4
    outcount = 2

    net = ann.geneticnetwork(incount, hidden_count, outcount)
    net.fitness_function = func

    net.mingroup = int(mingroup)

    # Be explicit here even though I changed the defaults
    net.connection_mutation_chance = 0.0
    net.activation_mutation_chance = 0
    # Training parameters used for all experiments
    net.crossover_method = net.CROSSOVER_TWOPOINT
    net.selection_method = net.SELECTION_TOURNAMENT
    net.population_size = 200
    net.generations = 1000
    net.weight_mutation_chance = 0.5
    net.weight_mutation_factor = 1.5
    net.crossover_chance = 0.75

    # Connect the neurons in a single layer
    ann.utils.connect_feedforward(net, [hidden_count],
                                  hidden_act=net.TANH,
                                  out_act=net.SOFTMAX)

    return net
