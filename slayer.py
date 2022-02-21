import numpy as np
import matplotlib.pyplot as plt
import math
from netParams import NetParams, SimParams
import logging

netParams = NetParams(50, 1.2, 1.0, 2, 1)
netParams.init_weights()
simParams = SimParams(10, 10, 10, 200)


def conv_kernel(params: NetParams, simulation: SimParams):
    PSP = []
    for i in range(simulation.marginKer):
        ker = i / params.tauSR * math.exp(1 - i / params.tauSR)
        PSP.append(ker)

    return np.array(PSP)


def ref_kernel(params: NetParams, simulation: SimParams):
    REF = []
    for i in range(simulation.marginRef):
        ker = - params.scaleRef * params.threshold * \
            math.exp(1 - i / params.tauSR)
        REF.append(ker)

    return np.array(REF)


def spike_to_PSP(spikes, netParams: NetParams, simParams: SimParams):
    PSP = []
    ker = conv_kernel(netParams, simParams)
    for i in range(len(spikes)):
        neuron_PSP = np.zeros((len(spikes), len(spikes[1])))
        for j in range(len(spikes[i])):
            if spikes[i][j] == 1:
                count = 0
                for k in range(j, j + simParams.marginKer):

                    # Gatekeep the Index out of bounds
                    if k > len(spikes[i]):
                        break

                    # Add the effect of spikes
                    neuron_PSP[j] += ker[count]
                    count += 1

    return PSP


def compute_potential_next_layer(PSP, weights):
    return PSP @ weights


def spike_function(potential, netParams: NetParams, simParams: SimParams):
    '''
        Spike function and addition of the refractory kernel\
            if the threshold is reached
    '''
    refKernel = ref_kernel(netParams, simParams)
    spikes = np.zeros([potential.shape[0], potential.shape[1]])
    for i in range(len(potential)):
        for j in range(len(potential[i])):
            if potential[i][j] > netParams.threshold:

                # Spike
                spikes[i][j] = 1

                # Add refractory kernel
                count = 0
                for k in range(j, simParams.marginRef + 1):
                    # Gatekeep for Index Out of Bounds error
                    if k > len(potential[i]):
                        break
                    # Decrease the potential
                    potential[i][k] += refKernel[count]
                    count += 1

    return spikes


def rho_function(netParams: NetParams, simParams: SimParams, u_t):
    '''
        Derivative of the spike function
    '''
    RHO = []
    alpha = 2
    beta = 1.3
    for i in range(simParams.marginRho):
        ker = 1/alpha * math.exp(-beta * math.abs(u_t - netParams.threshold))
        RHO.append(ker)

    return np.array(RHO)


def spike_time_loss_function(spike_in, spike_out):
    '''
        Loss based on the timing of the spikes
    '''
    error = spike_to_PSP(spike_in) - spike_to_PSP(spike_out)
    return 1/2 * np.sum(error**2)


def spike_count_loss_function(spike_in, spike_out):
    '''
        Loss based on the number of spikes
    '''
    interaval = simParams.timeline
    error_count = (spike_in - spike_out) / interaval
    return 1/2 * np.sum(error_count ** 2)


def forward(sample, netParams: NetParams, simParams: SimParams):
    '''
        Makes a forward pass and returns pikes
    '''
    layer1 = spike_function(compute_potential_next_layer(spike_to_PSP(
        sample, netParams, simParams), netParams.weights1), netParams, simParams)
    layer2 = spike_function(compute_potential_next_layer(spike_to_PSP(
        layer1, netParams, simParams), netParams.weights2), netParams, simParams)

    max = 0
    i = 0
    for neuron in layer2:
        if np.sum(neuron) > max:
            cat = i
            max = np.sum(neuron)
        i+=1