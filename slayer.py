import numpy as np
import matplotlib.pyplot as plt
import math
from netParams import NetParams, SimParams

netParams = NetParams(50, 1.2, 1.0, 2, 1)
simParams = SimParams(10, 10)


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


# PSP = conv_kernel(netParams, simParams)
# plt.plot(np.arange(len(PSP)), PSP)
# plt.show()

# REF = ref_kernel(netParams, simParams)
# plt.plot(np.arange(len(REF)), REF)
# plt.show()


def spike_to_PSP(spikes, netParams: NetParams, simParams: SimParams):
    PSP = []
    ker = conv_kernel(netParams, simParams)
    for i in range(len(spikes)):
        neuron_PSP = np.zeros((len(spikes), len(spikes[1])))
        for j in range(len(spikes[i])):
            if spikes[i][j] == 1:
                count = 0
                for k in range(j, j + simParams.marginKer):
                    neuron_PSP[j] += ker[count]
                    count += 1


def compute_potential_next_layer(PSP, weights):
    return PSP @ weights


def spike_function(potential, netParams: NetParams, simParams: SimParams):
    refKernel = ref_kernel(netParams, simParams)
    spikes = np.zeros([potential.shape[0], potential.shape[1]])
    for i in range(len(potential)):
        for j in range(len(potential[i])):
            if potential[i][j] > netParams.threshold:

                # Spike
                spikes[1]

                # Add refractory kernel
                count = 0
                for k in range(j, simParams.marginRef + 1):
                    # Gatekeep for Index Out of Bounds error
                    if k > len(potential[i]):
                        break
                    # Decrease the potential
                    potential[i][k] += refKernel[count]
                    count += 1
