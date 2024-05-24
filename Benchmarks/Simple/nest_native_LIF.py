#
# Modified to support further configurations and to store spikes and optionally weights.
#

import nest
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from globparams import *
import random
import pandas as pd


#############################################################
# model generation
#############################################################

from pynestml.codegeneration.nest_code_generator_utils import NESTCodeGeneratorUtils

simple_neuron_str = """
neuron simple_neuron:

    state:
        v mV = 0 mV

    equations:
        v' = (I_e/pA * mV - v * decay) / ms

    parameters:
        vt real = 6.1
        vr real = 0.0
        input_strength real = 1.0
        decay real = 0.1


    input:
        spikes <- spike
        I_e pA <- continuous

    output:
        spike

    update:
        integrate_odes()
        v += spikes * input_strength * s * mV

        # threshold crossing
        if v >= vt * mV:
            v = vr * mV
            emit_spike()
"""

simple_stdp_synapse = """
synapse stdp_nn_symm:
    state:
        w real = 1.0
        tb ms = 0. ms

    parameters:
        dly ms = 1.0 ms  @nest::delay
        tau_tr_pre ms = 1.0 ms
        stdp_speed real = 0.01

    input:
        pre_spikes <- spike
        post_spikes <- spike

    output:
        spike

    onReceive(post_spikes):
        if t <= (tb + 1*ms):
            if tb < t:
                w += stdp_speed

    onReceive(pre_spikes):
        tb = t
        w = w
        deliver_spike(w, dly)
"""

(
    module_name,
    neuron_model_name,
    synapse_model_name,
) = NESTCodeGeneratorUtils.generate_code_for(
    simple_neuron_str, simple_stdp_synapse, post_ports=["post_spikes"])

print(module_name,
    neuron_model_name,
    synapse_model_name)
nest.Install(module_name)


#############################################################
# simulation
#############################################################

# Define parameters
num_neurons = SIZE
simulation_time = DURATION  # ms
dt = 1.0  # ms

# Set up the NEST simulation
nest.ResetKernel()
nest.SetKernelStatus(
    {"resolution": dt, "print_time": False, "local_num_threads": os.cpu_count(),
         "overwrite_files": True, "rng_seed": int(1 + random.random() * (2**32-2))}
)

# Create neurons
neuron_params = {"vt": VT, "vr": VR, "decay": OM_DECAY}
neurons = nest.Create(neuron_model_name, num_neurons, params=neuron_params)

# Create synapses
synapse_params = {
    "synapse_model": synapse_model_name,
    "w": nest.random.uniform(min=0.0, max=1.0 / num_neurons),
    "stdp_speed": STDP_SPEED,
}

if CONN:
    nest.Connect(neurons, neurons, "all_to_all", synapse_params)

# add voltage fluctuations to neurons

for i in range(num_neurons):
    times = list(np.arange(1.0, simulation_time * 1.0 + 1, 1.0 * dt))
    values = list(np.random.rand(int(simulation_time)))
    ng = nest.Create("step_current_generator")
    ng.set({"amplitude_times": times, "amplitude_values": values})
    nest.Connect(ng, neurons[i])

sr = nest.Create("spike_recorder")
nest.Connect(neurons, sr)

if WGTS:
    conns = nest.GetConnections(neurons, neurons)
    w_ini = conns.w

with nest.RunManager():
    start = time.time()
    nest.Run(simulation_time)
    print("simulation time: ", time.time() - start)

if WGTS:
    w_post = conns.w

base_out = f"nest_native_LIF_{num_neurons}_Conn_{CONN}_STDP_{STDP_SPEED>0}"
pd.DataFrame.from_dict(sr.events).to_csv(f"{base_out}_spikes.dat", sep="\t")

if WGTS:
    pd.DataFrame.from_dict({'w_ini': w_ini, 'w_post': w_post}).to_csv(f"{base_out}_weights.dat", sep="\t")
if PLOT:
    spike_rec = nest.GetStatus(sr, keys="events")[0]
    print(f"Total spikes: {len(spike_rec['times'])}")
    plt.plot(spike_rec["times"], spike_rec["senders"], ".k")
    plt.ylabel("neurons")
    plt.xlabel("t")
    plt.show()

