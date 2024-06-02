#
# Modified to support further configurations and to store spikes and optionally weights.
#

from brian2 import *
import time
from globparams import *
import pandas as pd
import numpy as np
from pathlib import Path

defaultclock.dt = 1*ms
prefs.core.default_float_dtype = float32

if DELTA:
    eqs_neurons = '''
    dv/dt = (rand() - v * OM_DECAY) / (1*ms) : 1
    dspiked/dt = -spiked / (1*ms) : 1
    '''
    pre = '''
    v_post += w
    spiked_pre = 1
    '''
else:
    eqs_neurons = '''
    dv/dt = (ge + rand() - v * OM_DECAY) / (1*ms) : 1
    dge/dt = -ge / (1*ms) : 1
    dspiked/dt = -spiked / (1*ms) : 1
    '''
    pre = '''
    ge_post += w
    spiked_pre = 1
    '''

synaptic_model = '''
w : 1
'''

post = '''
w = clip(w + spiked_pre * STDP_SPEED, 0.0, 1.0) 
'''

N = NeuronGroup(SIZE, eqs_neurons, threshold='v>VT', reset='v = VR', method=METHOD)

if CONN:
    S = Synapses(N, N, synaptic_model, on_pre=pre, on_post=post)

    S.connect()
    S.w = 'rand()/SIZE' #initialize

    if WGTS:
        w_ini = np.array(S.w)


M = SpikeMonitor(N)

start = time.time()
run(DURATION*ms, report='text')
print("simulation time: ", time.time()-start)

base_out = Path("tmpdata") / f"brian_LIF_Delta_{DELTA}_{METHOD}_{SIZE}_Conn_{CONN}_STDP_{STDP_SPEED>0}"

pd.DataFrame.from_dict({'senders': np.array(M.i), 'times': M.t/ms}).to_csv(
                              f"{base_out}_spikes.dat", sep="\t")

if WGTS:
    w_post = np.array(S.w)
    pd.DataFrame.from_dict({'w_ini': w_ini, 'w_post': w_post}).to_csv(
        f"{base_out}_weights.dat", sep="\t")

if PLOT:
    plot(M.t/ms, M.i, '.')
    show()
