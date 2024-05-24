import sys

PLOT = not 'no_plot' in sys.argv            # do not create plots
CONN = not 'no_conn' in sys.argv            # do not create any connections
WGTS = CONN and 'rec_weights' in sys.argv   # record initial and post-simulation weights
DELTA = 'psc_delta' in sys.argv             # use delta-shaped instead of exponential post-synaptic currents

STDP_SPEED = 0 if 'no_stdp' in sys.argv else 0.001

# method is only relevant for Brian2
METHOD = 'exponential_euler' if 'exponential_euler' in sys.argv else 'euler'

DURATION = 300
SIZE = int(sys.argv[1])

VT = 6.0
VR = 0.0

# The following correspond to tau = 10 ms, C = 1 pF
DECAY = 0.9
OM_DECAY = 1 - DECAY
