from PymoNNto import *
import time

settings = {'dtype': float32, 'synapse_mode': SxD}


class SpikeGeneration(Behavior):
    def initialize(self, neurons):
        neurons.spikes = neurons.vector('bool')
        neurons.spikesOld = neurons.vector('bool')
        neurons.voltage = neurons.vector()
        self.threshold = self.parameter('threshold')
        self.decay = self.parameter('decay')

    def iteration(self, neurons):
        neurons.spikesOld = neurons.spikes.copy()
        neurons.spikes = neurons.voltage > self.threshold
        #print(np.sum(neurons.spikes))# number of active neurons around 1.5%
        #neurons.voltage.fill(0.0)
        neurons.voltage *= np.invert(neurons.spikes) #reset
        neurons.voltage *= self.decay #voltage decay



class Input(Behavior):
    def initialize(self, neurons):
        self.strength = self.parameter('strength')
        for s in neurons.synapses(afferent, 'GLU'):
            s.W = s.matrix('random')
            s.W /= np.sum(s.W, axis=0) #normalize during initialization

    def iteration(self, neurons):
        neurons.voltage += neurons.vector('random')
        for s in neurons.synapses(afferent, 'GLU'):
            input = np.sum(s.W[s.src.spikes], axis=0)
            s.dst.voltage += input * self.strength


class STDP(Behavior):
    def initialize(self, neurons):
        self.speed = self.parameter('speed')

    def iteration(self, neurons):
        for s in neurons.synapses(afferent, 'GLU'):
            mask = np.ix_(s.src.spikesOld, s.dst.spikes)
            s.W[mask] += self.speed
            s.W[mask] = np.clip(s.W[mask], 0.0, 1.0)


#class Norm(Behavior):
#    def iteration(self, neurons):
#        if neurons.iteration % 10 == 9:
#            for s in neurons.synapses(afferent, 'GLU'):
#                s.W /= np.sum(s.W, axis=0)


net = Network(settings=settings)
NeuronGroup(net, tag='NG', size=10000, behavior={
    1: SpikeGeneration(threshold=6.1, decay=0.9),
    2: Input(strength=1.0),
    3: STDP(speed=0.001),
    #4: Norm(),
    5: EventRecorder(['spikes'])
})

SynapseGroup(net, src='NG', dst='NG', tag='GLU')
net.initialize()

start = time.time()
net.simulate_iterations(100)
print("simulation time: ", time.time()-start)

plt.plot(net['spikes.t', 0], net['spikes.i', 0], '.k')
plt.show()