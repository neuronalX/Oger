import mdp
import numpy as np
try:
    from NeuroTools import stgen
    from pyNN.pcsim import *
    from pypcsim import *
except ImportError:
    pass

class poisson_gen:
    '''Container class for a poisson generator signal-to-spiketrain convertor
    '''
    def __init__(self, rngseed, RateScale=1e6, Tstep=10):
        ''' Create a poisson generator, using the given seed for the random number generator
        '''
        self.RateScale = RateScale
        self.Tstep = Tstep
        self.rng = NumpyRNG(seed=rngseed)
        self.stgen = stgen.StGen(self.rng)

    def __call__(self, xi):
        '''
            PoissonGen(xi) -> spikes
            Return a poisson spike train for the given signal xi
        '''
        return self.stgen.inh_poisson_generator(self.RateScale * xi, mdp.numx.arange(0, self.Tstep * len(xi), self.Tstep), t_stop=self.Tstep * len(xi), array=True)


def spikes_to_states(spikes, kernel, steps, Tstep, simDT):
    '''
        spikes_to_states(spikes, kernel, steps, Tstep, simDT) -> states
        Convert spikes to liquid states using a given convolution kernel
        Input arguments: 
            - spikes: the spike train to be converted
            - kernel: the convolution kernel (a function of a vector which returns a vector)
            - steps: number of timesteps in the resulting vector in the analog domain
            - Tstep: timestep (sampling period) of the resulting vector in the analog domain
            - simDt: simulation timestep
    '''
    spikes_bin = np.zeros((len(spikes), int(steps * Tstep / simDT)))

    for i in range(len(spikes)):
        for j in range(len(spikes[i])):
            spikes_bin[i, min(int(spikes[i][j] / simDT), spikes_bin.shape[1] - 1)] += 1.0

    liq_states = np.array([np.convolve(spikes_bin[i], kernel, mode='same') for i in range(len(spikes_bin)) ], dtype=float)
    states = np.swapaxes(liq_states, 0, 1)[::int(Tstep / simDT), :]
    return states


def inputs_to_spikes(x, inp2spikes_conversion, *args):
    ''' 
        inputs_to_spikes(x, inp2spikes_conversion) -> in_spikes
        Convert an N-d analog signal to spiketrains, using the given spiketrain conversion function
        Input arguments:
            - x: the dataset to be converted (a numpy array)
            - inp2spikes_conversion: the function that performs the spike conversion
    '''
    n_inputs = x.shape[1]
    in_spikes = []
    for i in range(n_inputs):
            in_spikes.append(inp2spikes_conversion(x[:, i], *args))
    return in_spikes


def deltasigma(input):
    ''' deltasigma(input) -> spikes
        Apply a delta-sigma modulator the input to generate spikes.
        
        Input arguments:
            - input: the signal to be converted (a numpy array)
        
        See: Benjamin Schrauwen Towards applicable spiking neural networks Doctoraatsproefschrift Faculteit Ingenieurswetenschappen, Universiteit Gent, pp. (2008) 
    '''

    i = 0
    s = 0
    spike = np.zeros_like(input)
    for j, inp in enumerate(input):
        i = i + inp - s

        s = (i > 0) * 2 - 1
        spike[j] = s
    return spike

def HSA(input, filter, threshold):
    ''' HSA(input, filter, threshold) -> spikes
        Apply the Hough Spiker algorithm to the input to generate spikes using the given filter and threshold
        
        Input arguments:
            - input: the signal to be converted (a numpy array)
            - filter: the decoding filter
            - threshold: the threshold based on which to generate the spikes
        See: Benjamin Schrauwen Towards applicable spiking neural networks Doctoraatsproefschrift Faculteit Ingenieurswetenschappen, Universiteit Gent, pp. (2008) 
    '''

    N = input.shape[0]
    P = filter.shape[0]

    spikes = np.zeros(N)
    input = np.concatenate((input, np.zeros(P)))
    for i in range (N):
        segment = input[i:i + P ]

        if np.sum((filter - segment) * (segment < filter)) <= threshold:
            spikes[i] = 1
            input[i:i + P ] = segment - filter

    return spikes

def BSA(input, filter, threshold):
    ''' BSA(input, filter, threshold) -> spikes
        Apply the Bens Spiker algorithm to the input to generate spikes using the given filter and threshold
        
        Input arguments:
            - input: the signal to be converted (a numpy array)
            - filter: the decoding filter
            - threshold: the threshold based on which to generate the spikes
        See: Benjamin Schrauwen Towards applicable spiking neural networks Doctoraatsproefschrift Faculteit Ingenieurswetenschappen, Universiteit Gent, pp. (2008) 
    '''
    N = input.shape[0]
    P = filter.shape[0]

    spikes = np.zeros(N)
    input = np.concatenate((input, np.zeros(P)))
    for i in range (N):
        segment = input[i:i + P ]

        if np.sum(np.absolute(filter - segment)) <= np.sum(np.absolute(segment)) - threshold:
            spikes[i] = 1
            input[i:i + P ] = segment - filter

    return spikes

def exp_kernel(tau, dt):
    '''
        exp_kernel(tau, dt) -> kernel_result
        Exponential kernel for filtering spike trains
    '''
    return mdp.numx.exp(-mdp.numx.arange(0, 10 * tau, dt) / tau)

def gauss_kernel(tau, dt):
    '''
        gauss_kernel(tau, dt) -> kernel_result
        Gaussian kernel for filtering spike trains
    '''
    x = -mdp.numx.arange(-5 * tau, 5 * tau, dt) / tau
    return mdp.numx.exp(-x ** 2)
