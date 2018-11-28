import mdp
try:
    from pyNN.pcsim import *
    from pypcsim import *
except ImportError:
    pass
try:
    import brian
except ImportError:
    pass

import scipy
from Oger.utils import spikes_to_states, inputs_to_spikes

class BrianIFReservoirNode(mdp.Node):
    def __init__(self, input_dim, output_dim, dtype, input_scaling=100, input_conn_frac=.5, dt=1, we_scaling=2, wi_scaling=.5, we_sparseness=.1, wi_sparseness=.1):
        super(BrianIFReservoirNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.taum = 20 * brian.ms
        self.taue = 5 * brian.ms
        self.taui = 10 * brian.ms
        self.Vt = 15 * brian.mV
        self.Vr = 0 * brian.mV
        self.frac_e = .75
        self.input_scaling = input_scaling
        self.input_conn_frac = input_conn_frac
        self.dt = dt
        self.we_scaling = we_scaling
        self.wi_scaling = wi_scaling
        self.we_sparseness = we_sparseness
        self.wi_sparseness = wi_sparseness

        self.eqs = brian.Equations('''
              dV/dt  = (I-V+ge-gi)/self.taum : volt
              dge/dt = -ge/self.taue    : volt 
              dgi/dt = -gi/self.taui    : volt
              I: volt
              ''')
        self.G = brian.NeuronGroup(N=output_dim, model=self.eqs, threshold=self.Vt, reset=self.Vr)
        self.Ge = self.G.subgroup(int(scipy.floor(output_dim * self.frac_e))) # Excitatory neurons 
        self.Gi = self.G.subgroup(int(scipy.floor(output_dim * (1 - self.frac_e))))

        self.internal_conn = brian.Connection(self.G, self.G)
        self.we = self.we_scaling * scipy.random.rand(len(self.Ge), len(self.G)) * brian.nS
        self.wi = self.wi_scaling * scipy.random.rand(len(self.Ge), len(self.G)) * brian.nS

        self.Ce = brian.Connection(self.Ge, self.G, 'ge', sparseness=self.we_sparseness, weight=self.we)
        self.Ci = brian.Connection(self.Gi, self.G, 'gi', sparseness=self.wi_sparseness, weight=self.wi)

        #self.internal_conn.connect(self.G, self.G, self.w_res)

        self.Mv = brian.StateMonitor(self.G, 'V', record=True, timestep=10)
        self.Ms = brian.SpikeMonitor(self.G, record=True)
        self.w_in = self.input_scaling * (scipy.random.rand(self.output_dim, self.input_dim)) * (scipy.random.rand(self.output_dim, self.input_dim) < self.input_conn_frac)
        self.network = brian.Network(self.G, self.Ce, self.Ci, self.Ge, self.Gi, self.Mv, self.Ms)

    def is_trainable(self):
        return False

    def is_invertible(self):
        return False

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        #self.G.I = brian.TimedArray(10000 * x * brian.mV, dt=1 * brian.ms)
        self.G.I = brian.TimedArray(100 * scipy.dot(x, self.w_in.T) * brian.mV, dt=1 * brian.ms)
        self.network = brian.Network(self.G, self.Mv, self.Ms)
        self.network.reinit()
        self.network.run((x.shape[0] + 1) * brian.ms)
        retval = self.Mv.values[:, 0:x.shape[0]].T

class GenericSpikingReservoirNode(mdp.Node):
    """
    A generic PyNN spiking reservoir node composed of a population of excitatory neurons
    and a population of inhibitory neurons. The populations do not have any spatial structure.
    """
    def __init__(self, input_dim, size, dtype, rngseed,
                 exc_frac, exc_cell_type, exc_cell_params,
                 inh_cell_type, inh_cell_params, exc_connector, inh_connector, input_connector,
                 kernel, inp2spikes_conversion,
                 syn_dynamics, inp_syn_dynamics):
        """ Create the spiking neural network in PyNN
        """
        output_dim = size
        super(GenericSpikingReservoirNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)

        self.kernel = kernel
        self.inp2spikes_conversion = inp2spikes_conversion

        # === Define parameters ======================================================== 
        self.n = output_dim    # Number of cells
        n_exc = int(exc_frac * self.n)
        n_inh = self.n - n_exc
        n_inp = input_dim

        self.simDT = dt = 1           # (ms)

        setup(timestep=dt, min_delay=dt)

        self.rng = NumpyRNG(seed=rngseed)

        self.exc_cells = Population((n_exc,), exc_cell_type, exc_cell_params, label="Excitatory_Cells")
        self.inh_cells = Population((n_inh,), inh_cell_type, inh_cell_params, label="Inhibitory_Cells")

        self.input_cells = Population((n_inp,), SpikeSourceArray, {'spike_times': array([]) }, label="input")


        self.e2e_conn = Projection(self.exc_cells, self.exc_cells, exc_connector, target='excitatory', rng=self.rng, synapse_dynamics=syn_dynamics['e2e'])
        self.e2i_conn = Projection(self.exc_cells, self.inh_cells, exc_connector, target='excitatory', rng=self.rng, synapse_dynamics=syn_dynamics['e2i'])
        self.i2e_conn = Projection(self.inh_cells, self.exc_cells, inh_connector, target='inhibitory', rng=self.rng, synapse_dynamics=syn_dynamics['i2e'])
        self.i2i_conn = Projection(self.inh_cells, self.inh_cells, inh_connector, target='inhibitory', rng=self.rng, synapse_dynamics=syn_dynamics['i2i'])

        self.inp_exc_conn = Projection(self.input_cells, self.exc_cells, input_connector, rng=self.rng, synapse_dynamics=inp_syn_dynamics['2exc'])
        self.inp_inh_conn = Projection(self.input_cells, self.inh_cells, input_connector, rng=self.rng, synapse_dynamics=inp_syn_dynamics['2inh'])

        self.exc_cells.record()
        self.inh_cells.record()

        self.ids_exc = [ id for id in self.exc_cells.all() ]
        self.ids_inh = [ id for id in self.inh_cells.all() ]



    def is_trainable(self):
        return False

    def is_invertible(self):
        return False

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        """ Executes simulation with input vector x.
        """

        steps = x.shape[0]

        Tstep = 10.0  # in milliseconds

        in_spikes = inputs_to_spikes(x, self.inp2spikes_conversion)

        # setup the new inputs and reset
        for cell, val in zip(self.input_cells, in_spikes):
            setattr(cell, "spike_times", val)

        reset()
        run(steps * Tstep)

        # retrieve the spikes
        exc_rec_spikes = self.exc_cells.getSpikes()
        inh_rec_spikes = self.inh_cells.getSpikes()


        exc_spikes_dict = {}
        for id in self.ids_exc:
            exc_spikes_dict[id] = []

        for id, st in exc_rec_spikes:
            exc_spikes_dict[id].append(st)

        inh_spikes_dict = {}
        for id in self.ids_inh:
            inh_spikes_dict[id] = []

        for id, st in inh_rec_spikes:
            inh_spikes_dict[id].append(st)

        spikes = []
        for id in self.ids_exc:
            spikes.append(exc_spikes_dict[id])

        for id in self.ids_inh:
            spikes.append(inh_spikes_dict[id])

        self.states = spikes_to_states(spikes, self.kernel, steps, Tstep, self.simDT)

        return self.states


class SpikingRandomIFReservoirNode(GenericSpikingReservoirNode):
    """
    A PyNN reservoir node composed of current-based Leaky Integrate-and-fire neurons 
    with random connectivity. The synapses are static. 
    """
    def __init__(self, input_dim, size, dtype, rngseed,
                 exc_frac, exc_w, inh_w,
                 input_w, cell_params, syn_delay,
                 Cprob_exc, Cprob_inh, Cprob_inp, kernel, inp2spikes_conversion):
        """ Create the spiking neural network in PyNN
        """

        super(SpikingRandomIFReservoirNode, self).__init__(input_dim=input_dim, size=size, dtype=dtype, rngseed=rngseed,
                             exc_frac=exc_frac, exc_cell_type=IF_curr_exp, exc_cell_params=cell_params,
                             inh_cell_type=IF_curr_exp, inh_cell_params=cell_params,
                             exc_connector=FixedProbabilityConnector(Cprob_exc, weights=exc_w, delays=syn_delay),
                             inh_connector=FixedProbabilityConnector(Cprob_inh, weights=inh_w, delays=syn_delay),
                             input_connector=FixedProbabilityConnector(Cprob_inp, weights=input_w, delays=syn_delay),
                             kernel=kernel, inp2spikes_conversion=inp2spikes_conversion,
                             syn_dynamics={'e2e':None, 'e2i':None, 'i2e':None, 'i2i':None},
                             inp_syn_dynamics={'2exc':None, '2inh':None },)



class SpikingRandomIFDynSynReservoirNode(GenericSpikingReservoirNode):
    """
    A PyNN reservoir node composed of current-based Leaky Integrate-and-fire neurons 
    with random connectivity. The synapses are dynamic with the default parameters
    set from the publications:
    
    Markram H,Wang Y, Tsodyks M, Differential signaling via the same axon of 
    neocortical pyramidal neurons. Proc Natl Acad Sci USA 1998;95:5323-5328. 
    
    Gupta A, Wang Y, Markram H. 2000. Organizing principles for a diversity
    of GABAergic interneurons and synapses in the neocortex. Science
    287:273--278. 
    """
    def __init__(self, input_dim, size, dtype, rngseed,
                 exc_frac, exc_w, inh_w,
                 input_w, cell_params, syn_delay,
                 Cprob_exc, Cprob_inh, Cprob_inp, kernel, inp2spikes_conversion,
                 syn_dynamics = None,
                 inp_syn_dynamics={ '2exc': None,
                                     '2inh': None }):
        """ Create the spiking neural network in PyNN
        """
        if syn_dynamics is None:
            syn_dynamics = { 'e2e': SynapseDynamics(
                                        fast=TsodyksMarkramMechanism(U=0.5, tau_rec=1100.0, tau_facil=50.0)),
                                  'e2i': SynapseDynamics(
                                        fast=TsodyksMarkramMechanism(U=0.05, tau_rec=125.0, tau_facil=1200.0)),
                                  'i2e': SynapseDynamics(
                                        fast=TsodyksMarkramMechanism(U=0.25, tau_rec=700.0, tau_facil=20.0)),
                                  'i2i': SynapseDynamics(
                                        fast=TsodyksMarkramMechanism(U=0.32, tau_rec=144.0, tau_facil=60.0))
                                }
        super(SpikingRandomIFDynSynReservoirNode, self).__init__(input_dim=input_dim, size=size, dtype=dtype, rngseed=rngseed,
                             exc_frac=exc_frac, exc_cell_type=IF_curr_exp, exc_cell_params=cell_params,
                             inh_cell_type=IF_curr_exp, inh_cell_params=cell_params,
                             exc_connector=FixedProbabilityConnector(Cprob_exc, weights=exc_w, delays=syn_delay),
                             inh_connector=FixedProbabilityConnector(Cprob_inh, weights=inh_w, delays=syn_delay),
                             input_connector=FixedProbabilityConnector(Cprob_inp, weights=0.2, delays=syn_delay),
                             kernel=kernel, inp2spikes_conversion=inp2spikes_conversion,
                             syn_dynamics=syn_dynamics,
                             inp_syn_dynamics=inp_syn_dynamics)




class SpatialSpikingReservoirNode(mdp.Node):
    """
    A generic PyNN spiking reservoir node with two populations of excitatory and inhibitory cells
    that have spatial structure.
    """
    def __init__(self, input_dim, dtype, rngseed,
                 n_exc, n_inh, exc_structure, inh_structure,
                 exc_cell_type, exc_cell_params,
                 inh_cell_type, inh_cell_params,
                 exc_connector, inh_connector, input_connector,
                 kernel, inp2spikes_conversion,
                 syn_dynamics, inp_syn_dynamics):
        """ Create the spiking neural network in PyNN
        """
        output_dim = n_exc + n_inh
        super(SpatialSpikingReservoirNode, self).__init__(input_dim=input_dim,
                                                          output_dim=output_dim, dtype=dtype)

        self.kernel = kernel
        self.inp2spikes_conversion = inp2spikes_conversion

        # === Define parameters ========================================================

        self.n = output_dim    # Number of cells

        n_inp = input_dim

        self.simDT = dt = 1           # (ms)

        setup(timestep=dt, min_delay=dt)

        self.rng = NumpyRNG(seed=rngseed)

        self.exc_cells = Population(n_exc, exc_cell_type, exc_cell_params, structure=exc_structure)
        self.inh_cells = Population(n_inh, inh_cell_type, inh_cell_params, structure=inh_structure)

        self.input_cells = Population(n_inp, SpikeSourceArray, {'spike_times': array([]) }, structure=None)


        self.e2e_conn = Projection(self.exc_cells, self.exc_cells, exc_connector, target='excitatory', rng=self.rng, synapse_dynamics=syn_dynamics['e2e'])
        self.e2i_conn = Projection(self.exc_cells, self.inh_cells, exc_connector, target='excitatory', rng=self.rng, synapse_dynamics=syn_dynamics['e2i'])
        self.i2e_conn = Projection(self.inh_cells, self.exc_cells, inh_connector, target='inhibitory', rng=self.rng, synapse_dynamics=syn_dynamics['i2e'])
        self.i2i_conn = Projection(self.inh_cells, self.inh_cells, inh_connector, target='inhibitory', rng=self.rng, synapse_dynamics=syn_dynamics['i2i'])

        self.inp_exc_conn = Projection(self.input_cells, self.exc_cells, input_connector, rng=self.rng, synapse_dynamics=inp_syn_dynamics['2exc'])
        self.inp_inh_conn = Projection(self.input_cells, self.inh_cells, input_connector, rng=self.rng, synapse_dynamics=inp_syn_dynamics['2inh'])

        self.exc_cells.record()
        self.inh_cells.record()

        self.ids_exc = [ id for id in self.exc_cells.all() ]
        self.ids_inh = [ id for id in self.inh_cells.all() ]



    def is_trainable(self):
        return False

    def is_invertible(self):
        return False

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        """ Executes simulation with input vector x.
        """

        steps = x.shape[0]

        Tstep = 10.0  # in milliseconds

        in_spikes = inputs_to_spikes(x, self.inp2spikes_conversion)

        # setup the new inputs and reset
        for cell, val in zip(self.input_cells, in_spikes):
            setattr(cell, "spike_times", val)

        reset()
        run(steps * Tstep)

        # retrieve the spikes
        exc_rec_spikes = self.exc_cells.getSpikes()
        inh_rec_spikes = self.inh_cells.getSpikes()


        exc_spikes_dict = {}
        for id in self.ids_exc:
            exc_spikes_dict[id] = []

        for id, st in exc_rec_spikes:
            exc_spikes_dict[id].append(st)

        inh_spikes_dict = {}
        for id in self.ids_inh:
            inh_spikes_dict[id] = []

        for id, st in inh_rec_spikes:
            inh_spikes_dict[id].append(st)

        spikes = []
        for id in self.ids_exc:
            spikes.append(exc_spikes_dict[id])

        for id in self.ids_inh:
            spikes.append(inh_spikes_dict[id])

        self.states = spikes_to_states(spikes, self.kernel, steps, Tstep, self.simDT)
        return self.states
