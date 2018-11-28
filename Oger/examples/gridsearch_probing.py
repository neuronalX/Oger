import mdp
import Oger
import pylab
import scipy as sp

class ReservoirNode_lyapunov(Oger.nodes.ReservoirNode):
    def __init__ (self, input_dim=1, output_dim=None, spectral_radius=0.9,
                 nonlin_func=sp.tanh, bias_scaling=0, input_scaling=1, dtype='float64', _instance=0,
                 w_in=None, w=None, w_bias=None, lyapunov_skip=1):
        ''' Note the additional parameter lyapunov_skip, which determines the number of timesteps to skip before
            recomputing the local lyapunov exponents
        '''
        super(ReservoirNode_lyapunov, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype,
                                                     spectral_radius=spectral_radius, nonlin_func=nonlin_func, bias_scaling=bias_scaling,
                                                     input_scaling=input_scaling, _instance=_instance, w_in=w_in, w=w, w_bias=w_bias)
        self.lyapunov_skip = lyapunov_skip
        self.probe_data = []

    def initialize(self):
        super(ReservoirNode_lyapunov, self).initialize()
        self.probe_data = []

    def _post_update_hook(self, states, inp, n):
        ''' Compute the Lyapunov exponents for this reservoir and store it in probe_data
            Assumes a tanh nonlinearity
        '''
        # We use broadcasting of the statevector
        if not (n + 1) % self.lyapunov_skip:
            jacob = (1 - states[n, :, mdp.numx.newaxis] ** 2) * self.w
            ll = sp.absolute(sp.linalg.eigvals(jacob))
            self.local_lyapunov[:, n / self.lyapunov_skip] = ll

    def _execute(self, x):
        # How many local lyapunov spectra will be computed?
        self.n_lyapunov = x.shape[0] / self.lyapunov_skip
        # Initialize the matrix which contains the local lyapunov exponents
        self.local_lyapunov = mdp.numx.zeros((self.output_dim, self.n_lyapunov))

        # Simulate the reservoir
        out = super(ReservoirNode_lyapunov, self)._execute(x)

        # Append the computed local lyapunov exponents to the probe_data field, so it can be saved by the Optimizer
        self.probe_data.append(sp.amax((self.local_lyapunov)))
        return out


if __name__ == '__main__':
    ''' Example of doing a grid-search with additional probing of (in this case) the Lyapunov exponents of the reservoir.
        The Optimizer will check for every node in the flow if there is a member variable probe_data. If so, the results of the probing 
        are stored in the Optimizer, in the member variable probe_data. This member variable is an N-dimensional list, with N the number of
        parameters being ranged over. Each element of this N-d list is a dictionary, indexed by the nodes in the flow, whose values are 
        the corresponding contents of the probe_data.
    '''
    input_size = 1
    inputs, outputs = Oger.datasets.narma30()

    data = [[], zip(inputs, outputs)]

    # construct individual nodes
    reservoir = ReservoirNode_lyapunov(input_size, 100, lyapunov_skip=100)
    readout = Oger.nodes.RidgeRegressionNode()

    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout])

    # Nested dictionary
    gridsearch_parameters = {reservoir:{'input_scaling': mdp.numx.arange(0.1, 2.2, .7), 'spectral_radius':mdp.numx.arange(0.1, 2.2, .7)}}

    # Instantiate an optimizer
    opt = Oger.evaluation.Optimizer(gridsearch_parameters, Oger.utils.nrmse)

    # Do the grid search
    opt.grid_search(data, flow, cross_validate_function=Oger.evaluation.train_test_only, training_fraction=.9)

    # Plot the maximal LLE for each parameter setting
    lle_max = mdp.numx.zeros_like(opt.errors)
    for i in range(opt.errors.shape[0]):
        for j in range(opt.errors.shape[1]):
            lle_max[i, j] = mdp.numx.amax(mdp.numx.mean(mdp.numx.array(opt.probe_data[i, j][reservoir]), 0))

    pylab.figure()
    pylab.imshow(mdp.numx.flipud(lle_max), cmap=pylab.jet(), interpolation='nearest', aspect="auto", extent=opt.get_extent(opt.parameters))
    pylab.ylabel('Spectral Radius')
    pylab.xlabel('Input scaling')
    pylab.suptitle('Max LLE')
    pylab.colorbar()
    pylab.show()
