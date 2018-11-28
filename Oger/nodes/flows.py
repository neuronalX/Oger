import mdp
from mdp import numx

# TODO: biMDP has the ability to introspect flows and nodes, we could use
# that mechanism instead of this custom flow. This would possibly also
# allow parallelization 


class FreerunFlow(mdp.Flow):
    ''' This flow enables freerun execution, e.g. for signal generation tasks. The constructor takes an additional argument freerun_steps.
        The constructor takes two additional arguments:
            - freerun_steps: the number of timesteps to run in freerun mode (i.e. non-teacher-forced). This needs to be set explicitly.
            - external_input_range (optional): a numpy array denoting the subset of the input dimensions which are provided externally (i.e. not fed back). If this is None (default), all signals are fed back.  This range should start at zero.
   
        Its functionality differs from the standard flow in two ways:
            - The train function internally generates (x,y) training pairs from the input argument x, where y is shifted one timestep to the right w.r.t. x for one-step-ahead prediction
            - The execute function takes a numpy array as input, whereby the timesteps up to N-freerun_steps are used to warmup the flow (i.e. teacher forcing), and the final freerun_steps timesteps are executed in freerun mode, i.e. the output of the flow is fed back as input. The return argument of the execute function is a concatenation of the warmup (teacher forced) signals with the freerun (generated) signals. It has the same length as the input.
    '''
    def __init__(self, flow, crash_recovery=False, verbose=False, freerun_steps=None, external_input_range=None):
        super(FreerunFlow, self).__init__(flow, crash_recovery, verbose)
        if freerun_steps is None:
            errstr = ("The FreerunFlow must be initialized with an explicit freerun horizon.")
            raise mdp.FlowException(errstr)
        self.freerun_steps = freerun_steps
        self.external_input_range = external_input_range

    def train(self, data_iterables):
        data_iterables = self._train_check_iterables(data_iterables)

        if self.external_input_range is None:
            external_input_range = []
        else:
            external_input_range = self.external_input_range

        # train each Node successively
        for i in range(len(self.flow)):
            if self.verbose:
                print "Training node #%d (%s)" % (i, str(self.flow[i]))
            if not data_iterables[i] == []:
                # Delay the input timeseries with len(flow)-1, because every node connection introduces a one timestep delay
                datax = [x[0:-i, :] for x in data_iterables[i][0]]
                datay = []
                for x in data_iterables[i][0]:
                    c = numx.array([True] * x.shape[1])
                    c[external_input_range] = False
                    datay.append(x[i:, c])
            else:
                datax, datay = [], []
            self._train_node(zip(datax, datay), i)
            if self.verbose:
                print "Training finished"

        self._close_last_node()


    def execute(self, x, nodenr=None):
        if not isinstance(x, mdp.numx.ndarray):
            errstr = ("FreerunFlows can only be executed using numpy arrays as input.")
            raise mdp.FlowException(errstr)

        if self.freerun_steps >= x.shape[0]:
            errstr = ("Number of freerun steps (%d) should be less than the number of timesteps in x (%d)" % (self.freerun_steps, x.shape[0]))
            raise mdp.FlowException(errstr)

        # Run the flow for warmup
        if self.freerun_steps > x.shape[0]:
            errstr = ("The number of freerun steps (%d) is larger than the input (%d):" % (self.freerun_steps, x.shape[0]))
            raise mdp.FlowException(errstr)

        if self.external_input_range is None:
            external_input_range = []
        else:
            external_input_range = self.external_input_range

        self._execute_seq(x[:-self.freerun_steps, :])
        freerun_range = mdp.numx.setdiff1d(range(x.shape[1]), external_input_range)
        self.fb_value = mdp.numx.atleast_2d(x[-self.freerun_steps, freerun_range])

        res = mdp.numx.zeros((self.freerun_steps, x.shape[1]))
        if self.external_input_range is None:
            for step in range(self.freerun_steps):
                res[step] = self.fb_value
                self.fb_value = self._execute_seq(mdp.numx.atleast_2d(self.fb_value))
        else:
            for step in range(self.freerun_steps):
                external_input = mdp.numx.atleast_2d(x[-self.freerun_steps + step, external_input_range])
                total_input = mdp.numx.atleast_2d(mdp.numx.concatenate((external_input, self.fb_value), 1))
                res[step] = total_input
                self.fb_value = self._execute_seq(total_input)
        return mdp.numx.concatenate((x[:-self.freerun_steps, :], res))

    def __add__(self, other):
        # append other to self
        if isinstance(other, mdp.Flow):
            flow_copy = list(self.flow).__add__(other.flow)
            # check dimension consistency
            self._check_nodes_consistency(flow_copy)
            # if no exception was raised, accept the new sequence
            return self.__class__(flow_copy, freerun_steps=self.freerun_steps)
        elif isinstance(other, mdp.Node):
            flow_copy = list(self.flow)
            flow_copy.append(other)
            # check dimension consistency
            self._check_nodes_consistency(flow_copy)
            # if no exception was raised, accept the new sequence
            return self.__class__(flow_copy)
        else:
            err_str = ('can only concatenate flow or node'
                       ' (not \'%s\') to flow' % (type(other).__name__))
            raise TypeError(err_str)

