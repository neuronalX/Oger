'''
This file contains parallel versions of the optimization algorithms for the Optimizer class

Created on Feb 23, 2011

@author: dvrstrae
'''
from Oger.evaluation import Optimizer
from Oger.evaluation import validate
from parallel import ParallelFlow
import mdp.parallel
import itertools
import scipy as sp

class ParameterSettingNode(mdp.Node):

    def __init__(self, flow, loss_function, cross_validation, input_dim=None, output_dim=None, dtype=None, *args, **kwargs):
        super(ParameterSettingNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.flow = flow
        self.loss_function = loss_function
        self.cross_validation = cross_validation

        # TODO: this is a bit messy...
        self.args = args
        self.kwargs = kwargs

    def execute(self, x):
        params = x[0]
        data = x[1]

        # Set all parameters of all nodes to the correct values
        node_set = set()
        for node_index, node_dict in params.items():
            for parameter, value in node_dict.items():
                node = self.flow[node_index]
                node_set.add(node)
                node.__setattr__(parameter, value)

        # Re-initialize all nodes that have the initialize method (e.g. reservoirs nodes)
        for node in node_set:
            if hasattr(node, 'initialize'):
                node._is_initialized = False

        # TODO: could this set of functions also be a parameter?
        return (mdp.numx.mean(validate(data, self.flow, self.loss_function, self.cross_validation, progress=False, *self.args, **self.kwargs)), self.flow)

    def is_trainable(self):
        return False


@mdp.extension_method("parallel", Optimizer)
def grid_search (self, data, flow, cross_validate_function, *args, **kwargs):
    ''' Do a combinatorial grid-search of the given parameters and given parameter ranges, and do cross-validation of the flowNode
        for each value in the parameter space.
        Input arguments are:
            - data: a list of iterators which would be passed to MDP flows
            - flow : the MDP flow to do the grid-search on
    '''

    if not hasattr(self, 'scheduler') or self.scheduler is None:
        err = ("No scheduler was assigned to the Optimizer so cannot run in parallel mode.")
        raise Exception(err)

    min_error = float('+infinity')

    # Get the number of parameter points to be evaluated for each of the parameters
    self.paramspace_dimensions = [len(r) for r in self.parameter_ranges]

    self.errors = mdp.numx.zeros(self.paramspace_dimensions)

    data_parallel = []

    # Construct all combinations
    self.param_space = list(itertools.product(*self.parameter_ranges))

    # Loop over all points in the parameter space
    for paramspace_index_flat, parameter_values in enumerate(self.param_space):
        params = {}

        for parameter_index, node_parameter in enumerate(self.parameters):
            node_index = flow.flow.index(node_parameter[0])
            if not node_index in params:
                params[node_index] = {}

        # Re-initialize all nodes that have the initialize method (e.g. reservoirs nodes)
        for node in flow:
            if hasattr(node, 'initialize'):
                node._is_initialized = False

            params[node_index][node_parameter[1]] = parameter_values[parameter_index]

        #data_parallel.append([[params, data]])
        data_parallel.append([[params]])

    parallel_flow = ParallelFlow([ParameterSettingNode(flow, self.loss_function, cross_validate_function, *args, **kwargs), ], data)
    #parallel_flow = mdp.parallel.ParallelFlow([ParameterSettingNode(flow, self.loss_function, cross_validate_function, *args, **kwargs), ])

    # Call the parallel execution of the flow. This returns a list of error, 
    results = parallel_flow.execute(data_parallel, scheduler=self.scheduler, execute_callable_class=FlowExecuteCallableNoChecks)

    error_list = results[::2]
    flow_list = results[1::2]

    i = 0
    for paramspace_index_flat, parameter_values in enumerate(self.param_space):
        paramspace_index_full = mdp.numx.unravel_index(paramspace_index_flat, self.paramspace_dimensions)
        self.errors[paramspace_index_full] = error_list[i]
        # If the current error is minimal, store the current flow
        if error_list[i] < min_error:
            min_error = error_list[i]
            self.optimal_flow = flow_list[i]
        i += 1

    self.scheduler.shutdown()

@mdp.extension_method("parallel", Optimizer)
def cma_es (self, data, flow, cross_validate_function, options=None, internal_gridsearch_parameters=None, validate_suffix_flow=None, *args, **kwargs):
    if not hasattr(self, 'scheduler') or self.scheduler is None:
        err = ("No scheduler was assigned to the Optimizer so cannot run in parallel mode.")
        raise Exception(err)

    min_error = float('+infinity')
    self.evaluated_parameters = []
    try:
        import cma
    except:
        print 'CMA-ES not found. cma.py should be in the python path.'
        raise

    if sp.any([sp.squeeze(p).shape != (2,) for p in self.parameter_ranges]):
        raise mdp.FlowException('When using cma_es as optimization algorithm, the parameter ranges should be two-dimensional, the first is assumed the mean and the second the standard devation of the search space.')

    # Initial value is the mean element of each array in the optimization range
    # CMA-ES assumes the optimum lies within x0 +- 3*sigma, so we rescale the parameters
    # passed to and from the cma algorithm

    x0 = [p[0] for p in self.parameter_ranges]
    stds = [p[1] for p in self.parameter_ranges]

    iteration = 0
    # Initialize a CMA instance
    es = cma.CMAEvolutionStrategy(x0, 1, options)

    while not es.stop():
        print 'Iteration ' + str(iteration)

        data_parallel = []

        # Ask the next generation of parameter vectors from the optimization algorithm
        parameter_vectors = es.ask()
        #print parameter_vectors


        params = {}
        for parameter_vector in parameter_vectors:
            for ((parameter_index, node_parameter), std) in zip(enumerate(self.parameters), stds):
                node_index = flow.flow.index(node_parameter[0])
                params[node_index] = {}
                params[node_index][node_parameter[1]] = parameter_vector[parameter_index] * std
            data_parallel.append([[params, data]])

        # Re-initialize all nodes that have the initialize method (e.g. reservoirs nodes)
        for node in flow:
            if hasattr(node, 'initialize'):
                node._is_initialized = False

        parallel_flow = mdp.parallel.ParallelFlow([ParameterSettingNode(flow, self.loss_function, cross_validate_function, *args, **kwargs), ])

        results = parallel_flow.execute(data_parallel, scheduler=self.scheduler, execute_callable_class=FlowExecuteCallableNoChecks)

        error_list = results[::2]
        flow_list = results[1::2]

        es.tell(parameter_vectors, error_list)
        for parameters in mdp.numx.array(parameter_vectors).T:
            self.evaluated_parameters.append(parameters)
        self.errors.extend(error_list)
        if sp.amin(error_list) < min_error:
            min_error = sp.amin(error_list)
            self.optimal_flow = flow_list[sp.argmin(error_list)]
        iteration += 1

    self.scheduler.shutdown()


class FlowExecuteCallableNoChecks(mdp.parallel.FlowExecuteCallable):
    ''' This class is used in the parallel grid search for the Optimizer class, 
        to avoid the pre_execution_checks done by MDP. In the function below, x 
        is a list, which doesn't pass the checks since these assume a numpy array. 
        The code below skips the tests. 
    '''
    def __call__(self, x):
        y = self._flownode._execute(x, nodenr=self._nodenr)
        if self._flownode.use_execute_fork():
            if self._purge_nodes:
                mdp.parallel._purge_flownode(self._flownode)
            return (y, self._flownode)
        else:
            return (y, None)
