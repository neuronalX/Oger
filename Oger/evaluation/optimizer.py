import Oger
import mdp.utils
import itertools
import scipy.stats
import scipy.optimize
import scipy as sp
from copy import deepcopy


class Optimizer(object):
    ''' Class to perform optimization of the parameters of a flow using cross-validation.
        Supports grid-searching of a parameter space.
    '''
    def __init__(self, optimization_dict, loss_function):
        ''' Construct an Optimizer object.
            optimization_dict: a dictionary of dictionaries. 
            In the top-level dictionary, every key is a node, and the corresponding item is again a dictionary 
            where the key is the parameter name to be optimize and the corresponding item is a list of values for the parameter
            Example: to gridsearch the spectral radius and input scaling of reservoirnode1, and the input scaling of reservoirnode2 over the range .1:.1:1 this would be:
            opt_dict = {reservoirnode1: {'spec_radius': np.arange(.1,1,.1), 'input_scaling': np.arange(.1,1,.1)}, reservoirnode2: {'input_scaling': np.arange(.1,1,.1)}}
            loss_function: the function used to compute the loss
        '''
        # Initialize attributes
        self.optimization_dict = optimization_dict
        self.loss_function = loss_function
        self.parameter_ranges = []
        self.parameters = []
        self.errors = []
        self.optimal_flow = []
        self.probe_data = None

        # Construct the parameter space
        # Loop over all nodes that need their parameters set
        for node_key in self.optimization_dict.keys():
            # Loop over all parameters that need to be set for that node
            for parameter in self.optimization_dict[node_key].keys():
                # Append the parameter name and ranges to the corresponding lists
                self.parameter_ranges.append((self.optimization_dict[node_key])[parameter])
                self.parameters.append((node_key, parameter))


    def grid_search (self, data, flow, cross_validate_function, progress=True, validation_suffix_flow=None, internal_gridsearch_parameters = None, *args, **kwargs):
        ''' Do a combinatorial grid-search of the given parameters and given parameter ranges, and do cross-validation of the flowNode
            for each value in the parameter space.
            Input arguments are:
                - data: a list of iterators which would be passed to MDP flows
                - flow : the MDP flow to do the grid-search on
                - cross-validate_function: the function to use for cross-validation
                - progressinfo (default: True): show a progress bar
                - validation_suffix_flow: append this flow to the given flow during validation (e.g. for adding noise during freerun)
            If any of the nodes in the flow have a member variable probe_data, the contents of this variable are also stored for each parameter
            point in Optimizer.probe_data. This member variable is an N-dimensional list, with N the number of parameters being ranged over. 
            Each element of this N-d list is a dictionary, indexed by the nodes in the flow, whose values are the corresponding contents 
            of the probe_data.
        '''
        # Get the number of parameter points to be evaluated for each of the parameters
        paramspace_dimensions = [len(r) for r in self.parameter_ranges]

        self.errors = mdp.numx.zeros(paramspace_dimensions)
        self.probe_data = {}
        errors_per_fold = {}
        min_error = float('+infinity')

        # Construct all combinations
        param_space = list(itertools.product(*self.parameter_ranges))

        if progress:
            iteration = mdp.utils.progressinfo(enumerate(param_space), style='timer', length=len(param_space))
        else:
            iteration = enumerate(param_space)

        # Loop over all points in the parameter space
        for paramspace_index_flat, parameter_values in iteration:
            # Set all parameters of all nodes to the correct values
            node_set = set()
            for parameter_index, node_parameter in enumerate(self.parameters):
                # Add the current node to the set of nodes whose parameters are changed, and which should be re-initialized
                node_set.add(node_parameter[0])
                node_parameter[0].__setattr__(node_parameter[1], parameter_values[parameter_index])

            # Re-initialize all nodes that have the initialize method (e.g. reservoirs nodes)
            for node in flow:
                if hasattr(node, 'initialize'):
                    node._is_initialized = False

            # After all node parameters have been set and initialized, do the cross-validation
            paramspace_index_full = mdp.numx.unravel_index(paramspace_index_flat, paramspace_dimensions)

            # Keep the untrained flow for later
            current_flow = deepcopy(flow)

            # Do the validation
            validation_result = Oger.evaluation.validate(data, flow,
                                                         self.loss_function, cross_validate_function,
                                                         progress=False, internal_gridsearch_parameters=internal_gridsearch_parameters,
                                                         validation_suffix_flow=validation_suffix_flow,
                                                         *args, **kwargs)

            if internal_gridsearch_parameters is not None:
                validation_errors, current_flow = validation_result
            else:
                validation_errors = validation_result

            mean_validation_error = mdp.numx.mean(validation_errors)

            # If the current error is minimal, store the current flow
            if mean_validation_error < min_error:
                min_error = mean_validation_error
                self.optimal_flow = current_flow

            # Store the current error in the errors array
            self.errors[paramspace_index_full] = mean_validation_error
            errors_per_fold[paramspace_index_full] = validation_errors

            # Collect probe data if it is present
            for node in flow:
                if hasattr(node, 'probe_data'):
                    # If the key exists, append to it, otherwise insert an empty list and append to that
                    self.probe_data.setdefault(paramspace_index_full, {})[node] = node.probe_data

    def cma_es (self, data, flow, cross_validate_function, options=None, *args, **kwargs):
        ''' Perform optimization of the parameters given at construction time using CMA-ES (Covariance Matrix Adaptation - Evolution Strategy).
            The file cma.py (available from http://www.lri.fr/~hansen/cmaes_inmatlab.html#python) needs to be in the python path.
            Each parameter should have a starting value x0 and a standard deviation of the search space. This should be given as a 2d array in gridsearch_parameters,
            e.g. {reservoir: {'spectral_radius':scipy.array([1,.5])}} for searching around 1 with a std. dev. of 1/2.
            Input arguments are:
                - data: a list of iterators which would be passed to MDP flows
                - flow : the MDP flow to do the grid-search on
                - cross-validate_function: the function to use for cross-validation
                - options: an Options instance as defined by CMA-ES, useful e.g. for defining bounds on the parameters. See the documentation for CMA-ES for more information.
            If any of the nodes in the flow have a member variable probe_data, the contents of this variable are also stored for each parameter
            point in Optimizer.probe_data. This member variable is an N-dimensional list, with N the number of parameters being ranged over. 
            Each element of this N-d list is a dictionary, indexed by the nodes in the flow, whose values are the corresponding contents 
            of the probe_data.
        '''
        min_error = float('+infinity')
        evaluated_parameters = []
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
            # Ask the next generation of parameter vectors from the optimization algorithm
            parameter_vectors = es.ask()
            #print parameter_vectors

            # Empty list to contain the errors per parameter vector
            error_list = []

            for parameter_vector in parameter_vectors:
                node_set = set()
                for ((parameter_index, node_parameter), std) in zip(enumerate(self.parameters), stds):
                    # Add the current node to the set of nodes whose parameters are changed, and which should be re-initialized
                    node_set.add(node_parameter[0])
                    node_parameter[0].__setattr__(node_parameter[1], parameter_vector[parameter_index] * std)

                # Re-initialize all nodes that have the initialize method (e.g. reservoirs nodes)
                for node in node_set:
                    if hasattr(node, 'initialize'):
                        node.initialize()

                current_flow = deepcopy(flow)
                validation_errors = Oger.evaluation.validate(data, flow, self.loss_function, cross_validate_function, progress=False, *args, **kwargs)
                mean_validation_error = mdp.numx.mean(validation_errors)

                # If the current error is minimal, store the current flow
                if mean_validation_error < min_error:
                    min_error = mean_validation_error
                    self.optimal_flow = current_flow
                #print mean_validation_error
                error_list.append(mean_validation_error)
            es.tell(parameter_vectors, error_list)
            for parameters in mdp.numx.array(parameter_vectors).T:
                evaluated_parameters.append(parameters)
            self.errors.extend(error_list)
            self.parameter_ranges
            iteration += 1

    def plot_results(self, node_param_list=None, vmin=None, vmax=None, cmap=None, log_x=False, axes=None, title=None, plot_variance=True):
        ''' Plot the results of the optimization. 
            
            Works for 1D and 2D linear sweeps, yielding a 2D resp. 3D plot of the parameter(s) vs. the error.
            Arguments:
                - node_param_list: a list of (node, param_string) tuples. Before plotting, the mean will be taken over all these node.param_string combinations, which is useful to plot/reduce multi-dimensional parameter sweeps.
                - vmin/vmax: can be used to truncate the errors between lower and upper bounds before plotting.
                - cmap: passed as a matplotlib colormap when plotting 2D images.
                - log_x: boolean to indicate if a 1D plot should use a log scale for the x-axis.
                - axes: optional Axes object to use for plotting
                - title: optional title for the plot
                - plot_variance: should variance be plotted in case of taking the mean over certain parameters. Default True. 
        '''

        try:
            import pylab
        except ImportError:
            print "It looks like matplotlib isn't installed. Plotting is impossible."
            return

        if axes is None:
            axes = pylab.axes()

        errors_to_plot, var_errors, parameters = self.mean_and_var(node_param_list)
        if vmin != None:
            errors_to_plot[errors_to_plot < vmin] = vmin
        if vmax != None:
            errors_to_plot[errors_to_plot > vmax] = vmax

        # If we have ranged over only one parameter
        if len(parameters) == 1:
            # Get the index of the remaining parameter to plot using the correct 
            # parameter ranges
            param_index = self.parameters.index(parameters[0])
            if var_errors is not None and plot_variance:
                pylab.errorbar(self.parameter_ranges[param_index], errors_to_plot, var_errors, axes=axes)
            else:
                if log_x:
                    pylab.semilogx(self.parameter_ranges[param_index], errors_to_plot, axes=axes)
                else:
                    pylab.plot(self.parameter_ranges[param_index], errors_to_plot, axes=axes)

            pylab.xlabel(str(parameters[0][0]) + '.' + parameters[0][1])
            pylab.ylabel(self.loss_function.__name__)
            if title is not None:
                pylab.title(title)
            pylab.show()
        elif len(parameters) == 2:
            # Get the extreme values of the parameter values
            p1 = self.parameters.index(parameters[0])
            p2 = self.parameters.index(parameters[1])

            xm = mdp.numx.amin(self.parameter_ranges[p1])
            ym = mdp.numx.amin(self.parameter_ranges[p2])

            xM = mdp.numx.amax(self.parameter_ranges[p1])
            yM = mdp.numx.amax(self.parameter_ranges[p2])

            # For optimization algorithms which have non-uniform sampling of the parameter space, we interpolate here
            # This has no effect on the plot for optimizations using gridsearch
            xi = mdp.numx.linspace(xm, xM, len(self.parameter_ranges[p1]))
            yi = mdp.numx.linspace(ym, yM, len(self.parameter_ranges[p2]))
            (x, y) = mdp.numx.meshgrid(self.parameter_ranges[p1], self.parameter_ranges[p2])

            # Create an interpolation grid
            zi = mdp.numx.fliplr(pylab.griddata(x.flatten(), y.flatten(), errors_to_plot.flatten('F'), xi, yi)).T

            pylab.imshow(zi, cmap=pylab.jet(), interpolation='nearest',
             extent=self.get_extent(parameters), aspect="auto", axes=axes)
            pylab.xlabel(str(parameters[1][0]) + '.' + parameters[1][1])
            pylab.ylabel(str(parameters[0][0]) + '.' + parameters[0][1])
            if title is not None:
                pylab.suptitle(title)
            pylab.colorbar()

            if var_errors is not None and plot_variance:
                pylab.figure()
                pylab.imshow(mdp.numx.flipud(var_errors), cmap=cmap, interpolation='nearest',
                             extent=self.get_extent(parameters), aspect="auto", vmin=vmin, vmax=vmax)
                pylab.xlabel(str(parameters[1][0]) + '.' + parameters[1][1])
                pylab.ylabel(str(parameters[0][0]) + '.' + parameters[0][1])
                pylab.suptitle('variance')
                pylab.colorbar()

            pylab.show()
        else:
            raise Exception("Too many parameter dimensions to plot: " + str(errors_to_plot.ndim))

    def mean_and_var(self, node_param_list=None):
        ''' Return a tuple containing the mean and variance of the errors over a certain parameter.
            
            Gives the mean/variance of the errors w.r.t. the parameter given by 
            node_param_list, where each element is a (node, parameter) tuple.
            If the list has only one element, the variance w.r.t this parameter
            is also returned, otherwise the second return argument is None.
        '''
        # In case of an empty list, we just return the errors
        if node_param_list is None:
            return mdp.numx.array(self.errors), None, self.parameters

        # Check if we have the requested node.parameter combinations in the optimization_dict
        for node, param_string in node_param_list:
            if not node in self.optimization_dict:
                raise Exception('Cannot take the mean, given node ' + str(node) + ' is not in optimization_dict.')
            if not param_string in self.optimization_dict[node]:
                raise Exception("Cannot take the mean, given parameter '" + param_string + "' is not in optimization_dict.")

        # take a copy, so we can eliminate some of the parameters later
        # However, we don't want to do a deep copy, just create a new list 
        # with references to the old elements, hence the [:]
        parameters = self.parameters[:]
        errors = self.errors[:]

        # Loop over all parameters in node_param_list and iteratively compute 
        # the mean
        for node, param_string in node_param_list:
            # Find the axis we need to take the mean across
            axis = parameters.index((node, param_string))

            # Compute the mean
            mean_errors = scipy.stats.nanmean(errors, axis)

            # In case we take the mean over only one dimension, we can return
            # the variance as well
            if len(node_param_list) == 1:
                # Use ddof = 1 to mimic matlab var
                var = scipy.stats.nanstd(errors, axis, bias=True) ** 2
            else:
                var = None

            # Remove the corresponding dimension from errors and parameters for
            # the next iteration of the for loop
            errors = mean_errors
            parameters.remove((node, param_string))

        return errors, var, parameters


    def get_minimal_error(self, node_param_list=None):
        '''Return the minimal error, the corresponding parameter values as a tuple:
        (error, param_values), where param_values is a dictionary of dictionaries,  
        with the key of the outer dictionary being the node, and inner dictionary
        consisting of (parameter:optimal_value) pairs.
        If the optional argument node_param_list is given, first the mean of the
        error will be taken over all (node, parameter) tuples in the node_param_list
        before taking the minimum
        '''
        if self.errors is None:
            raise Exception('Errors array is empty. No optimization has been performed yet.')

        errors, _, parameters = self.mean_and_var(node_param_list)
        min_parameter_dict = {}
        minimal_error = mdp.numx.amin(errors)
        min_parameter_indices = mdp.numx.unravel_index(mdp.numx.argmin(errors), errors.shape)

        for index, param_d in enumerate(parameters):
            global_param_index = self.parameters.index(param_d)
            opt_parameter_value = self.parameter_ranges[global_param_index][min_parameter_indices[index]]
            # If there already is an entry for the current node in the dict, add the 
            # optimal parameter/value entry to the existing dict
            if param_d[0] in min_parameter_dict:
                min_parameter_dict[param_d[0]][param_d[1]] = opt_parameter_value
            # Otherwise, create a new dict
            else:
                min_parameter_dict[param_d[0]] = {param_d[1] : opt_parameter_value}

        return (minimal_error, min_parameter_dict)

    def get_optimal_flow(self, verbose=False):
        ''' Return the optimal flow obtained by the optimization algorithm.
            If verbose=True, this will print the optimized parameters and their corresponding values.
        '''
        if verbose:
            # Get the minimal error
            min_error, parameters = self.get_minimal_error()
            print 'The minimal error is ' + str(min_error)
            print 'The corresponding parameter values are: '
            output_str = ''
            for node in parameters.keys():
                for node_param in parameters[node].keys():
                    output_str += str(node) + '.' + node_param + ' : ' + str(parameters[node][node_param]) + '\n'
            print output_str

        return self.optimal_flow


    def get_extent(self, parameters):
        '''Compute the correct boundaries of the parameter ranges for
            a 2D plot
        '''
        param_index0 = self.parameters.index(parameters[1])
        param_index1 = self.parameters.index(parameters[0])

        extent = [self.parameter_ranges[param_index0][0],
                    self.parameter_ranges[param_index0][-1],
                    self.parameter_ranges[param_index1][0],
                    self.parameter_ranges[param_index1][-1]]

        # Fix the range bounds
        xstep = (-extent[0] + extent[1]) / len(self.parameter_ranges[param_index0])
        ystep = (-extent[2] + extent[3]) / len(self.parameter_ranges[param_index1])

        return [extent[0] - xstep / 2, extent[1] + xstep / 2, extent[2] - ystep / 2, extent[3] + ystep / 2]

    def save(self, fname):
        ''' Save the current optimizer using pickle.
        '''
        import pickle
        fhandle = open(fname, 'w')
        pickle.dump(self, fhandle)
        fhandle.close()

    def scipy_optimize (self, data, flow, cross_validate_function, opt_func=scipy.optimize.fmin, options=None, *args, **kwargs):
        '''Optimize the given flow on the data using cross_validate_function,
            with any of the scipy optimizers (default: scipy.optimize.fmin).
            Input arguments are:
                - data: a list of iterators which would be passed to MDP flows
                - flow : the MDP flow to do the grid-search on
                - opt_func: the scipy optimize function to use for the
                  optimization
                - cross-validate_function: the function to use for cross-validation
                - validation_suffix_flow: append this flow to the given flow during validation (e.g. for adding noise during freerun)
            If any of the nodes in the flow have a member variable probe_data, the contents of this variable are also stored for each parameter
            point in Optimizer.probe_data. This member variable is an N-dimensional list, with N the number of parameters being ranged over.
            Each element of this N-d list is a dictionary, indexed by the nodes in the flow, whose values are the corresponding contents
            of the probe_data.
        '''

        def _f_opt(parameter_values, data, flow, cross_validate_function):
            node_set = set()
            for (parameter_index, node_parameter) in enumerate(self.parameters):
                # Add the current node to the set of nodes whose parameters are changed, and which should be re-initialized
                node_set.add(node_parameter[0])
                node_parameter[0].__setattr__(node_parameter[1], parameter_values[parameter_index])

            # Re-initialize all nodes that have the initialize method (e.g. reservoirs nodes)
            for node in node_set:
                if hasattr(node, 'initialize'):
                    node.initialize()

            validation_errors = Oger.evaluation.validate(data, flow, self.loss_function, cross_validate_function, progress=False, *args, **kwargs)
            return mdp.numx.mean(validation_errors)

        evaluated_parameters = []

        x0 = [p[0] for p in self.parameter_ranges]

        if options is None:
            optimization_results = opt_func(_f_opt, x0, (data, flow, cross_validate_function))
        else:
            optimization_results = opt_func(_f_opt, x0, (data, flow, cross_validate_function), full_output=True, retall=True, **options)

        xopt = optimization_results[0]
        self.errors = optimization_results[1]
        evaluated_parameters = optimization_results[-1]

        # Set the parameters of the optimal flow
        for (parameter_index, node_parameter) in enumerate(self.parameters):
            node_parameter[0].__setattr__(node_parameter[1], xopt[parameter_index])

        self.optimal_flow = flow
