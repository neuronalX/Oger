import Oger
import mdp.parallel
import time

if __name__ == '__main__':
    ''' Example of doing a grid-search
        Runs the NARMA 30 task for bias values = 0 to 2 with 0.5 stepsize and spectral radius = 0.1 to 1 with stepsize 0.5
    '''
    input_size = 1
    inputs, outputs = Oger.datasets.narma30()

    data = [[], zip(inputs, outputs)]

    # construct individual nodes
    reservoir = Oger.nodes.ReservoirNode(input_size, 200)
    readout = Oger.nodes.RidgeRegressionNode()

    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout])

    # Nested dictionary
    gridsearch_parameters = {reservoir:{'spectral_radius':mdp.numx.arange(0.1, 1.5, 0.5), 'input_scaling': mdp.numx.arange(0.1, 1, 0.2)}}

    # Instantiate an optimizer
    opt = Oger.evaluation.Optimizer(gridsearch_parameters, Oger.utils.nrmse)


    print 'Sequential execution...'
    start_time = time.time()
    opt.grid_search(data, flow, cross_validate_function=Oger.evaluation.n_fold_random, n_folds=5)
    seq_duration = int(time.time() - start_time)
    print 'Duration: ' + str(seq_duration) + 's'

    # Do the grid search
    print 'Parallel execution...'
    opt.scheduler = mdp.parallel.ProcessScheduler(n_processes=2)
    mdp.activate_extension("parallel")
    start_time = time.time()
    opt.grid_search(data, flow, cross_validate_function=Oger.evaluation.n_fold_random, n_folds=5)
    par_duration = int(time.time() - start_time)
    print 'Duration: ' + str(par_duration) + 's'

    print 'Speed up factor: ' + str(float(seq_duration) / par_duration)

    # Get the minimal error
    min_error, parameters = opt.get_minimal_error()

    opt.plot_results()
    print 'The minimal error is ' + str(min_error)
    print 'The corresponding parameter values are: '
    output_str = ''
    for node in parameters.keys():
        for node_param in parameters[node].keys():
            output_str += str(node) + '.' + node_param + ' : ' + str(parameters[node][node_param]) + '\n'
    print output_str
