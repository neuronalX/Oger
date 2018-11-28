import Oger
import scipy as sp
import time
import mdp.parallel

if __name__ == '__main__':
    ''' Example of using CMA_ES to optimize the parameters of a reservoir+readout on the NRMSE for NARMA30, once sequentially and once in parallel if the machine is multicore.
    The CMA-ES is given an initial value x0 and standard devation for each of the parameters.
    '''
    input_size = 1
    inputs, outputs = Oger.datasets.narma30()

    data = [[], zip(inputs, outputs)]

    # construct individual nodes
    reservoir = Oger.nodes.ReservoirNode(input_size, 100)
    readout = Oger.nodes.RidgeRegressionNode()

    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout])

    # Nested dictionary
    # For cma_es, each parameter 'range' consists of an initial value and a standard deviation
    # For input_scaling, x0=.3 and std = .5
    # For spectral_radius, x0 = .9 and std = .5
    gridsearch_parameters = {reservoir:{'input_scaling': mdp.numx.array([0.3, .5]), 'spectral_radius':mdp.numx.array([.9, .5])}}

    # Instantiate an optimizer
    opt = Oger.evaluation.Optimizer(gridsearch_parameters, Oger.utils.nrmse)

#    # Additional options to be passed to the CMA-ES algorithm. We impose a lower bound on the input_scaling such that values of zero
#    # do not occur (this causes an error in the training of the readout because the reservoir output is all zeros).
    options = {'maxiter':20, 'bounds':[0.01, None]}

    # Do the optimization
    print 'Parallel execution...'
    # Instantiate a new optimizer, otherwise CMA_ES doesn't 
    opt = Oger.evaluation.Optimizer(gridsearch_parameters, Oger.utils.nrmse)
    opt.scheduler = mdp.parallel.ProcessScheduler(n_processes=2)
    #opt.scheduler = Oger.parallel.GridScheduler()
    mdp.activate_extension("parallel")
    start_time = time.time()
    opt.cma_es(data, flow, cross_validate_function=Oger.evaluation.n_fold_random, n_folds=5, options=options)
    par_duration = int(time.time() - start_time)
    print 'Duration: ' + str(par_duration) + 's'

    # Get the optimal flow and run cross-validation with it 
    opt_flow = opt.get_optimal_flow()

    print 'Performing cross-validation with the optimal flow. Note that this result can differ slightly from the one above because of different choices of randomization of the folds.'

    errors = Oger.evaluation.validate(data, opt_flow, Oger.utils.nrmse, cross_validate_function=Oger.evaluation.n_fold_random, n_folds=5, progress=False)
    print 'Mean error over folds: ' + str(sp.mean(errors))
