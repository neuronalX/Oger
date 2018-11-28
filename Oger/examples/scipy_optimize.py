import Oger
import scipy as sp
import mdp

if __name__ == '__main__':
    ''' Example of doing a grid-search
        Runs the NARMA 30 task for input scaling values = 0.1 to 1 with 0.2 stepsize and spectral radius = 0.1 to 1.5 with stepsize 0.3
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
    gridsearch_parameters = {reservoir:{'input_scaling': mdp.numx.array([0.3]), 'spectral_radius':mdp.numx.array([.9])}}

    # Instantiate an optimizer
    opt = Oger.evaluation.Optimizer(gridsearch_parameters, Oger.utils.nrmse)

    opt.scipy_optimize(data, flow, cross_validate_function=Oger.evaluation.n_fold_random, opt_func=sp.optimize.fmin_powell, n_folds=5, options={'maxiter':20, })

    # Get the optimal flow and run cross-validation with it 
    opt_flow = opt.get_optimal_flow()

    print 'Performing cross-validation with the optimal flow.'
    errors = Oger.evaluation.validate(data, opt_flow, Oger.utils.nrmse, cross_validate_function=Oger.evaluation.n_fold_random, n_folds=5, progress=False)
    print 'Mean error over folds: ' + str(sp.mean(errors))

