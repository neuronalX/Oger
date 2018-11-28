import mdp.nodes
import Oger

if __name__ == '__main__':
    inputs, outputs = Oger.datasets.memtest(n_samples=2, sample_len=10000, n_delays=100)

    data = [[], zip(inputs, outputs)]

    # construct individual nodes
    reservoir = Oger.nodes.ReservoirNode(1, 200)
    readout = mdp.nodes.LinearRegressionNode()

    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout])

    # Nested dictionary
    gridsearch_parameters = {reservoir:{'input_scaling': mdp.numx.arange(0.1, 1, .2), 'spectral_radius':mdp.numx.arange(0.1, 1.5, .3)}}

    # Instantiate an optimizer
    opt = Oger.evaluation.Optimizer(gridsearch_parameters, Oger.utils.mem_capacity)
    
    # Do the grid search
    opt.grid_search(data, flow, cross_validate_function=Oger.evaluation.train_test_only, training_fraction=0.5, random=False)

    # Get the minimal error
    min_error, parameters = opt.get_minimal_error()
    
    print 'The minimal error is ' + str(min_error)
    print 'The corresponding parameter values are: '
    output_str = '' 
    for node in parameters.keys():
        for node_param in parameters[node].keys():
            output_str += str(node) + '.' + node_param + ' : ' + str(parameters[node][node_param]) + '\n'
    print output_str

    opt.plot_results()
