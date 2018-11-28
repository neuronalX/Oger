import Oger
import pylab
import scipy

if __name__ == "__main__":

    freerun_steps = 1000
    training_sample_length = 5000
    n_training_samples = 3
    test_sample_length = 5000

    train_signals = Oger.datasets.mackey_glass(sample_len=training_sample_length, n_samples=n_training_samples)
    test_signals = Oger.datasets.mackey_glass(sample_len=test_sample_length, n_samples=1)

    reservoir = Oger.nodes.LeakyReservoirNode(output_dim=400, leak_rate=0.4, input_scaling=.05, bias_scaling=.2, reset_states=False)
    readout = Oger.nodes.RidgeRegressionNode()
    Oger.utils.enable_washout(Oger.nodes.RidgeRegressionNode, 500)

    flow = Oger.nodes.FreerunFlow([reservoir, readout], freerun_steps=freerun_steps)

    gridsearch_parameters = {readout:{'ridge_param': 10 ** scipy.arange(-4, 0, .3)}}

    # Instantiate an optimizer
    loss_function = Oger.utils.timeslice(range(training_sample_length - freerun_steps, training_sample_length), Oger.utils.nrmse)
    opt = Oger.evaluation.Optimizer(gridsearch_parameters, loss_function)

    # Do the grid search
    opt.grid_search([[], train_signals], flow, cross_validate_function=Oger.evaluation.leave_one_out)

    # Get the optimal flow and run cross-validation with it 
    opt_flow = opt.get_optimal_flow(verbose=True)

    print 'Freerun on test_signals signal with the optimal flow...'
    opt_flow.train([[], train_signals])
    freerun_output = opt_flow.execute(test_signals[0][0])

    pylab.plot(scipy.concatenate((test_signals[0][0][-2 * freerun_steps:])))
    pylab.plot(scipy.concatenate((freerun_output[-2 * freerun_steps:])))
    pylab.xlabel('Timestep')
    pylab.legend(['Target signal', 'Predicted signal'])
    pylab.axvline(pylab.xlim()[1] - freerun_steps + 1, pylab.ylim()[0], pylab.ylim()[1], color='r')
    print opt_flow[1].ridge_param
    pylab.show()
