import Oger
import pylab
from mdp import numx

def switching_signals(f1, f2, T, n_switches, n_samples=1):
    samples = []
    # seconds per simulation timestep
    t = numx.arange(T)
    proto_1 = numx.atleast_2d(numx.sin(2 * numx.pi * t * f1)).T
    proto_2 = numx.atleast_2d(numx.sin(2 * numx.pi * t * f2)).T

    for _ in range(n_samples):
        n_periods1 = numx.random.randint(4, 8, size=(n_switches))
        n_periods2 = numx.random.randint(4, 8, size=(n_switches))

        #n_periods1, n_periods2 = [1], [0]

        switch = []
        signal = []
        for p1, p2 in zip(n_periods1, n_periods2):
            switch.extend([numx.ones_like(proto_1)] * p1)
            switch.extend([-1 * numx.ones_like(proto_2)] * p2)
            signal.extend([proto_1] * p1)
            signal.extend([proto_2] * p2)

        samples.append([numx.concatenate((numx.concatenate(switch), numx.concatenate(signal)), 1)])
    return samples

if __name__ == "__main__":

    n_train_samples = 3

    # Frequencies of different templates
    f1 = .01
    f2 = .02

    # Number of timesteps in one signal template
    T = 200

    # Number of switches between templates in one chunk
    n_switches = 8

    # Number of reservoir neurons
    N = 100
    freerun_steps = 6000

    train_signals = switching_signals(f1, f2, T, n_switches, n_samples=n_train_samples)
    test_signals = switching_signals(f1, f2, T, n_switches, n_samples=1)

    # create a reservoir node and enable leak rate
    reservoir = Oger.nodes.LeakyReservoirNode(output_dim=N, input_scaling=.1)
    reservoir.leak_rate = .1

    # create a ridge regression node and enable washout during training (disregarding the first timesteps)
    readout = Oger.nodes.RidgeRegressionNode()
    Oger.utils.enable_washout(Oger.nodes.RidgeRegressionNode, 100)

    # Create the freerun flow, set the first dimension of the input to be external (i.e. not self-generated or fed-back)
    flow = Oger.nodes.FreerunFlow([reservoir, readout], external_input_range=[0], freerun_steps=freerun_steps)

    reservoir.reset_states = False

    # optimize the ridge parameter
    gridsearch_parameters = {readout:{'ridge_param': 10 ** numx.arange(-6, 0, .5)}}

    # Instantiate an optimizer
    loss_function = Oger.utils.timeslice(range(-freerun_steps, 0), Oger.utils.nrmse)
    opt = Oger.evaluation.Optimizer(gridsearch_parameters, loss_function)

    # Do the grid search
    opt.grid_search([[], train_signals], flow,
                    cross_validate_function=Oger.evaluation.leave_one_out)

    # get the optimal flow
    opt_flow = opt.get_optimal_flow(verbose=True)

    # Train the optimized flow to do one-step ahead prediction using the teacher-forced signal
    # An additional input giving the frequency of the desired sine wave is also given   
    opt_flow.train([[] , train_signals])

    # execute the trained flow on a test signal
    freerun_output = opt_flow.execute(test_signals[0][0])

    # plot the test results
    pylab.plot(test_signals[0][0][:, 1], 'b')
    pylab.plot(freerun_output, 'g')
    pylab.axvline(test_signals[0][0].shape[0] - freerun_steps + 1, pylab.ylim()[0], pylab.ylim()[1], color='r')

    pylab.show()
