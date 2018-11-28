import Oger
import pylab
import mdp.parallel

if __name__ == "__main__":
    """ This example shows parallelization of the samples in a dataset. In this case parallellization is done using
        threads, but it can also be done over a computer cluster by providing a different scheduler. See the website for more information.
    """
    inputs = 1
    timesteps = 10000
    washout = 30

    nx = 3
    ny = 1

    [x, y] = Oger.datasets.narma30(sample_len=1000)

    # construct individual nodes
    reservoir = Oger.nodes.ReservoirNode(inputs, 100, input_scaling=0.05)
    readout = Oger.nodes.RidgeRegressionNode()

    # build network with MDP framework
    flow = mdp.parallel.ParallelFlow([reservoir, readout], verbose=1)

    #scheduler = mdp.parallel.ThreadScheduler(n_threads=2, verbose=True)
    scheduler = mdp.parallel.ProcessScheduler(n_processes=2, verbose=True)
#    scheduler = mdp.parallel.pp_support.LocalPPScheduler(ncpus=2, max_queue_length=0, verbose=True)

    data = [[], zip(x[0:-1], y[0:-1])]

    # train the flow 
    flow.train(data, scheduler)

    scheduler.shutdown()

    #apply the trained flow to the training data and test data
    outputs = flow.execute(x)

    print "NRMSE: " + str(Oger.utils.nrmse(y[9], outputs[-1000:, :]))

    #plot the input
    pylab.subplot(nx, ny, 1)
    pylab.plot(x[0])

    #plot everything
    pylab.subplot(nx, ny, 2)
    pylab.plot(outputs[1:1000], 'r')
    pylab.plot(y[0], 'b')

    pylab.subplot(nx, ny, 3)
    pylab.plot(outputs[-1000:, :], 'r')
    pylab.plot(y[9], 'b')
    pylab.show()

