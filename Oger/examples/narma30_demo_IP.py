import Oger
import pylab
import mdp
import numpy as np

#This example demonstrates a very simple reservoir+readout setup on the 30th order NARMA task, while pretraining the reservoir with IP.

def run_IP(sr, data):
    [x, _] = data

    # construct individual nodes
    reservoir = Oger.nodes.GaussianIPReservoirNode(input_dim=1, output_dim=100, w_in=np.random.randn, spectral_radius=sr, eta=.001, sigma_squared=.04)
    
    states_before = reservoir.execute(x[0])

    # Initialize the reservoir weights so we can compute the spectral radius
    reservoir.initialize()
    print 'Spectral radius before adaptation: ', Oger.utils.get_spectral_radius(reservoir.w)

    # build network with MDP framework
    flow = mdp.Flow([reservoir])
    Oger.utils.make_inspectable(Oger.nodes.GaussianIPReservoirNode)

    data = [x]

    # train the flow 
    flow.train(data)

    print 'Spectral radius after adaptation: ', Oger.utils.get_spectral_radius(reservoir.w)

    states_after = reservoir.execute(x[0])

    nx = 4
    ny = 1

    n = 500

    pylab.figure()
    pylab.subplot(nx, ny, 1)
    pylab.plot(np.vstack(reservoir.aa)[:n, :])
    pylab.title('Evolution of a')

    pylab.subplot(nx, ny, 2)
    pylab.plot(np.vstack(reservoir.bb)[:n, :])
    pylab.title('Evolution of b')

    pylab.subplot(nx, ny, 3)
    pylab.plot(states_before)

    pylab.subplot(nx, ny, 4)
    pylab.plot(states_after)

if __name__ == '__main__':
    # Get the dataset
    data = Oger.datasets.narma30()

    run_IP(.85, data)
    run_IP(1.5, data)


