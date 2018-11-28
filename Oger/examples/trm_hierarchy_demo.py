# This file demonstrates the use of a hierarchy of CRBMs, PCA and reservoirs on
# a very easy toy problem.

import pylab
import mdp.nodes
import mdp.hinet
import numpy as np
import Oger

# Some recycled functions for data creation.

def generate_data(N):
    """Creates a noisy dataset with some simple pattern in it."""
    T = N * 38
    u = np.mat(np.zeros((T, 20)))
    for i in range(1, T, 38):
        if i % 76 == 1:
            u[i - 1:i + 19, :] = np.eye(20)
            u[i + 18:i + 38, :] = np.eye(20)[np.arange(19, -1, -1)]
            u[i - 1:i + 19, :] += np.eye(20)[np.arange(19, -1, -1)] 
        else:
            u[i - 1:i + 19, 1] = 1
            u[i + 18:i + 38, 8] = 1
    return u


if __name__ == '__main__':

    u = np.array(generate_data(60))
    t = np.zeros(u.shape)
    t[:-1, :] = u[1:, :]

    epochs = 10
    crbm1_size = 100
    crbm2_size = 100

    # Several nodes will be created to create a hierarchy of CRBMs that use context
    # data coming from reservoirs. Moreover, one reservoir receives input that went
    # throught a PCA node as well. This is solved by using layers and identity nodes.

    # First reservoir layer
    reservoir1 = Oger.nodes.ReservoirNode(input_dim=20, output_dim=300)
    shift1 = Oger.nodes.ShiftNode(input_dim=20, n_shifts=1)

    ReservoirLayer1 = mdp.hinet.SameInputLayer([shift1, reservoir1])

    # First CRBM.
    # Note that the output of the InputLayer will be 320 dimensional.
    crbmnode1 = Oger.nodes.CRBMNode(hidden_dim=crbm1_size, visible_dim=20, context_dim=300)

    x = ReservoirLayer1(u)

    print 'Training first CRBM layer...'
    for epoch in range(epochs):
        for i in range(len(x) - 1):
            crbmnode1.train(x[i:i + 1, :], epsilon=.001, decay=.0002)

    theflow = ReservoirLayer1 + crbmnode1

    # PCA layer
    pcanode = mdp.nodes.PCANode(input_dim=crbm1_size, output_dim=40)
    shift2 = Oger.nodes.ShiftNode(input_dim=crbm1_size, n_shifts=1)

    PCALayer = mdp.hinet.SameInputLayer([shift2, pcanode])

    x = crbmnode1(x)
    PCALayer.train(x)
    PCALayer.stop_training()
    theflow += PCALayer

    # Second reservoir layer.
    # Note that this time a normal layer is used to split the PCA and raw data.
    x = PCALayer(x)
    reservoir2 = Oger.nodes.ReservoirNode(input_dim=40, output_dim=300)
    identity2 = mdp.nodes.IdentityNode(input_dim=crbm1_size)

    ReservoirLayer2 = mdp.hinet.Layer([identity2, reservoir2])

    theflow += ReservoirLayer2

    # Second CRBM.
    x = ReservoirLayer2(x)

    crbmnode2 = Oger.nodes.CRBMNode(hidden_dim=crbm2_size, visible_dim=crbm1_size, context_dim=300)

    print 'Training second CRBM layer...'
    for epoch in range(epochs):
        for i in range(len(x) - 1):
            crbmnode2.train(x[i:i + 1, :], epsilon=.001, decay=.0002)

    theflow += crbmnode2
    x = crbmnode2(x)

    # And finally a linear classifier to put on top.
    readout = Oger.nodes.RidgeRegressionNode(ridge_param=.0000001,
                                                      input_dim=crbm2_size, output_dim=20)

    readout.train(x, t)
    readout.stop_training()

    theflow += readout
    theflow = mdp.hinet.FlowNode(theflow)

    y = theflow(u)

    error = np.mean((t.ravel() - y.ravel())**2)
    print 'Final MSE:', error

    pylab.clf()
    p = pylab.subplot(211)
    p = pylab.imshow(u[:500, :].T)
    p = pylab.subplot(212)
    p = pylab.imshow(y[:500, :].T)
    pylab.show()

