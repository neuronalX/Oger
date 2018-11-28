# This file demonstrates the use of the CRBMNode (on a very trivial task...).

import pylab
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

def get_context(u, N=4):
    T, D = u.shape
    x = np.zeros((T, D * N))
    for i in range(N - 1, T):
        dat = u[i - 1, :]
        for j in range(2, N + 1):
            dat = np.concatenate((dat, u[i - j, :]), 1)
        x[i, :] = dat
    return x

if __name__ == '__main__':

    u = np.array(generate_data(60))

    # Size of the context.
    N = 12

    epochs = 10

    x = np.array(get_context(u, N))
    x += np.random.normal(0, .001, x.shape)

    # The context is concatenated to the input as if it where one signal.
    v = np.concatenate((u, x), 1)

    crbmnode = Oger.nodes.CRBMNode(hidden_dim=100, visible_dim=20, context_dim=N * 20, gaussian=True)

    t, d = u.shape
    print 'Training for %d epochs...' % epochs
    for i in range(epochs):
        print 'Epoch', i
        # Update every two time steps as batch gradient descent barely converges.
        for j in range(t - 1):
            crbmnode.train(v[j:j + 1, :], n_updates=1, epsilon=.0001, momentum=.9, decay=.0002)
        # Check the energy and error after each training epoch.
        # Sampling causes the training phase to end so a copy is made first.
        crbmnode_test = crbmnode.copy()
        hiddens, sampl_h = crbmnode_test.sample_h(u, x)
        energy = crbmnode_test.energy(u, sampl_h, x)
        print 'Energy:', sum(energy)
        visibles, sampl_v = crbmnode_test.sample_v(sampl_h, x)
        error = np.mean((u.ravel() - visibles.ravel())**2)
        print 'MSE:', error

    crbmnode.stop_training()

    hiddens, sampl_h = crbmnode.sample_h(u, x)

    print 'Sampling...'
    v_zero = np.random.normal(0, 1, u.shape)
    for i in range(25):
        visibles, sampl_v = crbmnode.sample_v(sampl_h, x)
        hiddens, sampl_h = crbmnode.sample_h(sampl_v, x)

    visibles, sampl_v = crbmnode.sample_v(sampl_h, x)
    error = np.mean((u.ravel() - visibles.ravel())**2)
    print 'Final MSE:', error

    pylab.clf()
    p = pylab.subplot(211)
    p = pylab.imshow(u[:500, :].T)
    p = pylab.subplot(212)
    p = pylab.imshow(visibles[:500, :].T)
    p = pylab.show()

