# This file demonstrates the use of the ERBMNode (on a very trivial task...).

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

    epochs = 10


    v = u

    rbmnode = Oger.nodes.ERBMNode(hidden_dim=100, visible_dim=20, gaussian=True)

    t, d = u.shape
    print 'Training for %d epochs...' % epochs
    for i in range(epochs):
        print 'Epoch', i
        # Update every two time steps as batch gradient descent barely converges.
        for j in range(t - 1):
            rbmnode.train(v[j:j + 1, :], n_updates=1, epsilon=.0001, momentum=.9, decay=.0002)
        # Check the energy and error after each training epoch.
        # Sampling causes the training phase to end so a copy is made first.
        rbmnode_test = rbmnode.copy()
        hiddens, sampl_h = rbmnode_test.sample_h(u)
        energy = rbmnode_test.energy(u, sampl_h)
        print 'Energy:', sum(energy)
        visibles, sampl_v = rbmnode_test.sample_v(sampl_h)
        error = np.mean((u.ravel() - visibles.ravel())**2)
        print 'MSE:', error

    rbmnode.stop_training()



