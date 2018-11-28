# This just measures mse on next symbol prediction.
# TODO: make a discriminative version to predict labels instead.
import time
import cudamat as cm
import mdp.utils
import Oger
import scipy as sp
from Oger.nodes import CUDATRM as ccrbm


reservoir_size = 200
hiddens = 400
n_labels = 53
data_size = 1

nx, ny = 4, 1
train_frac = .9

[x, y, samplename] = Oger.datasets.timit()

# y is coded {1, -1} and should be coded {1, 0}
for i in range(len(y)):
    z = y[i]
    z[z == -1] = 0

# Normalize data (this should be a node in the toolbox!)

print 'Normalizing data...'

# Get mean
data_mean = 0
N = 0
for i in range(len(x)):
    xt = x[i]
    data_mean += xt.sum(axis=0)
    N += xt.shape[0]

data_mean /= N

# Subtract it and get variance.

data_var = 0

for i in range(len(x)):
    xt = x[i]
    xt -= data_mean
    data_var += (xt ** 2).sum(axis=0)

data_var /= N

scaler = sp.diag(1 / sp.sqrt(data_var))

# Rescale data.

for i in range(len(x)):
    x[i] = sp.dot(x[i], scaler)

# Check
data_var = 0

for xt in x:
    data_var += (xt ** 2).sum(axis=0)

data_var /= N

### Actual experiment...

x = x[:data_size]
n_samples = len(x)
n_train_samples = int(round(n_samples * train_frac))
n_test_samples = int(round(n_samples * (1 - train_frac)))


inputs = x[0].shape[1]

# cudamat boiler plate

cm.init()
cm.CUDAMatrix.init_random(seed=42)
# construct reservoir

crbm = ccrbm(hiddens, inputs, reservoir_size, gaussian=True)

# Gathering the train states in advance is too memory intensive for bigger
# reservoirs but for testing it is no problem.
#print "Gathering ESN test states..."
#states_test = []
#for xt in mdp.utils.progressinfo(x[n_train_samples:]):
#    states_test.append(reservoir.simulate(cm.CUDAMatrix(xt)))

print "Training..."
epochs = 250
for epoch in range(epochs):
    for xt, yt in mdp.utils.progressinfo(zip(x[0:n_train_samples],
                                                y[0:n_train_samples])):
        batch_size = xt.shape[0] / 2
        state = crbm.reservoir.simulate(cm.CUDAMatrix(xt))

        crbm.v = cm.CUDAMatrix(xt[:batch_size, :])
        crbm.train(state.get_row_slice(0, batch_size), decay=0, epsilon=.001, momentum=.9)
        crbm.v = cm.CUDAMatrix(xt[batch_size:, :])
        crbm.train(state.get_row_slice(batch_size, xt.shape[0]), decay=0, epsilon=.001, momentum=.9)
        
    print 'epoch', epoch, 'finished'
    error = 0
    for xt, yt in mdp.utils.progressinfo(zip(x[0:n_train_samples],
                                                y[0:n_train_samples])):
        state = crbm.reservoir.simulate(cm.CUDAMatrix(xt))
        v = cm.CUDAMatrix(sp.random.normal(0, 1, (xt.shape)))
        crbm.v = v
        n = xt.shape[0]
        dynamic_h = cm.empty((n, crbm.output_dim))
        dynamic_v = cm.empty((n, crbm.visible_dim))
        cm.dot(state, crbm.ag, dynamic_v)
        cm.dot(state, crbm.bg, dynamic_h)
        for i in range(25):
            crbm._sample_h(crbm.v, dynamic_h, sample=True, x_is_bias=True)
            crbm._sample_v(crbm.h, dynamic_v, x_is_bias=True)
        error += sp.mean((crbm.v.asarray() - xt) ** 2)
    print error / n_train_samples

    # Evaluate reconstruction error
