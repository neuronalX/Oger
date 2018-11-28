''' A demo to demonstrate the different implemented logistic regression methods
    convergence time might be very long for some algorithms...
    For comparison, simple ridge regression is also computed, which doesn't perform well in case of an unbalanced dataset.
'''

import mdp
from Oger.nodes import RidgeRegressionNode, IRLSLogisticRegressionNode, LogisticRegressionNode, ThresholdNode
from Oger.gradient import GradientDescentTrainer, CGTrainer, BFGSTrainer
from Oger.utils import loss_01
import scipy as sp
import pylab

def generate_cluster (c, N, var):
    return sp.random.randn(N, 2) * var + sp.tile(c, (N, 1))

# create 2 sets of points for 2 clusters (one training, one testing)
a_mean = [.1, .2]
b_mean = [.3, .5]
std = .1
a_train = generate_cluster(a_mean, 10, std)
b_train = generate_cluster(b_mean, 200, std)

a_test = generate_cluster(a_mean, 20, std)
b_test = generate_cluster(b_mean, 20., std)

xtrain = sp.concatenate((a_train, b_train))
xtest = sp.concatenate((a_test, b_test))
ytrain = sp.concatenate((sp.ones(a_train.shape[0]), sp.zeros(b_train.shape[0]))).reshape((-1, 1))
ytest = sp.concatenate((sp.ones(a_test.shape[0]), sp.zeros(b_test.shape[0]))).reshape((-1, 1))

# plot the data
for data, style in zip([a_train, b_train, a_test, b_test], ['g*', 'rd', 'go', 'rs']):
    pylab.plot(data[:, 0], data[:, 1], style, label='_nolegend_')

r = pylab.gca().get_xlim()

# Generate the nodes that will be trained
nodes = (RidgeRegressionNode(),
         IRLSLogisticRegressionNode(),
         LogisticRegressionNode(GradientDescentTrainer(), 'epochs', 2000),
         LogisticRegressionNode(CGTrainer()),
         LogisticRegressionNode(BFGSTrainer()),
         )

names = ('ridge regression', 'IRLS', 'gradient descent', 'conjugate gradient', 'BGFS')

# This threshold node will be appended to the trained classifier node, to transform the probabilities output by the logistic regression to the set {0, 1}
threshold_node = ThresholdNode(threshold=.5, output_values=[0, 1])

# Loop over the different training algorithms and train and test them
for node, name in zip(nodes, names):

    # Construct the flow
    flow = mdp.Flow([node, threshold_node])

    # Train the regression_node
    flow.train([[[xtrain, ytrain]], []])

    # Apply to the test set
    y = flow.execute(xtest)

    # Compute the discriminant line
    if isinstance(node, RidgeRegressionNode):
        y1 = (node.beta[1] * r[0] + node.beta[0] - .5) / -node.beta[2]
        y2 = (node.beta[1] * r[1] + node.beta[0] - .5) / -node.beta[2]
    else:
        y1 = (node.w[0] * r[0] + node.b) / -node.w[1]
        y2 = (node.w[0] * r[1] + node.b) / -node.w[1]

    # Plot the discriminant line
    pylab.plot(r, [y1, y2], label=name)

    # Calculate the error
    print "error rate %s: %.3f" % (name, loss_01(y, ytest))

pylab.legend()
pylab.show()
