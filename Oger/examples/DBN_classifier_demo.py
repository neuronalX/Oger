import mdp.utils
import pylab
import numpy as np
import cPickle
import Oger

print 'Loading data...'
data = cPickle.load(open('../mnist/mnist.p'))

n_train = 40000
n_test = 5000
epochs = 1

image_data = data['trainimages']
image_labels = data['trainlabels']

image_data /= 255.

train_data = image_data[:n_train]
train_labels = image_labels[:n_train]

test_data = image_data[n_train:n_train + n_test]
test_labels = image_labels[n_train:n_train + n_test]

# Generate a small subset of the data.

rbmnode1 = Oger.nodes.ERBMNode(784, 100)
rbmnode2 = Oger.nodes.ERBMNode(100, 200)
percnode = Oger.nodes.PerceptronNode(200, 10, transfer_func=Oger.utils.SoftmaxFunction)

# Greedy pretraining of RBMs

print 'Training first layer...'
for epoch in range(epochs):
    for c in mdp.utils.progressinfo(train_data):
        rbmnode1.train(c.reshape((1, 784)), n_updates=1, epsilon=.1)

hiddens = rbmnode1(train_data)

print 'Training second layer...'
for epoch in range(epochs):
    for c in mdp.utils.progressinfo(hiddens):
        rbmnode2.train(c.reshape((1, 100)), n_updates=1, epsilon=.1)

# Create flow and backpropagation node.

# Store weights.
w_generative = rbmnode1.w.copy()

myflow = rbmnode1 + rbmnode2 + percnode

bpnode = Oger.gradient.BackpropNode(myflow, Oger.gradient.GradientDescentTrainer(momentum=.9), loss_func=Oger.utils.ce)

# Fine-tune for classification
print 'Fine-tuning for classification...'
for epoch in range(epochs):
    for i in mdp.utils.progressinfo(range(len(train_data))):
        label = np.array(np.eye(10)[train_labels[i], :])
        bpnode.train(x=train_data[i].reshape((1, 784)), t=label.reshape((1, 10)))

# Evaluate performance on test set.
out = bpnode(test_data)

out[np.arange(out.shape[0]), np.argmax(out, axis=1)] = 1
out[out < 1] = 0
t_test = np.array([int(i) for i in test_labels])
correct = np.sum(out[np.arange(len(t_test)), t_test])
print 'Proportion of correctly classified digits:', correct / float(len(test_labels))

# Draw a picture of the input weights before and after fine-tuning.
image = np.zeros((28 * 20, 28 * 10))

for i in range(10):
    for j in range(10):
        image[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = w_generative[:, i * 10 + j].ravel().reshape((28, 28))


for i in range(10, 20):
    for j in range(10):
        image[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = rbmnode1.w[:, (i - 10) * 10 + j].ravel().reshape((28, 28))


pylab.imshow(image, cmap=pylab.gray())
pylab.show()

