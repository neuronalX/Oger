import Oger
import mdp
import pylab
import scipy as sp
import random
from Oger.utils import ConfusionMatrix, plot_conf

def loss_01_time(x, y):
    return Oger.utils.loss_01(mdp.numx.atleast_2d(sp.argmax(mdp.numx.mean(x, axis=0))), mdp.numx.atleast_2d(sp.argmax(mdp.numx.mean(y, axis=0))))

if __name__ == "__main__":

    n_subplots_x, n_subplots_y = 2, 1
    train_frac = .9

    [inputs, outputs] = Oger.datasets.analog_speech(indir="../datasets/Lyon_decimation_128")

    n_samples = len(inputs)
    n_train_samples = int(round(n_samples * train_frac))
    n_test_samples = int(round(n_samples * (1 - train_frac)))

    # Shuffle the data randomly
    data = zip(inputs, outputs)
    random.shuffle(data)
    inputs, outputs = zip(*data)

    input_dim = inputs[0].shape[1]

    # construct individual nodes
    reservoir = Oger.nodes.LeakyReservoirNode(input_dim=input_dim, output_dim=100, input_scaling=.1, leak_rate=.3)
    readout = Oger.nodes.RidgeRegressionNode()

    flow = mdp.Flow([reservoir, readout])

    # Tell the reservoir to save its states for later plotting 
    Oger.utils.make_inspectable(Oger.nodes.LeakyReservoirNode)

    # Plot an example input to the reservoir 
    pylab.subplot(n_subplots_x, n_subplots_y, 1)
    pylab.imshow(inputs[0].T, aspect='auto', interpolation='nearest')
    pylab.title("Cochleogram (input to reservoir)")
    pylab.ylabel("Channel")


    print "Training..."
    flow.train([[], zip(inputs[0:n_train_samples], outputs[0:n_train_samples])])

    ytest = []

    print "Applying to testset..."
    for xtest in inputs[n_train_samples:]:
        ytest.append(flow(xtest))

    pylab.subplot(n_subplots_x, n_subplots_y, 2)
    pylab.plot(reservoir.inspect()[0])
    pylab.title("Sample reservoir states")
    pylab.xlabel("Timestep")
    pylab.ylabel("Activation")

    print "Error : " + str(mdp.numx.mean([loss_01_time(sample, target) for (sample, target) in zip(ytest, outputs[n_train_samples:])]))

    ymean = sp.array([sp.argmax(mdp.numx.atleast_2d(mdp.numx.mean(sample, axis=0))) for sample in
                      outputs[n_train_samples:]])
    ytestmean = sp.array([sp.argmax(mdp.numx.atleast_2d(mdp.numx.mean(sample, axis=0))) for sample in ytest])

    # use ConfusionMatrix to compute some more information about the 
    confusion_matrix = ConfusionMatrix.from_data(10, ytestmean, ymean) # 10 classes
    print "Error rate: %.4f" % confusion_matrix.error_rate # this comes down to 0-1 loss
    print "Balanced error rate: %.4f" % confusion_matrix.ber
    print

    # compute precision and recall for each class vs. all others
    print "Per-class precision and recall"
    binary_confusion_matrices = confusion_matrix.binary()
    for c in range(10):
        m = binary_confusion_matrices[c]
        print "label %d - precision: %.2f, recall %.2f" % (c, m.precision, m.recall)
    print

    # properties of the ConfusionMatrix and BinaryConfusionMatrix classes can also be used
    # as error measure functions, as follows:
    ber = ConfusionMatrix.error_measure('ber', 10) # 10-class balanced error rate
    print "Balanced error rate: %.4f" % ber(ytestmean, ymean)

    # plot confusion matrix (balanced, each class is equally weighted)
    pylab.figure()
    plot_conf(confusion_matrix.balance())

