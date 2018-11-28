# Trains an ESN on sentences created by a probabilistic context free grammar.
# The performance is evaluated by measuring the cosine between the network
# outputs and the true probabilities distributions over the words according to
# the grammar itself.

import os
import gzip
import Oger

import mdp
import pylab

if __name__ == "__main__":

    l = Oger.datasets.simple_pcfg()

    testdata = []

    # Test sentences all have structure 'N V the N that N V .'
    print 'Generating data...'

    for N1 in l.nouns:
        for V1 in l.verbs:
            for N2 in l.nouns:
                for N3 in l.nouns:
                    for V2 in l.verbs:
                        s = [N1, V1, 'the', N2, 'that', N3, V2, '.']
                        testdata.append(s)

    # Number of sentences to train on.
    N = 5000

    trainsents = [l.S() for i in range(N)]
    trainwords = []

    for i in trainsents:
        trainwords.extend(i)

    testwords = []
    for i in testdata:
        testwords.extend(i)

    Nx = len(trainwords)

    vocab = l.nouns + l.verbs + [l.THAT] + [l.WITH] + [l.FROM] + [l.THE]\
            + [l.END]
    inputs = len(vocab)
    translate = dict([(vocab[i], i) for i in range(inputs)])

    reservoir = Oger.nodes.ReservoirNode(inputs, 100, input_scaling=1)
    readout = Oger.nodes.RidgeRegressionNode()

    # Build MDP network.
    flow = mdp.Flow([reservoir, readout], verbose=1)

    flownode = mdp.hinet.FlowNode(flow)

    # Contstruct a suitable train data matrix 'x'.
    indices = [translate[i] for i in trainwords]
    
    x = mdp.numx.zeros((Nx, inputs))
    x[mdp.numx.arange(Nx), mdp.numx.array(indices)] = 1

    # y contains a timeshifted version of the data in x.
    y = mdp.numx.mat(mdp.numx.zeros((Nx, inputs)))
    y[:-1, :] = x[1:, :]
    y[-1, :] = x[0, :]

    # Open file with true probabilities.
    trueprobs_file = os.getcwd() + '/../datasets/trueprobs_simple_pcfg.txt.gz'
    
    try:
        fin = gzip.open(trueprobs_file)
    except:
        print '''The dataset for this task was not found. Please download it from http://organic.elis.ugent.be/oger
        and put it in ''' + trueprobs_file
        exit()
        
    dat = fin.readlines()
    fin.close()

    testprobs = []
    for i in dat:
        line = i.strip().split()
        datapoint = mdp.numx.array([float(j) for j in line])
        testprobs.append(datapoint)

    print "Training..."
    flownode.train(x, y)
    flownode.stop_training()

    print "Testing..."

    # Contstruct a suitable train data matrix 'x' for the testset.
    indices = [translate[i] for i in testwords]
    
    Nx = len(testwords)
    x = mdp.numx.zeros((Nx, inputs))
    x[mdp.numx.arange(Nx), mdp.numx.array(indices)] = 1

    # Save test results in ytest.
    ytest = flownode(x)

    results = [Oger.utils.cosine(ytest[i], testprobs[i + 1]) for i in range(Nx - 1)]
    print 'Average cosine between outputs and ground distributions:', mdp.numx.mean(results)
    results = mdp.numx.array(results)[:-7]
    results = results.reshape(((Nx - 8) / 8, 8))
    means = mdp.numx.mean(results, 0)
    example = ['boys', 'see', 'the', 'clowns', 'that', 'pigs', 'follow', '<end>']
    pylab.xticks(mdp.numx.arange(8), example)
    pylab.bar(mdp.numx.arange(8), means)
    pylab.show()
    print 'Finished'

