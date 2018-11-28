import mdp.utils
import Oger

if __name__ == "__main__":

    nx, ny = 4, 1
    washout = 0
    train_frac = .9

    try:
        [x, y, samplename] = Oger.datasets.timit()
    except:
        print '''The dataset for this task was not found. Please download it from http://organic.elis.ugent.be/oger
        and put it in ../datasets/''' 
        exit()
    
    
    n_samples = len(x)
    n_train_samples = int(round(n_samples * train_frac))
    n_test_samples = int(round(n_samples * (1 - train_frac)))
    

    inputs = x[0].shape[1]
    # construct individual nodes
    reservoir = Oger.nodes.ReservoirNode(inputs, 500, input_scaling=1)
    readout = Oger.nodes.RidgeRegressionNode()

    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout], verbose=1)
    flownode = mdp.hinet.FlowNode(flow)
    
    print "Training..."
    # train and test it
    for xt, yt in mdp.utils.progressinfo(zip(x[0:n_train_samples - 1], y[0:n_train_samples - 1])):
        flownode.train(xt, yt)
    flownode.stop_training()

    ytrain, ytest = [], []
    
    print "Applying to trainingset..."
    for xtrain in mdp.utils.progressinfo(x[0:n_train_samples - 1]):
        ytrain.append(flownode(xtrain))

    print "Applying to testset..."
    for xtest in mdp.utils.progressinfo(x[n_train_samples:]):
        ytest.append(flownode(xtest))
    
    #pylab.subplot(nx,ny,2)
    #pylab.plot(ytrain[0], 'r')
    #pylab.plot(y[0], 'b')
    #pylab.title("Sample train output")
    #pylab.ylabel("Output")
    

    #pylab.subplot(nx,ny,3)
    #pylab.plot(ytest[5], 'r')
    #pylab.plot(y[305], 'b')
    #pylab.title("Sample test output")
    #pylab.ylabel("Output")
    
    #pylab.subplot(nx,ny,4)
    #pylab.plot(reservoir.states)
    #pylab.title("Sample reservoir states")
    #pylab.xlabel("Timestep")
    #pylab.ylabel("Activation")
    #pylab.show()
    
    
    #pylab.show()
    #ymean=[sp.argmax(sp.mean(sample, axis=0)) for sample in y[n_train_samples:]]
    #ytestmean=[sp.argmax(sp.mean(sample, axis=0)) for sample in ytest]
    
    
    #print Oger.utils.mean_error(ymean, ytestmean, lambda x, y, z: x!=y)
    print "finished"
