import Oger

if __name__ == "__main__":

    n_subplots_x, n_subplots_y = 2, 1
    train_frac = .05

    [inputs, outputs] = Oger.datasets.analog_speech(indir="../datasets/Lyon_128")

    n_samples = len(inputs)
    n_train_samples = int(round(n_samples * train_frac))
    n_test_samples = int(round(n_samples * (1 - train_frac)))

    input_dim = inputs[0].shape[1]

    reservoir = Oger.nodes.BrianIFReservoirNode(inputs[0].shape[1], output_dim=10, input_conn_frac=.5, input_scaling=100, dt=1, dtype=float, we_scaling=1, wi_scaling=1, we_sparseness=.05, wi_sparseness=.1)
    readout = Oger.nodes.RidgeRegressionNode(0.001)
    mnnode = Oger.nodes.MeanAcrossTimeNode()

    # build network with MDP framework
    #flow = mdp.Flow([reservoir, readout, mnnode])
    flow = Oger.nodes.InspectableFlow([reservoir, readout, mnnode])

    print "Training..."
    # train and test it
    flow.train([None, zip(inputs[0:n_train_samples - 1], outputs[0:n_train_samples - 1]), None])

    ytrain, ytest = [], []
