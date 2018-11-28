import Oger
import mdp
import pylab
import scipy as sp
from pyNN.pcsim import *
from pyNN import space
    
if __name__ == "__main__":

    n_subplots_x, n_subplots_y = 2, 1
    train_frac = .9
    
    [inputs, outputs] = Oger.datasets.analog_speech(indir="Lyon128")
    
    n_samples = len(inputs)
    n_train_samples = int(round(n_samples * train_frac))
    n_test_samples = int(round(n_samples * (1 - train_frac)))
    
    input_dim = inputs[0].shape[1]
    
    cell_params = {
            'tau_m'      : 20.0, # (ms)
            'tau_syn_E'  : 2.0, # (ms)
            'tau_syn_I'  : 4.0, # (ms)
            'tau_refrac' : 2.0, # (ms)
            'v_rest'     : 0.0, # (mV)            
            'v_reset'    : 0.0, # (mV)
            'v_thresh'   : 20.0, # (mV)
            'cm'         : 0.5}  # (nF)
    
    RateScale = 1e6
    Tstep = 10.0
    
    rngseed = 1240498
    
    
    width = 1.0
    depth = 1.0
    height = 6.0
    
    exc_structure = space.RandomStructure(boundary = space.Cuboid(width, height, depth))
    inh_structure = space.RandomStructure(boundary = space.Cuboid(width, height, depth))
    
    c_lambda = 1.0
    
    exc_connector = DistanceDependentProbabilityConnector(
                                    'exp(-d*d/(%f*%f))' % (c_lambda, c_lambda),
                                    weights=0.1,
                                    delays=1.0)
    
    inh_connector = DistanceDependentProbabilityConnector(
                                    'exp(-d*d/(%f*%f))' % (c_lambda, c_lambda),
                                    weights=-0.4, 
                                    delays=1.0)
    
    input_connector = FixedProbabilityConnector(p_connect = 0.2, 
                                                weights=0.2, delays=1.0)
    
    # construct individual nodes
    reservoir = Oger.nodes.SpatialSpikingReservoirNode(
                    input_dim, dtype = float, rngseed = rngseed,
                    n_exc = 320, n_inh = 80, 
                    exc_structure = exc_structure,
                    inh_structure = inh_structure, 
                    exc_cell_type = IF_curr_alpha, 
                    exc_cell_params = cell_params,
                    inh_cell_type = IF_curr_alpha, 
                    inh_cell_params = cell_params, 
                    exc_connector = exc_connector, 
                    inh_connector = inh_connector,
                    input_connector = input_connector, 
                    kernel=Oger.utils.exp_kernel(tau=30, dt=1),
                    inp2spikes_conversion=Oger.utils.poisson_gen(rngseed, RateScale, Tstep),
                    syn_dynamics = { 'e2e': SynapseDynamics(
                                        fast= TsodyksMarkramMechanism(U=0.5, tau_rec=1100.0, tau_facil = 50.0)),
                                  'e2i': SynapseDynamics(
                                        fast= TsodyksMarkramMechanism(U=0.05, tau_rec=125.0, tau_facil = 1200.0)),
                                  'i2e': SynapseDynamics(
                                        fast= TsodyksMarkramMechanism(U=0.25, tau_rec=700.0, tau_facil = 20.0)),
                                  'i2i': SynapseDynamics(
                                        fast= TsodyksMarkramMechanism(U=0.32, tau_rec=144.0, tau_facil = 60.0))
                                }, inp_syn_dynamics = { '2exc': None,
                                                        '2inh': None })
    

    print "Total number of connections in the reservoir is ", len(reservoir.e2e_conn) + \
        len(reservoir.e2i_conn) + len(reservoir.i2e_conn) + len(reservoir.i2i_conn)
     
    readout = Oger.nodes.RidgeRegressionNode(0.001)
    mnnode = Oger.nodes.MeanAcrossTimeNode()
    
    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout, mnnode])
    
    pylab.subplot(n_subplots_x, n_subplots_y, 1)
    pylab.imshow(inputs[0].T, aspect='auto', interpolation='nearest')
    pylab.title("Cochleogram (input to reservoir)")
    pylab.ylabel("Channel")
    
    
    print "Training..."
    # train and test it
    flow.train([[], \
                zip(inputs[0:n_train_samples - 1], \
                    outputs[0:n_train_samples - 1]), \
                []])
    
    ytrain, ytest = [], []
    print "Applying to trainingset..."
    for xtrain in inputs[0:n_train_samples - 1]:
        ytrain.append(flow(xtrain))
    print "Applying to testset..."
    for xtest in inputs[n_train_samples:]:
        ytest.append(flow(xtest))
    
    pylab.subplot(n_subplots_x, n_subplots_y, 2)
    pylab.plot(reservoir.states)
    pylab.title("Sample reservoir states")
    pylab.xlabel("Timestep")
    pylab.ylabel("Activation")
    
    ymean = sp.array([sp.argmax(sample) for sample in 
                      outputs[n_train_samples:]])
    ytestmean = sp.array([sp.argmax(sample) for sample in ytest])
    
    print "Error: " + str(mdp.numx.mean(Oger.utils.loss_01(ymean,
                                                               ytestmean)))
    pylab.show()
