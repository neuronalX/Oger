'''
The following implements an example of the experiments run in: "Unsupervised Learning 
in Reservoir Computing: Modeling Hippocampal Place Cells for Small Mobile Robots" by
E. Antonelo and B. Schrauwen

It takes robot sensor data, runs it through a hierarchy of a reservoir, then slow 
feature analysis, then independent component analysis. Finally 20 of the outputs of
the ICA layer are plotted along with the numbered robot location (see paper).

It can be seen that certain outputs repeatedly produce a spike of activity at precise locations.

Location of the robot is the topmost plot.

'''
from scipy.io.matlab.mio import loadmat
from scipy.signal.signaltools import resample
import Oger
import matplotlib.pyplot as plt
import mdp.nodes
import numpy as np



if __name__ == '__main__':
    # take matlab file in as a dictionary
    try:
        dictFile = loadmat("../datasets/eric_robotsensors.mat", struct_as_record=True)
    except:
        print '''The dataset for this task was not found. Please download it from http://organic.elis.ugent.be/oger 
        and put it in ../datasets/''' 
        exit()
    
    # the matlab file contains:
    # 'data_info' : holds xy position, location number, etc.
    # 'sensors' : the sensor information at each time step
    # 'sensors_resampled' : a downsampling of the sensor data with x50 less timesteps
    
    # these have time along the x axis
    sensorData = np.array(dictFile.get('sensors_resampled')) 
    dataInfo = np.array(dictFile.get('data_info'))
    # 5th index contains the location number at each timestep
    location = dataInfo[4, :]
        
    resDims = 400 #dimensions in the reserv
    sfaNum = 70 #number of slow features to extract
    icaNum = 70 # number of independent components to extract
    leakRate = 0.6 # leak rate of the reservoir
    specRadius = 0.9 # spectral radius

    inputDims = sensorData.shape[0]
    
    # define the reservoir and pass the spectrogram through it
    resNode = Oger.nodes.LeakyReservoirNode(input_dim=inputDims,
                  output_dim=resDims, spectral_radius=specRadius, leak_rate=leakRate)

    # Creation of the input weight matrix according to paper
    # -0.2,0.2 and 0 with probabilities of 0.15,0.15 and 0.7 respectively 
    w_in = np.zeros((resDims, inputDims))
    for i in range(resDims):
        for j in range(inputDims):
            ran = np.random.rand()
            if ran < 0.15:
                w_in[i, j] = -0.2
            elif ran < 0.3:
                w_in[i, j] = 0.2
            else:
                w_in[i, j] = 0 
                
    # set the input weight matrix for reservoir                
    resNode.w_in = w_in
    resNode.initialize()

    # define the sfa node
    sfaNode = mdp.nodes.SFANode(output_dim=sfaNum)
    
    # define the ica node
    icaNode = mdp.nodes.FastICANode()
    icaNode.set_output_dim(icaNum)

    #define the flow
    flow = mdp.Flow(resNode + sfaNode + icaNode)
    #train the flow
    flow.train(sensorData.T)
    # 
    icaOutput = flow.execute(sensorData.T)
    
    #resample the ica layer output back to 50 times the length 
    icaOutputLong = resample(icaOutput, location.shape[0])

    # number of components to plot
    plotsNum = 20
    
    plt.subplot(plotsNum + 1, 1, 1)
    # first subplot of the numbered location of the robot
    plt.plot(location)
    # plot the independent components
    for i in range(plotsNum):
        plt.subplot(plotsNum + 1, 1, i + 2)
        plt.plot(icaOutputLong[:, i])
    plt.show()
    





