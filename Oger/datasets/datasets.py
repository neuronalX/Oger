import mdp
import glob
import os.path
import cPickle as pickle

def narma10(n_samples=10, sample_len=1000):
    ''' 
    narma10(n_samples=10, sample_len=1000) -> inputs, outputs
    Return data for the 10th order NARMA task. 
    Parameters are:
        - sample_len: length of the time-series in timesteps, default 1000
        - n_samples: number of example timeseries to be generated, default 10
        
    WARNING: this is an unstable dataset. There is a small chance the system becomes 
    unstable, leading to an unusable dataset. It is better to use NARMA30 which 
    where this problem happens less often.
    '''
    system_order = 10
    inputs, outputs = [], []
    for sample in range(n_samples):
        inputs.append(mdp.numx.random.rand(sample_len, 1) * .5)
        inputs[sample].shape = (-1, 1)
        outputs.append(mdp.numx.zeros((sample_len, 1)))
        for k in range(system_order-1, sample_len - 1):
            outputs[sample][k + 1] = .3 * outputs[sample][k] + .05 * \
            outputs[sample][k] * mdp.numx.sum(outputs[sample][k - (system_order-1):k+1]) + \
            1.5 * inputs[sample][k - 9] * inputs[sample][k] + .1
    return inputs, outputs 

def narma30(n_samples=10, sample_len=1000):
    ''' 
    narma30(n_samples=10, sample_len=1000) -> inputs, outputs
    Return data for the 30th order NARMA task. 
    Parameters are:
        - sample_len: length of the time-series in timesteps, default 1000
        - n_samples: number of example timeseries to be generated, default 10
    '''
    system_order = 30
    inputs, outputs = [], []
    for sample in range(n_samples):
        inputs.append(mdp.numx.random.rand(sample_len, 1) * .5)
        inputs[sample].shape = (-1, 1)
        outputs.append(mdp.numx.zeros((sample_len, 1)))
        for k in range(system_order-1, sample_len - 1):
            outputs[sample][k + 1] = .2 * outputs[sample][k] + .04 * \
            outputs[sample][k] * mdp.numx.sum(outputs[sample][k - (system_order-1):k+1]) + \
            1.5 * inputs[sample][k - 29] * inputs[sample][k] + .001
    return inputs, outputs 

def memtest(n_samples=10, sample_len=1000, n_delays=10):
    ''' 
    memtest(n_samples=10, sample_len=1000, n_delays=10) -> inputs, outputs
    Generates a series of input timeseries and delayed versions as outputs. 
    Delay is given in number of timesteps. Can be used to empirically measure the 
    memory capacity of a system.
    '''
    inputs, outputs = [], []
    for sample in range(n_samples):
        # Generate uniform noise between -0.8 and 0.8
        inputs.append((mdp.numx.random.rand(sample_len, 1) - .5) * 1.6)
        inputs[sample].shape = (-1, 1)
        outputs.append(mdp.numx.zeros((sample_len, n_delays)))
        for k in range(n_delays):
            outputs[sample][:, k:k + 1] = mdp.numx.concatenate((mdp.numx.zeros((k + 1, 1)), inputs[sample][:-k - 1, :]))
    return inputs, outputs 


def pickled_data(file_or_folder = None):
    '''
    Loads data from a pickle file or folder containing pickle files
    Data must be saved in tuples of the form '(x,y)' and the filename must end with '.pickle'
    '''
    if os.path.isdir(file_or_folder):
        files = glob.glob(os.path.join(file_or_folder, '*.pickle'))
    else:
        files = [file_or_folder]
    X, Y = [], []
    for f in files:
        x, y = pickle.load(open(f))
        if isinstance(x, mdp.numx.ndarray):
            X.append(x)
            Y.append(y)
        else:
            X.extend(x)
            Y.extend(y)
    return X, Y
