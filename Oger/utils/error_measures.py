import mdp
import numpy as np

def check_signal_dimensions(input_signal, target_signal):
    if input_signal.shape != target_signal.shape:
        raise RuntimeError("Input shape (%s) and target_signal shape (%s) should be the same."% (input_signal.shape, target_signal.shape))
    
class timeslice():
    """
    timeslice(range, function) -> function
    Apply the given function only to the given time range of the data.
    Can be used to eg. apply an error metric only to a part of the data.
    """
    def __init__(self, range, function):
        from functools import update_wrapper
        self.function = function
        self.range = range
        update_wrapper(self, function)
    def __call__(self, x, y):
        return self.function(x[self.range, :], y[self.range, :])

def nrmse(input_signal, target_signal):
    """
    nrmse(input_signal, target_signal)-> error
    NRMSE calculation.
    Calculates the normalized root mean square error (NRMSE) of the input signal compared to the target signal.
    Parameters:
        - input_signal : array
        - target_signal : array
    """
    check_signal_dimensions(input_signal, target_signal)
    
    input_signal = input_signal.flatten()
    target_signal = target_signal.flatten()
    
    # Use normalization with N-1, as in matlab
    var = target_signal.std(ddof=1) ** 2
    
    error = (target_signal - input_signal) ** 2
    return mdp.numx.sqrt(error.mean() / var)
    
def nmse(input_signal, target_signal):
    """ 
    nmse(input_signal, target_signal)->error
    NMSE calculation.
    Calculates the normalized mean square error (NMSE) of the input signal compared to the target signal.
    Parameters:
        - input_signal : array
        - target_signal : array
    """
    
    check_signal_dimensions(input_signal, target_signal)
    
    input_signal = input_signal.flatten()
    targetsignal = target_signal.flatten()

    var = targetsignal.std()**2
 
    error = (targetsignal - input_signal) ** 2
    return error.mean() / var
    
def rmse(input_signal, target_signal):
    """
    rmse(input_signal, target_signal)->error 
    RMSE calculation.
    
    Calculates the root mean square error (RMSE) of the input signal compared target target signal.
    
    Parameters:
        - input_signal : array
        - target_signal : array
    """
    check_signal_dimensions(input_signal, target_signal)
    
    error = (target_signal.flatten() - input_signal.flatten()) ** 2
    return mdp.numx.sqrt(error.mean())
    
def mse(input_signal, target_signal):
    """ 
    mse(input_signal, target_signal)->error
    MSE calculation.
    
    Calculates the mean square error (MSE)
    of the input signal compared target signal.
    
    Parameters:
        - input_signal : array
        - target_signal : array
    """   
    check_signal_dimensions(input_signal, target_signal)
    
    error = (target_signal.flatten() - input_signal.flatten()) ** 2
    return error.mean()

def loss_01(input_signal, target_signal):
    """ 
    loss_01(input_signal, target_signal)->error
    Compute zero one loss function 
    
    Returns the fraction of timesteps where input_signal is unequal to target_signal
    
    Parameters:
        - input_signal : array
        - target_signal : array
    """
    check_signal_dimensions(input_signal, target_signal)
    
    return np.mean(np.any(input_signal!= target_signal, 1))

def cosine(input_signal, target_signal):
    ''' 
    cosine(input_signal, target_signal)->cosine
    Compute cosine of the angle between two vectors
    
    This error measure measures the extent to which two vectors point in the same direction. 
    A value of 1 means complete alignment, a value of 0 means the vectors are orthogonal.
    '''
    check_signal_dimensions(input_signal, target_signal)
    
    return float(mdp.numx.dot(input_signal, target_signal)) / (mdp.numx.linalg.norm(input_signal) * mdp.numx.linalg.norm(target_signal))

def ce(input_signal, target_signal):
    """ 
    ce(input_signal, target_signal)-> cross-entropy
    Compute cross-entropy loss function

    Returns the negative log-likelyhood of the target_signal labels as predicted by
    the input_signal values.

    Parameters:
        - input_signal : array
        - target_signal : array
    """
    check_signal_dimensions(input_signal, target_signal)
    
    if np.rank(target_signal)>1 and target_signal.shape[1]>1:
        error = mdp.numx.sum(-mdp.numx.log(input_signal[target_signal == 1]))
        
        if mdp.numx.isnan(error):
            inp = input_signal[target_signal == 1]
            inp[inp ==0] = float(np.finfo(input_signal.dtype).tiny)
            error = -mdp.numx.sum(mdp.numx.log(inp))
    else:
        error = -mdp.numx.sum(mdp.numx.log(input_signal[target_signal == 1]))
        error -= mdp.numx.sum(mdp.numx.log(1 - input_signal[target_signal == 0]))
        
        if mdp.numx.isnan(error):
            inp = input_signal[target_signal == 1]
            inp[inp ==0] = float(np.finfo(input_signal.dtype).tiny)
            error = -mdp.numx.sum(mdp.numx.log(inp))
            inp = 1 - input_signal[target_signal == 0]
            inp[inp ==0] = float(np.finfo(input_signal.dtype).tiny)
            error -= mdp.numx.sum(mdp.numx.log(inp))
    
    return error


# TODO: if we add container object for the error metrics, we should add a field that
# signifies if we need to minimize or maximize the measure
def mem_capacity(input_signal, target_signal):
    """Computes the memory capacity defined by Jaeger in 
    H. Jaeger (2001): Short term memory in echo state networks. GMD Report 
    152, German National Research Center for Information Technology, 2001
    
    WARNING: currently this returns the negative of the memory capacity so 
    we can keep on using the minimization code.
    """
    check_signal_dimensions(input_signal, target_signal)
    
    score = []
    for k in range(target_signal.shape[1]):
        covariance_matrix = mdp.numx.cov(mdp.numx.concatenate((input_signal[:, k:k + 1].T, target_signal[:, k:k + 1].T)))
        score.append(covariance_matrix[0, 1] ** 2 / (covariance_matrix[0, 0] * covariance_matrix[1, 1]))
    
    return - mdp.numx.sum(score)


def threshold_before_error(input_signal, target_signal, error_measure=loss_01, thresh=None):
    """
    First applies a threshold to input_signal and target_signal and then determines the error using the error_measure function.
    The threshold is estimated as the mean of the target_signal maximum and minimum unless a threshold 'thresh' is specified
    
    Useful for classification error estimation. Example:
    error_measure = lambda x, y: Oger.utils.threshold_before_error(x, y, Oger.utils.loss_01)
    """
    check_signal_dimensions(input_signal, target_signal)
    
    if thresh == None:
        thresh = (max(target_signal) + min(target_signal)) / 2
    return error_measure(input_signal > thresh, target_signal > thresh)
    
    
def calcROC(Yh, Y):
    eps = 10**-10
    thrRange = np.fliplr(np.atleast_2d(np.unique(np.sort(np.hstack( (np.array([np.min(Yh)-eps, np.max(Yh)+eps]), Yh) ), ))))[0, :]
    fpr = np.zeros(thrRange.shape)
    tpr = np.zeros(thrRange.shape)
    Yp = Yh[Y == 1]
    Yf = Yh[Y != 1]
    for k in range(thrRange.shape[0]):
        thr = thrRange[k]
        fpr[k] = np.sum(1.0*Yf>=thr)/(1.0*Yf.shape[0])
        tpr[k] = np.sum(1.0*Yp>=thr)/(1.0*Yp.shape[0])
    AUC = np.sum((fpr[1:]-fpr[:-1])*(tpr[1:]+tpr[:-1]))/2.

    return AUC, fpr, tpr

#   error
#     
def auc_error(input_signal, target_signal):
    auc, fpr, tpr = calcROC(input_signal.flatten(), target_signal.flatten())
    return auc

