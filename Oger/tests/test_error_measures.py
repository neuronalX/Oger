import numpy as np
import nose
from Oger import utils

def test_nrmse():
    '''Test if nrmse gives correct known value for short vectors of three elements
    '''
    r = np.array([1, 1, 1.1])
    s = np.array([1, 1, 1.0])
    error = utils.nrmse(s, r)
    nose.tools.assert_almost_equal(error, 1)

def test_nrmse_unequal_length():
    '''Test if comparing unequal vectors raises an exception
    '''
    r = np.array([1, 1, 1.1])
    s = np.array([1, 1])
    try:
        utils.nrmse(s, r)
        err = "Nrmse did not complain about comparing vectors of unequal length."
        raise Exception(err)
    except RuntimeError:
        pass
    
def test_loss_01():
    ''' Test zero one loss on simple case '''
    assert utils.loss_01(np.array([[1], [2], [3]]), np.array([[1], [2], [4]])) == 1. / 3
