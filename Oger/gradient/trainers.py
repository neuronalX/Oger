"""
A collection of some trainers that use gradient information.

The convention is to minimize the objective (error) function.
"""

from mdp import numx

# More can be found here:
#  - http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
#  - http://openopt.org/Welcome
#  - http://wiki.sagemath.org/optimization

class ScipyTrainer(object):
    '''Base class for all scipy-optimize based trainers. Stores constructor args and kwargs for later passing to 
    the optimization call.'''
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

class CGTrainer(ScipyTrainer):
    """Trainer that uses conjugate gradient to optimize a loss function.
    
    See the documentation of scipy.optimize.fmin_cg for more details. 
    """
    def train(self, func, x0):
        import scipy.optimize as opt
        """Optimize parameters to minimze loss.

        Arguments:
            - func: A function of the parameters that returns a tuple with the gradient and the loss respectively.
            - x0: Parameters to use as starting point.

        Returns the parameters that minimize the loss.
        """
        fobj = lambda x: func(x)[1]
        fprime = lambda x: func(x)[0]
        return opt.fmin_cg(fobj, x0, fprime, *self.args, **self.kwargs)

class BFGSTrainer(ScipyTrainer):
    """Trainer that uses BFGS to optimize a loss function.
    
    See the documentation of scipy.optimize.fmin_bfgs for more details. 
    """
    def train(self, func, x0):
        import scipy.optimize as opt
        """Optimize parameters to minimze loss.

        Arguments:
            - func: A function of the parameters that returns a tuple with the gradient and the loss respectively.
            - x0: Parameters to use as starting point.

        Returns the parameters that minimize the loss.
        """
        fobj = lambda x: func(x)[1]
        fprime = lambda x: func(x)[0]
        return opt.fmin_bfgs(fobj, x0, fprime, *self.args, **self.kwargs)

class LBFGSBTrainer(ScipyTrainer):
    """Trainer that uses L-BFGS-B to optimize a loss function under cosntraints.
    
    See the documentation of scipy.optimize.fmin_l_bfgs_b for more details. 
    """
    def __init__(self, weight_bounds=(-1, 1), *args, **kwargs):
        super(LBFGSBTrainer, self).__init__(args, kwargs)
        if numx.rank(weight_bounds) == 0:
            self.weight_bounds = (-weight_bounds, weight_bounds)
        else:
            self.weight_bounds = weight_bounds
        self.args = args
        self.kwargs = kwargs

    def train(self, func, x0):
        import scipy.optimize as opt
        """Optimize parameters to minimze loss.

        Arguments:
            - func: A function of the parameters that returns a tuple with the gradient and the loss respectively.
            - x0: Parameters to use as starting point.

        Returns the parameters that minimize the loss.
        """
        fobj = lambda x: func(x)[1]
        fprime = lambda x: func(x)[0]
        bounds = [self.weight_bounds, ] * x0.size
        return opt.fmin_l_bfgs_b(fobj, x0, fprime=fprime, bounds=bounds, *self.args, **self.kwargs)[0]

class GradientDescentTrainer:
    def __init__(self, learning_rate=.01, momentum=0, epochs=1, decay=0):
        """
            - learning_rate: size of the gradient steps (default = .001)
            - momentum: momentum term (default = 0)
            - epochs: number of times to do updates on the same data (default = 1)
            - decay: weight decay term (default = 0)
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.decay = decay

        self.dparams = None

    def train(self, func, x0):
        """Optimize parameters to minimze loss.

        Arguments:
            - func: A function of the parameters that returns a tuple with the gradient and the loss respectively.
            - x0: Parameters to use as starting point.

        Returns the parameters that minimize the loss.
        """
        if self.dparams is None:
            self.dparams = numx.zeros(x0.shape)

        updated_params = x0

        for _ in range(self.epochs):
            gradient = func(updated_params)[0]
            self.dparams = self.momentum * self.dparams - self.learning_rate * gradient
            # TODO: how do we make sure that we do not decay the bias terms?
            updated_params += self.dparams - self.decay * updated_params

        return updated_params

class RPROPTrainer:
    def __init__(self, etamin=0.5, etaplus=1.2, deltamin=10e-6, deltamax=50, deltainit=0.0125, epochs=1):

        self._uW = None
        self.deltaW = None

        self.etamin = etamin
        self.etaplus = etaplus
        self.deltamin = deltamin
        self.deltamax = deltamax
        self.epochs = epochs
        self.deltainit = deltainit

    def train(self, func, x0):
        """Optimize parameters to minimze loss.

        Arguments:
            - func: A function of the parameters that returns a tuple with the gradient and the loss respectively.
            - x0: Parameters to use as starting point.

        Returns the parameters that minimize the loss.
        """
        if self._uW is None:
            # TODO: should we not refcast here?
            self._uW = numx.zeros_like(x0)
            self.deltaW = numx.ones_like(x0) * self.deltainit

        updated_params = x0.copy()

        for _ in range(self.epochs):
            # TODO: properly name variables
            uW = func(updated_params)[0]

            WW = self._uW * uW

            self.deltaW *= self.etaplus * (WW > 0) + self.etamin * (WW < 0) + 1 * (WW == 0)

            self.deltaW = numx.maximum(self.deltaW, self.deltamin)
            self.deltaW = numx.minimum(self.deltaW, self.deltamax)

            updated_params -= self.deltaW * numx.sign(uW)

            self._uW = uW

        return updated_params

