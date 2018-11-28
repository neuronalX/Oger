"""
Module for MDP Nodes that support gradient based learning.

This module contains the gradient node base class and some gradient based
implementations of MDP nodes.

"""

import mdp
import Oger
from mdp import numx
from mdp.utils import mult

def traverseHinet(item):
    # TODO: could be implemented with a generetor

    previous = []

    if isinstance(item, mdp.Flow):
        for child in item:
            previous += traverseHinet(child)
    elif isinstance(item, mdp.hinet.Layer):
        for child in item:
            previous += traverseHinet(child)
    elif isinstance(item, mdp.hinet.FlowNode):
        for child in item.flow:
            previous += traverseHinet(child)
    else:
        previous += [item, ]

    return previous

class GradientExtensionNode(mdp.ExtensionNode):
    """Base class for gradient based MDP nodes.

    The gradient method returns the gradient of a node based on its last input
    and output after a backpropagation sweep.  The params method returns the
    parameters of the node as a 1-d array.  The set_params method takes a
    1-d array as its argument and uses it to update the parameters of the node.
    """

    extension_name = "gradient"

    def _execute(self, x, *args, **kwargs):
        """ Standard _execute but it saves the state."""
        # TODO: This only seems to work for nodes that explicitly redefine
        # _execute and not for nodes that inherited it.
        y = self._non_extension__execute(x, *args, **kwargs)
        self._last_x = x
        self._last_y = y
        return y

    def _inverse(self, y):
        """Calls _calculate_gradient instead of the default _inverse."""
        return self._calculate_gradient(y)

    def is_invertible(self):
        return True

    def is_trainable(self):
        """Make sure that a node which is trainable, is made untrainable
        so that we can run execute on a possible untrained node."""
        return False

    def is_training(self):
        """Make sure that a node which is trainable, is made untrainable
        so that we can run execute on a possible untrained node."""
        return False

    def get_current_train_phase(self):
        """Make sure that a node which is trainable, is made untrainable
        so that we can run execute on a possible untrained node."""
        return - 1

    def gradient(self):
        """Return the gradient that was found after the last backpropagation sweep."""
        return self._gradient_vector

    def params(self):
        """Return the parameters of the node as a 1-d array."""
        return self._params()

    def set_params(self, x):
        """Update the parameters of the node with a 1-d array."""
        self._set_params(x)

    def _params(self):
        pass

    def _set_params(self, x):
        pass

    def _calculate_gradient(self, x):
        return x

    def _params_size(self):
        raise 0

class BackpropNode(mdp.Node):
    """Node that handles backpropagation through a flow.
        
    It contains methods for obtaining gradients and loss values from a flow
    of gradient nodes.  It also has a trainer assigned that uses these
    methods for optimization of the parameters of all the nodes in the flow.
    """

#    def _stop_train_dummy(self):
#        pass
#    
#    def _get_train_seq(self):
#        train_list = [(self._train, self._stop_train_dummy)] * (self._n_epochs)
#        train_list.append((self._train, self._stop_training))
#        return train_list

    def __init__(self, gflow, gtrainer, loss_func=None, derror=None, n_epochs=1, dtype='float64'):
        """Create a BackpropNode that encapsulates a flow of gradient nodes.

        Arguments:
            - gflow: The flow of gradient supported nodes to use.
            - gtrainer: A trainer object to use for training the flow.
            - loss_func: A scalar returning loss function.
            - derror: The gradient of the loss function with respect to the outputs.  By default this will taken to be 'y - t' where y is the output and t the desired value.
        """

        self.gflow = gflow
        self.gtrainer = gtrainer

        # TODO: can this combination be in an object
        self.loss_func = loss_func
        self.derror = derror

        if self.derror is None:
            #self.derror = lambda x, t: x - t
            self.derror = self.derror_linear

        input_dim = gflow[0].get_input_dim()
        output_dim = gflow[-1].get_output_dim()

        self._n_epochs = n_epochs

        super(BackpropNode, self).__init__(input_dim, output_dim, dtype)

    def derror_linear(self, x, t):
        return x - t

    @mdp.with_extension('gradient')
    def _train(self, x, *args, **kwargs):
        """Update the parameters according to the input 'x' and target output 't'."""

        # Extract target values and delete them so they don't interfere with
        # the train method call below.
        # TODO: Perhaps the use of target values should be optional to allow
        # for unsupervised algorithms etc.
        if (len(args) > 0):
            t = args[0]
        else:
            t = kwargs.get('t')

        # Generate objective function for the current data.
        def func(params):
            return self._objective(x, t, params)

        update = self.gtrainer.train(func, self._params())

        self._set_params(update)

    def _objective(self, x, t, params=None):
        """Get the gradient and loss of the objective.

        This method returns a tuple with the gradient as a 1-d array and the
        loss if available.  If params is defined it will first update the
        parameters.
        """

        if params is not None:
            self._set_params(params)

        y = self.gflow.execute(x)

        if self.loss_func:
            loss = self.loss_func(y, t)
        else:
            loss = None

        delta = self.derror(y, t)

        self.gflow.inverse(delta)
        gradient = self._gradient()

        return (gradient, loss)

    def _gradient(self):
        """Get the gradient with respect to the parameters.

        This gradient has been calculated during the last backprop sweep.
        """

        gradient = numx.array([])

        for n in traverseHinet(self.gflow):
            if hasattr(n, '_param_size') and n._param_size() > 0:
                gradient = numx.concatenate((gradient, n.gradient()))

        return gradient

    def _params(self):
        """Return the current parameters of the nodes."""

        params = numx.array([])

        for n in traverseHinet(self.gflow):
            if hasattr(n, '_param_size') and n._param_size() > 0:
                params = numx.concatenate((params, n.params()))

        return params

    def _set_params(self, params):

        # Number of parameters we distributed so far.
        counter = 0

        for n in traverseHinet(self.gflow):
            if hasattr(n, '_param_size') and n._param_size() > 0:
                length = n._param_size()
                n.set_params(params[counter:counter + length])
                counter += length

    def _execute(self, x):
        return self.gflow.execute(x)

    def is_trainable(self):
        return True



## MDP (Oger) gradient node implementations ##

class GradientPerceptronNode(GradientExtensionNode, Oger.nodes.PerceptronNode):
    """Gradient version of Oger Perceptron Node"""

    def _params(self):
        return numx.concatenate((self.w.ravel(), self.b.ravel()))

    def _set_params(self, x):
        nw = self.w.size
        self.w.flat = x[:nw]
        self.b = x[nw:]

    def _calculate_gradient(self, y):
        ''' y is the gradient that is propagated from the previous layer'''
        x = self._last_x
        dy = self.transfer_func.df(x, self._last_y) * y
        dw = mult(x.T, dy)
        self._gradient_vector = numx.concatenate((dw.ravel(), dy.sum(axis=0)))
        dx = mult(self.w, dy.T).T
        return dx

    def _param_size(self):
        return self.w.size + self.b.size

class GradientRBMNode(GradientExtensionNode, Oger.nodes.ERBMNode):
    """Gradient version of the Oger RBM Node.

    This gradient node is intended for use in a feedforward architecture. This
    means that the biases for the visibles are ignored.
    """

    def _params(self):
        return numx.concatenate((self.w.ravel(), self.bh.ravel()))

    def _set_params(self, x):
        nw = self.w.size
        self.w.flat = x[:nw]
        self.b = x[nw:]

    def _calculate_gradient(self, y):
        x = self._last_x
        dy = Oger.utils.LogisticFunction.df(x, self._last_y) * y
        dw = mult(x.T, dy)
        self._gradient_vector = numx.concatenate((dw.ravel(), dy.sum(axis=0)))
        dx = mult(self.w, dy.T).T
        return dx

    def _param_size(self):
        return self.w.size + self.bh.size

    def inverse(self, y):
        """Calls _gradient_inverse instead of the default _inverse."""
        return self._calculate_gradient(y)

