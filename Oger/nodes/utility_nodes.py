import mdp.nodes
from mdp import numx
#import scipy.signal #commented by xav
import numpy as np


class FeedbackNode(mdp.Node):
    """FeedbackNode creates the ability to feed back a certain part of a flow as
    input to the flow. It both implements the Node API and the generator API and
    can thus be used as input for a flow.

    The duration that the feedback node feeds back data can be given. Prior to using
    the node as data generator, it should be executed so it can store the previous
    state.

    When a FeedbackNode is reused several times, reset() should be called prior to
    each use which resets the internal counter.

    Note that this node keeps state and can thus NOT be used in parallel using threads.
    """
    def __init__(self, n_timesteps=1, input_dim=None, dtype=None):
        super(FeedbackNode, self).__init__(input_dim=input_dim, output_dim=input_dim, dtype=dtype)

        self.n_timesteps = n_timesteps
        self.last_value = None
        self.current_timestep = 0

    def reset(self):
        self.current_timestep = 0

    def is_trainable(self):
        return True

    def _train(self, x, y):
        self.last_value = mdp.numx.atleast_2d(y[-1, :])

    def __iter__(self):
        while self.current_timestep < self.n_timesteps:
            self.current_timestep += 1

            yield self.last_value

    def _execute(self, x):
        self.last_value = mdp.numx.atleast_2d(x[-1, :])
        return x

class MeanAcrossTimeNode(mdp.Node):
    """
        Compute mean across time (2nd dimension)

    """

    def __init__(self, input_dim=None, output_dim=None, dtype='float64'):
        super(MeanAcrossTimeNode, self).__init__(input_dim, output_dim, dtype)

    def is_trainable(self):
        return False

    def is_invertible(self):
        return False

    def _check_train_args(self, x, y):
        # set output_dim if necessary
        if self._output_dim is None:
            self._set_output_dim(y.shape[1])

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        e = mdp.numx.atleast_2d(mdp.numx.mean(x, axis=0, dtype=self.dtype))
        return e

class WTANode(mdp.Node):
    """
        Compute Winner take-all at every timestep (2nd dimension)

    """

    def __init__(self, input_dim=None, output_dim=None, dtype='float64'):
        super(WTANode, self).__init__(input_dim, output_dim, dtype)

    def is_trainable(self):
        return False

    def is_invertible(self):
        return False

    def _check_train_args(self, x, y):
        #set output_dim if necessary
        if self._output_dim is None:
            self._set_output_dim(y.shape[1])

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        max_indices = mdp.numx.argmax(x, axis=1)
        r = -mdp.numx.ones_like(x)
        for i in range(r.shape[0]):
            r[i, max_indices[i]] = 1
        return r


class ShiftNode(mdp.Node):
    """Return input data shifted one or more time steps.

    This is useful for architectures in which data from different time steps is
    needed. The values that are left over are set to zero.

    Negative shift values cause a shift back in time and positive ones forward in time.
    """

    def __init__(self, input_dim=None, output_dim=None, n_shifts=1,
                 dtype='float64'):
        super(ShiftNode, self).__init__(input_dim, output_dim, dtype)
        self.n_shifts = n_shifts

    def is_trainable(self):
        False

    def _execute(self, x):
        n = x.shape
        assert(n > 1)

        ns = self.n_shifts
        y = x.copy()

        if ns < 0:
            y[:ns] = x[-ns:]
            y[ns:] = 0
        elif ns > 0:
            y[ns:] = x[:-ns]
            y[:ns] = 0
        else:
            y = x

        return y

    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = n


class ResampleNode(mdp.Node):
    """ Resamples the input signal. Based on scipy.signal.resample

    CODE FROM: Georg Holzmann
    """

    def __init__(self, input_dim=None, ratio=0.5, dtype='float64', window=None):
        """ Initializes and constructs a random reservoir.
                - input_dim: the number of inputs (output dimension is always the same as input dimension)
                - ratio: ratio of up or down sampling (e.g. 0.5 means downsampling to half the samplingrate)
                - window: see window parameter in scipy.signal.resample
        """
        super(ResampleNode, self).__init__(input_dim, input_dim, dtype)
        self.ratio = ratio
        self.window = window

    def is_trainable(self):
        return False

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        """ Resample input vector x.
        """
        self.oldlength = len(x)
        newlength = self.oldlength * self.ratio
        sig = scipy.signal.resample(x, newlength, window=self.window)
        return sig.copy()

    def _inverse(self, y):
        """ Inverse the resampling.
        """
        sig = scipy.signal.resample(y, self.oldlength, window=self.window)
        return sig.copy()

class TimeFramesNode2(mdp.nodes.TimeFramesNode):
    """ An extension of TimeFramesNode that preserves the temporal
    length of the data.
    """
    def __init__(self, time_frames, input_dim=None, dtype=None):
        super(TimeFramesNode2, self).__init__(input_dim=input_dim, dtype=dtype, time_frames=time_frames)

    def _execute(self, x):
        tf = x.shape[0] - (self.time_frames - 1)
        rows = self.input_dim
        cols = self.output_dim
        y = mdp.numx.zeros((x.shape[0], cols), dtype=self.dtype)
        for frame in range(self.time_frames):
            y[-tf:, frame * rows:(frame + 1) * rows] = x[frame:frame + tf, :]
        return y

    def pseudo_inverse(self, y):
        pass


class FeedbackShiftNode(mdp.Node):
    """ Shift node that can be applied when using generators.
    The node works as a delay line with the number of timesteps the lengths of the delay line.
    """

    def __init__(self, input_dim=None, output_dim=None, n_shifts=1,
                 dtype='float64'):
        super(FeedbackShiftNode, self).__init__(input_dim, output_dim, dtype)
        self.n_shifts = n_shifts
        self.y = None
    def is_trainable(self):
        False

    def _execute(self, x):
        n = x.shape
        assert(n > 1)

        if self.y == None:
            self.y = np.zeros((self.n_shifts, self._input_dim))

        self.y = np.vstack([self.y, x.copy()])

        returny = self.y[:x.shape[0], :].copy()
        self.y = self.y[x.shape[0]:, :]
        return returny

    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = n


class RescaleZMUSNode(mdp.Node):
    '''
    Rescales the output with the mean and standard deviation seen during training

    If 'use_var' is set, the variance is used instead of the standard deviation

    Currently for 1 input only!!
    '''
    def __init__(self, use_var=False, input_dim=None, dtype=None):
        super(RescaleZMUSNode, self).__init__(input_dim=input_dim, dtype=dtype)
        self._mean = 0
        self._std = 0
        self._len = 0
        self._use_var = use_var

    def is_trainable(self):
        return True

    def _train(self, x):
        self._mean += mdp.numx.mean(x) * len(x)
        self._std += mdp.numx.sum(x ** 2) - mdp.numx.sum(x) ** 2
        self._len += len(x)

    def _stop_training(self):
        self._mean /= self._len
        self._std /= self._len
        if self._use_var:
            self._std = mdp.numx.sqrt(self._std)

    def _execute(self, x):
        return (x - self._mean) / self._std


class SupervisedLayer(mdp.hinet.Layer):
    """
    An extension of the MDP Layer class that is aware of target labels. This allows for
    more flexibility when using supervised techniques.

    The SupervisedLayer can mimic the behaviour of both the regular MDP Layer and the
    SameInputLayer, with regards to the partitioning of the input training data.

    In addition, the SupervisedLayer is also aware of target labels, and can partition
    them according to the output dimensions of the contained nodes, or not partition
    them at all. The latter is the behaviour of the label-agnostic MDP Layer
    and SameInputLayer classes.

    The SupervisedLayer has two flags that toggle input and target label partitioning:
    * input_partitioning (defaults to True)
    * target_partitioning (defaults to False)

    The defaults mimic the behaviour of the MDP Layer class. Setting 'input_partitioning'
    to False causes SameInputLayer-like behaviour.

    Because this class explicitly refers to target labels (second argument of the 'train'
    method), it will not work properly when used with unsupervised nodes.

    EXAMPLE

    A layer could contain 5 regressor nodes, each of which have 4-dimensional input and
    3-dimensional target labels. In that case, the input_dim of the layer is 5*4 = 20,
    and the output_dim is 5*3 = 15.

    A default Layer will split the input data according to the input_dims of the contained
    nodes, so the 20 input channels will be split into 5 sets of 4, which is the desired
    behaviour.

    However, the Layer is unaware of the target labels and simply passes through additional
    arguments to the 'train' function to the contained nodes. This means that each of the
    regressors will receive the same set of 15-dimensional target labels. The Layer should
    instead split the 15 target channels into 5 sets of 3, but it is not capable of doing
    so. Replacing the Layer with a SupervisedLayer(input_partitioning=True,
    target_partitioning=True) solves this problem.

    Another use case is one where the regressors have the same input data, but are trained
    with different target labels. In that case, a SuperivsedLayer(input_partitioning=False,
    target_partitioning=True) can be used. Using the previous example, each regressor then
    has an input_dim of 20 and an output_dim of 3.
    """

    def __init__(self, nodes, dtype=None, input_partitioning=True, target_partitioning=False):
        self.input_partitioning = input_partitioning
        self.target_partitioning = target_partitioning

        self.nodes = nodes
        # check nodes properties and get the dtype
        dtype = self._check_props(dtype)

        # set the correct input/output dimensions.
        # The output_dim of the Layer is always the sum of the output dims of the nodes,
        # Regardless of whether target partitioning is enabled.
        output_dim = self._get_output_dim_from_nodes()

        # The input_dim of the Layer however depends on whether input partitioning is
        # enabled. When input_partitioning is disabled, all contained nodes should have
        # the same input_dim and the input_dim of the layer should be equal to it.
        if self.input_partitioning:
            input_dim = 0
            for node in nodes:
                input_dim += node.input_dim

        else: # all nodes should have same input_dim, input_dim of the layer is equal to this
            input_dim = nodes[0].input_dim
            for node in nodes:
                if not node.input_dim == input_dim:
                    err = "The nodes have different input dimensions."
                    raise mdp.NodeException(err)

        # intentionally use MRO above Layer, not SupervisedLayer
        super(mdp.hinet.Layer, self).__init__(input_dim=input_dim,
                                    output_dim=output_dim,
                                    dtype=dtype)

    def is_invertible(self):
        return False # inversion is theoretically possible if input partitioning is enabled.
        # however, it is not implemented.

    def _train(self, x, y, *args, **kwargs):
        """Perform single training step by training the internal nodes."""
        x_idx, y_idx = 0, 0

        for node in self.nodes:
            if self.input_partitioning:
                next_x_idx = x_idx + node.input_dim
                x_selected = x[:, x_idx:next_x_idx] # selected input dimensions for this node
                x_idx = next_x_idx
            else:
                x_selected = x # use all input dimensions

            if self.target_partitioning:
                next_y_idx = y_idx + node.output_dim
                y_selected = y[:, y_idx:next_y_idx] # select target dimensions for this node
                y_idx = next_y_idx
            else:
                y_selected = y # use all target dimensions

            if node.is_training():
                node.train(x_selected, y_selected, *args, **kwargs)


    def _pre_execution_checks(self, x):
        """Make sure that output_dim is set and then perform normal checks."""
        if self.input_partitioning: # behaviour is like Layer, so just use the method of the parent class
            super(SupervisedLayer, self)._pre_execution_checks(x)
        else: # behaviour is like SameInputLayer
            if self.output_dim is None:
                # first make sure that the output_dim is set for all nodes
                for node in self.nodes:
                    node._pre_execution_checks(x)
                self.output_dim = self._get_output_dim_from_nodes()
                if self.output_dim is None:
                    err = "output_dim must be set at this point for all nodes"
                    raise mdp.NodeException(err)
            # intentionally use MRO above Layer, not SupervisedLayer
            super(mdp.hinet.Layer, self)._pre_execution_checks(x)

    def _execute(self, x, *args, **kwargs):
        """Process the data through the internal nodes."""
        if self.input_partitioning: # behaviour is like Layer, so just use the method of the parent class
            return super(SupervisedLayer, self)._execute(x, *args, **kwargs)
        else: # behaviour is like SameInputLayer
            out_start = 0
            out_stop = 0
            y = None
            for node in self.nodes:
                out_start = out_stop
                out_stop += node.output_dim
                if y is None:
                    node_y = node.execute(x, *args, **kwargs)
                    y = numx.zeros([node_y.shape[0], self.output_dim],
                                   dtype=node_y.dtype)
                    y[:, out_start:out_stop] = node_y
                else:
                    y[:, out_start:out_stop] = node.execute(x, *args, **kwargs)
            return y


class MaxVotingNode(mdp.Node):
    """
    This node finds the maximum value of all input channels at each timestep,
    and returns the corresponding label.

    If no labels are supplied, the index of the channel is returned.
    """

    def __init__(self, labels=None, input_dim=None, dtype='float64'):
        super(MaxVotingNode, self).__init__(input_dim, 1, dtype) # output_dim is always 1
        if labels is None:
            self.labels = None
        else:
            self.labels = np.asarray(labels)

    def is_trainable(self):
        return False

    def is_invertible(self):
        return False

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        if self.labels is None:
            self.labels = np.arange(self.input_dim) # default labels = channel indices
        indices = np.atleast_2d(np.argmax(x, 1)).T
        return self.labels[indices]
