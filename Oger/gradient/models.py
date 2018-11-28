"""
Several pre-defined models that use the gradient nodes.
"""
import mdp
import Oger
import trainers

# TODO: The Autoencoder has not been tested yet.


class MLPNode(mdp.Node):
    """Defines a multilayer perceptron with a hidden layer of tanh units.r
    """

    def __init__(self, input_dim, output_dim, hidden_dim=100,
                 trainer=trainers.GradientDescentTrainer(), loss='mse', dtype='float64'):
        """Initializes and constructs a multilayer perceptron.

        Arguments:
        
            - input_dim: input dimensionality
            - output_dim: output dimensionality
            - hidden_dim: number of hidden units
            - trainer: gradient based trainer to use, default: GradientDescentTrainer
            - loss: type of loss to minimize. Setting this to 'mse' will use linear outputs and minimize the mean squared error and setting this to 'ce' will use softmax outputs and minimize cross-entropy error.
        """


        
        super(MLPNode, self).__init__(input_dim, output_dim, dtype)

        if loss == 'mse':
            transfer = Oger.utils.LinearFunction()
            loss_f = Oger.utils.mse
        if loss == 'ce':
            transfer = Oger.utils.SoftmaxFunction()
            loss_f = Oger.utils.ce

        # TODO: Turn these into normal perceptron nodes once the extension
        # mechanism is fixed.
        perceptron1 = Oger.gradient.GradientPerceptronNode(input_dim, hidden_dim,
                                             transfer_func=Oger.utils.TanhFunction)
        perceptron2 = Oger.gradient.GradientPerceptronNode(hidden_dim, output_dim,
                                             transfer_func=transfer)

        # This is a flow.
        self.layers = perceptron1 + perceptron2

        self._bpnode = Oger.gradient.BackpropNode(self.layers, trainer, loss_f)

    def _train(self, x, t):
        """Train the perceptron to produce the desired output 't'."""

        self._bpnode.train(x=x, t=t)

    def _execute(self, x):

        return self._bpnode.execute(x)

    def is_trainable(self):
        return True
    
    def is_invertible(self):
        return False
    
    def _get_supported_dtypes(self):
        return ['float32', 'float64']


class AutoencoderNode(MLPNode):
    """Use a multilayer perceptron to reconstruct its input."""

    def __init__(self, input_dim, hidden_dim=100, trainer=trainers.GradientDescentTrainer(),
                 dtype='float64'):
        
        super(AutoencoderNode, self).__init__(input_dim, input_dim, hidden_dim, trainer,
                                              loss='mse', dtype=dtype)

    def _train(self, x):
        """Train the Autoencoder to recontstruct 'x'."""

        self._bpnode.train(x, x)

    def get_encoder(self):
        """Return the first PerceptronNode layer that encodes the input."""
        return self.layers[0]

