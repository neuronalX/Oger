"""
This subpackage contains nodes and trainers for gradient based learning.

Its purpose is to automate backpropagation and gradient calculation for models
that can be trained using gradient methods.  This allows complex architectures
to be trained using external optimizers that can use gradient and loss
information.

The main building block is the BackpropNode that takes a flow of nodes that are
supported by the gradient extension and uses a trainer object to optimize the
parameters of the nodes in the flow.

Some of the nodes that are currently supported are the PerceptronNode and the
ERBMNode.  Examples of optimization algorithms that have been wrapped into
trainer objects are gradient descent, RPROP, conjugate gradient and BFGS.

The 'models' package contains some pre-defined architectures that use the
gradient package for training.
"""

from gradient_nodes import (GradientExtensionNode, BackpropNode, GradientPerceptronNode, GradientRBMNode)
from trainers import (CGTrainer, BFGSTrainer, RPROPTrainer, GradientDescentTrainer, LBFGSBTrainer)
from models import (MLPNode, AutoencoderNode)

del gradient_nodes
del trainers
del models

__all__ = ['GradientExtensionNode', 'BackpropNode', 'GradientPerceptronNode', 'GradientRBMNode',
          'CGTrainer', 'BFGSTrainer', 'RPROPTrainer', 'GradientDescentTrainer', 'LBFGSBTrainer', 'MLPNode,', 'AutoencoderNode']
