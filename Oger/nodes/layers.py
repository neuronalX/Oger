import mdp

class SplitOutputLayer(mdp.hinet.Layer):
    """
    Like Layer, but during supervised training, the provided target data is split over
    all of the nodes.
     
    Needless to say, this is only useful for supervised nodes like regressors and classifiers.
    """

    def _train(self, x, y, *args, **kwargs):
        """Perform single training step by training the internal nodes."""
        x_idx = 0
        y_idx = 0
        for node in self.nodes:
            next_x_idx = x_idx + node.input_dim
            next_y_idx = y_idx + node.output_dim
            x_selected = x[:, x_idx:next_x_idx] # selected input dimensions for this node
            y_selected = y[:, y_idx:next_y_idx] # select output dimensions for this node
            x_idx = next_x_idx
            y_idx = next_y_idx

            node.train(x_selected, y_selected, *args, **kwargs) # train using only these input and output dimensions

class SplitOutputSameInputLayer(mdp.hinet.SameInputLayer):
    """
    Like SameInputLayer, but during supervised training, the provided target data is split over
    all of the nodes.
     
    Needless to say, this is only useful for supervised nodes like regressors and classifiers.
    """

    def _preprocess_training_data(self, node, x, y):
        """
        Hook for preprocessing the data (input + selected output dimensions) for each encapsulated node.
        Does nothing in this class, it's just here so it can be overridden in subclasses.
        """
        return x, y

    def _train(self, x, y, *args, **kwargs):
        """Perform single training step by training the internal nodes."""

        idx = 0
        for node in self.nodes:
            next_idx = idx + node.output_dim
            y_selected = y[:, idx:next_idx] # select output dimensions for this node
            idx = next_idx

            x_pr, y_pr = self._preprocess_training_data(node, x, y_selected)
            # check if there is any data left in x and y after preprocessing,
            # else the training function will throw an exception
            if (x_pr.size > 0):
                node.train(x_pr, y_pr, *args, **kwargs) # train using only these output dimensions
