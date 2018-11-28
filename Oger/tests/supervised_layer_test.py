import mdp
import unittest
import Oger
import numpy as np
from Oger.nodes import SupervisedLayer

class SupervisedLayerTest(unittest.TestCase):
    def setUp(self):
        # create a very simple dataset with 2 input channels and 2 output channels
        x = np.atleast_2d(mdp.numx.arange(50)).astype(float).T
        self.x = np.hstack([x, x**2])
        a, b = 0.5, 2
 
        y = a * x + b
        self.y = np.hstack([y, y**2])

        # add noise
        mu, sigma = 0, 0.1 # mean and standard deviation
        noise = np.random.normal(mu, sigma, self.y.shape)
        self.y += noise

    def test_no_partitioning(self): # compare with SameInputLayer
        n1 = mdp.nodes.LinearRegressionNode(input_dim=2, output_dim=2)
        n2 = mdp.nodes.LinearRegressionNode(input_dim=2, output_dim=2)
        l = SupervisedLayer([n1, n2], input_partitioning=False, target_partitioning=False)
        l.train(self.x, self.y)
        l.execute(self.x)
        
    def test_input_partitioning(self): # compare with Layer
        n1 = mdp.nodes.LinearRegressionNode(input_dim=1, output_dim=2)
        n2 = mdp.nodes.LinearRegressionNode(input_dim=1, output_dim=2)
        l = SupervisedLayer([n1, n2], input_partitioning=True, target_partitioning=False)
        l.train(self.x, self.y)
        l.execute(self.x)
        
    def test_output_partitioning(self):
        n1 = mdp.nodes.LinearRegressionNode(input_dim=2, output_dim=1)
        n2 = mdp.nodes.LinearRegressionNode(input_dim=2, output_dim=1)
        l = SupervisedLayer([n1, n2], input_partitioning=False, target_partitioning=True)
        l.train(self.x, self.y)
        l.execute(self.x)
        
    def test_all_partitioning(self): # input and output
        n1 = mdp.nodes.LinearRegressionNode(input_dim=1, output_dim=1)
        n2 = mdp.nodes.LinearRegressionNode(input_dim=1, output_dim=1)
        l = SupervisedLayer([n1, n2], input_partitioning=True, target_partitioning=True)
        l.train(self.x, self.y)
        l.execute(self.x)
        
        
        
if __name__ == '__main__':
    unittest.main()

