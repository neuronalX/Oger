import unittest
import numpy as np
from Oger.nodes import MaxVotingNode

class MaxVotingNodeTest(unittest.TestCase):
    def setUp(self):
        self.data = np.array([
            [0.3, 0.5, 0.2],    # 1
            [8, 6, 4],          # 0
            [-1, 0, 1],         # 2
            [0.2, 0.1, 0.1],    # 0
        ])

    def test_properties(self):
        node = MaxVotingNode(input_dim=3)
        self.assertEqual(node.input_dim, 3)
        self.assertEqual(node.output_dim, 1)
        self.assertFalse(node.is_trainable())
        self.assertFalse(node.is_invertible())

    def test_no_labels(self):
        node = MaxVotingNode(input_dim=3)
        out = list(node(self.data))
        self.assertEquals(out, [1, 0, 2, 0])
        
    def test_integer_labels(self):
        node = MaxVotingNode(input_dim=3, labels=[10, 20, 30])
        out = list(node(self.data))
        self.assertEquals(out, [20, 10, 30, 10])
        
    def test_float_labels(self):
        node = MaxVotingNode(input_dim=3, labels=[0.1, 0.2, 0.3])
        out = list(node(self.data))
        self.assertEquals(out, [0.2, 0.1, 0.3, 0.1])
        
    def test_string_labels(self):
        node = MaxVotingNode(input_dim=3, labels=['a', 'b', 'c'])
        out = list(node(self.data))
        self.assertEquals(out, ['b', 'a', 'c', 'a'])
        
        
if __name__ == '__main__':
    unittest.main()

