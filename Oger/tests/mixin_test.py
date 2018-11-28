import mdp
import unittest
import Oger

class MixinTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_washout_mixin_unsup(self):
        x = mdp.numx.ones((100, 1000)) + mdp.numx.random.randn(100, 1000)
        Oger.utils.enable_washout(mdp.nodes.PCANode)
        pca = mdp.nodes.PCANode(reduce=True, output_dim = 20)
        pca.washout = 10

        assert(pca.is_trainable())
        y = pca(x)
        assert(x.shape[0] == y.shape[0])
        Oger.utils.disable_washout(mdp.nodes.PCANode)

    def test_washout_mixin_sup(self):
        x = mdp.numx.zeros((100, 1000))
        y = mdp.numx.zeros((100, 1000))
        Oger.utils.enable_washout(mdp.nodes.LinearRegressionNode, 10)
        lin = mdp.nodes.LinearRegressionNode()

        assert(lin.is_trainable())
        lin.train(x, y=y)
        Oger.utils.disable_washout(mdp.nodes.LinearRegressionNode)

    def test_washout_mixin_execute(self):
        x = mdp.numx.zeros((100, 1000))
        y = mdp.numx.zeros((100, 1000))
        Oger.utils.enable_washout(mdp.nodes.SignumClassifier, 10, execute_washout = True)
        lin = mdp.nodes.SignumClassifier()

        #assert(lin.is_trainable())
        y = lin(x)
        assert(y.shape[0] == x.shape[0] - lin.washout)
        Oger.utils.disable_washout(mdp.nodes.SignumClassifier)
