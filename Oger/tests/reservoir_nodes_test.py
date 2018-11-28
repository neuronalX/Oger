import mdp
import nose
import unittest
import Oger
import scipy as sp

class ReservoirNodeTest(unittest.TestCase):
    def setUp(self):
        self.default_size = 10
        self.input_length = 10

    def test_spectral_radius(self):
        ''' Test if spectral radius is expected value '''

        # Default value
        r = Oger.nodes.ReservoirNode (input_dim = 1, output_dim=self.default_size)
        nose.tools.assert_almost_equals(Oger.utils.get_spectral_radius(r.w), sp.amax(sp.absolute(sp.linalg.eigvals(r.w)))) 

        # Custom value
        rho = 0.1
        r = Oger.nodes.ReservoirNode (input_dim = 1, output_dim=self.default_size, spectral_radius=rho)
        nose.tools.assert_almost_equals(Oger.utils.get_spectral_radius(r.w), rho) 

    def test_zero_input(self):
        ''' Test if zero input returns zero output without bias
        '''
        r = Oger.nodes.ReservoirNode (input_dim = 1, output_dim=self.default_size)
        assert sp.all(r(sp.zeros((self.input_length, 1))) == sp.zeros((self.input_length, self.default_size)))

    def test_input_mapping(self):
        ''' Test if input mapping is correct for zero internal weight matrix 
        '''
        r = Oger.nodes.ReservoirNode (input_dim = 1, output_dim=self.default_size, spectral_radius=0)
        assert sp.all(r(sp.ones((self.input_length, 1))) == sp.tanh(sp.dot(sp.ones((self.input_length, 1)), r.w_in.T)))
        
    def test_bias(self):
        ''' Test if turning on bias gives expected results 
        '''
        r = Oger.nodes.ReservoirNode (input_dim = 1, output_dim=self.default_size, spectral_radius=0, bias_scaling=1)
        assert sp.all(r(sp.zeros((self.input_length, 1))) == sp.tile(sp.tanh(r.w_bias), (self.input_length, 1)))

    def test_passing_custom_reservoir_matrix(self):
        ''' Test if passing your own reservoir weight matrix gives expected results 
        '''
        # Check if weight matrix is initialized correctly
        r = Oger.nodes.ReservoirNode (input_dim = 1, output_dim=2, w=sp.array([[1, 2], [3, 4]]))
        assert sp.all(r.w == sp.array([[1, 2], [3, 4]]))
        r.w = []
        r.initialize()
        assert sp.all(r.w == sp.array([[1, 2], [3, 4]]))
        
    def test_passing_custom_reservoir_matrix_function(self):
        ''' Test if passing your own reservoir weight matrix generation function gives expected results 
        '''
        # Check if weight matrix is initialized correctly
        r = Oger.nodes.ReservoirNode (input_dim = 1, output_dim=2, w=sp.ones((2, 2)))
        assert sp.all(r.w == sp.array([[1, 1], [1, 1]]))
        r.w = []
        r.initialize()
        assert sp.all(r.w == sp.array([[1, 1], [1, 1]]))
      
        # Check if dimensionality is checked correctly
        nose.tools.assert_raises(mdp.NodeException, Oger.nodes.ReservoirNode, input_dim = 1, output_dim=2, w=sp.ones((1, 1)))

    def test_passing_custom_input_matrix(self):
        ''' Test if passing your own input weight matrix gives expected results 
        '''
        # Check if weight matrix is initialized correctly
        r = Oger.nodes.ReservoirNode (input_dim = 1, output_dim=2, w_in=sp.array([[1], [2]]))
        assert sp.all(r.w_in == sp.array([[1], [2]]))
        r.w_in = []
        r.initialize()
        assert sp.all(r.w_in == sp.array([[1], [2]]))
        
        # Check if dimensionality is checked correctly
        nose.tools.assert_raises(mdp.NodeException, Oger.nodes.ReservoirNode, input_dim = 1, output_dim=2, w_in=sp.array([[1]]))

    def test_passing_custom_input_matrix_function(self):
        ''' Test if passing your own input weight matrix generation function gives expected results 
        '''
        # Check if weight matrix is initialized correctly
        r = Oger.nodes.ReservoirNode (input_dim = 1, output_dim=2, w_in=sp.ones((2, 1)))
        assert sp.all(r.w_in == sp.array([[1], [1]]))
        r.w_in = []
        r.initialize()
        assert sp.all(r.w_in == sp.array([[1], [1]]))
        
        # Check if dimensionality is checked correctly
        nose.tools.assert_raises(mdp.NodeException, Oger.nodes.ReservoirNode, input_dim = 1, output_dim=2, w_in=sp.ones((1, 1)))


    def test_passing_custom_bias_matrix(self):
        ''' Test if passing your own bias weight matrix gives expected results 
        '''
        # Check if weight matrix is initialized correctly
        r = Oger.nodes.ReservoirNode (input_dim = 1, output_dim=2, w_bias=sp.array([[1, 2]]))
        assert sp.all(r.w_bias == sp.array([1, 2]))
        r.w_bias = []
        r.initialize()
        assert sp.all(r.w_bias == sp.array([1, 2]))
        
        # Check if dimensionality is checked correctly
        nose.tools.assert_raises(mdp.NodeException, Oger.nodes.ReservoirNode, input_dim = 1, output_dim=2, w_bias=sp.array([[1]]))

    def test_passing_custom_bias_matrix_function(self):
        ''' Test if passing your own bias weight matrix generation function gives expected results 
        '''
        # Check if weight matrix is initialized correctly
        r = Oger.nodes.ReservoirNode (input_dim = 1, output_dim=2, w_bias=sp.ones((1,2)))
        assert sp.all(r.w_bias == sp.array([1, 1]))
        r.w_bias = []
        r.initialize()
        assert sp.all(r.w_bias == sp.array([1, 1]))
        
        # Check if dimensionality is checked correctly
        nose.tools.assert_raises(mdp.NodeException, Oger.nodes.ReservoirNode, input_dim=1, output_dim=2, w_bias=sp.ones((1)))
