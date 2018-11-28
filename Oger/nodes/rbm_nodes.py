# RBM based nodes. So far this file only contains the CRBM and supports only
# binary units as it is based on the RBMNode that has the same limitation.
# TODO: fix the energy functions so that they are correct for gaussian units.

import Oger
import mdp
try:
    import cudamat as cm
except:
    pass
from mdp.utils import mult
from mdp.nodes import RBMNode



random = mdp.numx_rand.random
randn = mdp.numx_rand.randn
exp = mdp.numx.exp


class ERBMNode(RBMNode):
    """'Enhanced' Restricted Boltzmann Machine node. This node implements the
    same model as the RBMNode class but has some additional functionality.
    Gaussian units can be used, a method has been added that returns the
    contrastive divergence gradient and the node has been made invertible.

    For simplicity, the Gaussian units are assumed to have unit variance.
    """

    def __init__(self, visible_dim, hidden_dim, gaussian=False, dtype=None):
        """
        Arguments:
            - hidden_dim: number of hidden variables
            - visible_dim:  number of observed variables
            - gaussian: use gaussian visible units (default is binary)
        """
        super(RBMNode, self).__init__(visible_dim, hidden_dim, dtype)
        self._initialized = False
        self._gaussian = gaussian


    def _energy(self, v, h):
        if self._gaussian:
            return ((((v - self.bv) ** 2).sum() / 2) - mult(h, self.bh) -
                    (mult(v, self.w) * h).sum(axis=1))
        else:
            return (-mult(v, self.bv) - mult(h, self.bh) -
                    (mult(v, self.w) * h).sum(axis=1))

    def train(self, v, n_updates=1, epsilon=0.1, decay=0., momentum=0.):
        """Update the internal structures according to the input data 'v'.
           The training is performed using Contrastive Divergence (CD).

               - v: a binary matrix having different variables on different columns and observations on the rows
               - n_updates: number of CD iterations. Default value: 1
               - epsilon: learning rate. Default value: 0.1
               - decay: weight decay term. Default value: 0.
               - momentum: momentum term. Default value: 0.
        """

        if not self.is_training():
            errstr = "The training phase has already finished."
            raise mdp.TrainingFinishedException(errstr)

        self._train_phase_started = True

        self._train(v, n_updates, epsilon, decay, momentum)

    def _train(self, v, n_updates=1, epsilon=0.1, decay=0., momentum=0.):
        """Update the internal structures according to the input data 'v'.
        The training is performed using Contrastive Divergence (CD).

            - v: a binary matrix having different variables on different columns and observations on the rows
            - n_updates: number of CD iterations. Default value: 1
            - epsilon: learning rate. Default value: 0.1
            - decay: weight decay term. Default value: 0.
            - momentum: momentum term. Default value: 0.
        """

        if not self._initialized:
            self._init_weights()

        # useful quantities
        n = v.shape[0]
        w, bv, bh = self.w, self.bv, self.bh

        # old gradients for momentum term
        dw, dbv, dbh = self._delta

        # get contrastive divergence gradient
        dwt, dbvt, dbht = self.get_CD_gradient(v, n_updates)

        # update parameters
        dw = momentum * dw + epsilon * dwt - decay * w
        w += dw

        dbv = momentum * dbv + epsilon * dbvt
        bv += dbv

        dbh = momentum * dbh + epsilon * dbht
        bh += dbh

        self._delta = (dw, dbv, dbh)


    def get_CD_gradient(self, v, n_updates=1):
        """Use Gibbs sampling to estimate the contrastive divergence gradient.

            - v: a binary matrix having different variables on different columns and observations on the rows
            - n_updates: number of CD iterations. Default value: 1

        Returns a tuple (dw, dbv, dbh) that contains the gradients of the
        weights and the biases of the visibles and the hidden respectively.
        """

        n = v.shape[0]

        # first update of the hidden units for the data term
        ph_data, h_data = self._sample_h(v)
        # n updates of both v and h for the model term
        h_model = h_data.copy()
        for i in range(n_updates):
            pv_model, v_model = self._sample_v(h_model)
            if self._gaussian:
                ph_model, h_model = self._sample_h(pv_model)
            else:
                ph_model, h_model = self._sample_h(v_model)

        # find dw
        data_term = mult(v.T, ph_data)
        model_term = mult(v_model.T, ph_model)
        dw = (data_term - model_term) / n

        # find dbv
        data_term = v.sum(axis=0)
        model_term = v_model.sum(axis=0)
        dbv = (data_term - model_term) / n

        # find dbh
        data_term = ph_data.sum(axis=0)
        model_term = ph_model.sum(axis=0)
        dbh = (data_term - model_term) / n

        return (dw, dbv, dbh)

    def is_invertible(self):
        return True

    def _sample_v(self, h):
        # returns  P(v=1|h,W,b) and a sample from it
        v_in = self.bv + mult(h, self.w.T)
        if self._gaussian:
            return v_in, v_in
        else:
            probs = 1. / (1. + exp(-v_in))
            v = (probs > random(probs.shape)).astype(self.dtype)
            return probs, v

    def _inverse(self, y, return_probs=True):
        """If 'return_probs' is True, returns the mean field of the
        visible variables v[n,i] conditioned on the states of the hiddens. 
        If 'return_probs' is False, return a sample from that probability.
        """
        probs, v = self._sample_v(y)
        if return_probs:
            return probs
        else:
            return v

    def is_trainable(self):
        return True

    # TODO: This _execute method is identical to the one in the original RBMNode and
    # could just have been inherited but somehow this messes up the
    # BackpropNode throwing that _non_extension__execute doesn't exist.
#    def _execute(self, v, return_probs=True):
#        """If 'return_probs' is True, returns the probability of the
#        hidden variables h[n,i] being 1 given the observations v[n,:].
#        If 'return_probs' is False, return a sample from that probability.
#        """
#        probs, h = self._sample_h(v)
#        if return_probs:
#            return probs
#        else:
#            return h
#

class CRBMNode(ERBMNode):
    """Conditional Restricted Boltzmann Machine node. This type of
    RBM models the joint probability of the hidden and visible
    variables conditioned on a certain context variable. See the
    documentation of the RBMNode for more information.

    The context variables are expected to be concatendated to the
    input data. Note that the sample functions do however expect
    these types of variables as separated arguments. This has been
    done to allow for easier construction of flows while being
    able to specify context data on the fly as well.
    """

    def __init__(self, hidden_dim, visible_dim=None, context_dim=None,
                 gaussian=False, dtype=None):
        """
        Arguments:

            - hidden_dim: number of hidden variables
            - visible_dim: number of observed variables
            - context_dim: number of context variables
            - gaussian: use gaussian visible units (default is binary)
        """
        super(RBMNode, self).__init__(hidden_dim, visible_dim + context_dim, dtype)
        self._input_dim = visible_dim + context_dim
        self._output_dim = hidden_dim

        self.context_dim = context_dim
        self.visible_dim = visible_dim
        self._initialized = False

        self._gaussian = gaussian

    def _init_weights(self):
        # weights and biases are initialized to small random values to
        # break the symmetry that might lead to degenerate solutions during
        # learning
        self._initialized = True

        # undirected weights
        self.w = self._refcast(randn(self.visible_dim, self.output_dim) * 0.01)
        # context to visible weights
        self.a = self._refcast(randn(self.context_dim, self.visible_dim) * 0.01)
        # context to hidden weights
        self.b = self._refcast(randn(self.context_dim , self.output_dim) * 0.01)
        # bias on the visible (input) units
        self.bv = self._refcast(randn(self.visible_dim) * 0.01)
        # bias on the hidden (output) units
        self.bh = self._refcast(randn(self.output_dim) * 0.01)

        # delta w, a, b, bv, bh used for momentum term
        self._delta = (0., 0., 0., 0., 0.)

    def _split_data(self, x):
        # split data into visibles and context respectively.
        return x[:, :self.visible_dim], x[:, self.visible_dim:]

    def _sample_h(self, v, x):
        # returns P(h=1|v,W,b) and a sample from it
        dynamic_b = mult(x, self.b)
        probs = Oger.utils.LogisticFunction.f(self.bh + mult(v, self.w) + dynamic_b)
        h = (probs > random(probs.shape)).astype(self.dtype)
        return probs, h

    def _sample_v(self, h, x):
        # returns  P(v=1|h,W,b) and a sample from it
        dynamic_b = mult(x, self.a)
        v_in = self.bv + mult(h, self.w.T) + dynamic_b
        if self._gaussian:
            return v_in, v_in
        else:
            probs = Oger.utils.LogisticFunction.f(v_in)
            v = (probs > random(probs.shape)).astype(self.dtype)
            return probs, v

    def train(self, x, n_updates=1, epsilon=0.1, decay=0., momentum=0.):
        """Update the parameters according to the input 'v' and context 'x'.
        The training is performed using Contrastive Divergence (CD).

            - v: a binary matrix having different variables on different columns and observations on the rows.
            - x: a matrix having different variables on different columns and observations on the rows.
            - n_updates: number of CD iterations. Default value: 1
            - epsilon: learning rate. Default value: 0.1
            - decay: weight decay term. Default value: 0.
            - momentum: momentum term. Default value: 0.
        """

        if not self.is_training():
            errstr = "The training phase has already finished."
            raise mdp.TrainingFinishedException(errstr)

        #self._check_input(x)

        self._train_phase_started = True
        self._train(x, n_updates, epsilon, decay, momentum)

    def get_CD_gradient(self, x, n_updates=1):
        """Use Gibbs sampling to estimate the contrastive divergence gradient.

            - x: a binary matrix having different variables on different columns and observations on the rows (concatenation of visibles and context)
            - n_updates: number of CD iterations. Default value: 1

        Returns a tuple (dw, dbv, dbh, da, db) that contains the gradients of the
        weights and the biases of the visibles and the hidden respectively and
        the autoregressive gradients da and db.
        """

        # useful quantities
        n = x.shape[0]
        v, x = self._split_data(x)
        w, a, b, bv, bh = self.w, self.a, self.b, self.bv, self.bh

        # first update of the hidden units for the data term
        ph_data, h_data = self._sample_h(v, x)
        # n updates of both v and h for the model term
        h_model = h_data.copy()
        for i in range(n_updates):
            pv_model, v_model = self._sample_v(h_model, x)
            ph_model, h_model = self._sample_h(v_model, x)

        # find dw
        data_term = mult(v.T, ph_data)
        model_term = mult(v_model.T, ph_model)
        dw = (data_term - model_term) / n

        # find da
        data_term = v
        model_term = v_model
        # Should I include the weight decay here as well?
        da = mult(x.T, data_term - model_term) / n

        # find db
        data_term = ph_data
        model_term = ph_model
        db = mult(x.T, data_term - model_term) / n

        # find dbv
        data_term = v.sum(axis=0)
        model_term = v_model.sum(axis=0)
        dbv = (data_term - model_term) / n

        # find dbh
        data_term = ph_data.sum(axis=0)
        model_term = ph_model.sum(axis=0)
        dbh = (data_term - model_term) / n

        return (dw, dbv, dbh, da, db)

    def _train(self, x, n_updates=1, epsilon=0.1, decay=0., momentum=0.):
        """Update the parameters according to the input 'v' and context 'x'.
        The training is performed using Contrastive Divergence (CD).

            - v: a binary matrix having different variables on different columns and observations on the rows.
            - x: a matrix having different variables on different columns and observations on the rows.
            - n_updates: number of CD iterations. Default value: 1
            - epsilon: learning rate. Default value: 0.1
            - decay: weight decay term. Default value: 0.
            - momentum: momentum term. Default value: 0.
        """
        if not self._initialized:
            self._init_weights()


        # useful quantities
        n = x.shape[0]
        w, a, b, bv, bh = self.w, self.a, self.b, self.bv, self.bh

        # old gradients for momentum term
        dw, da, db, dbv, dbh = self._delta

        # get the gradient
        dwt, dbvt, dbht, dat, dbt = self.get_CD_gradient(x, n_updates)

        # update w
        dw = momentum * dw + epsilon * dwt - decay * w
        w += dw

        # update a
        da = momentum * da + epsilon * dat - decay * a
        a += da

        # update b
        db = momentum * db + epsilon * dbt - decay * b
        b += db

        # update bv
        dbv = momentum * dbv + epsilon * dbvt
        bv += dbv

        # update bh
        dbh = momentum * dbh + epsilon * dbht
        bh += dbh

        self._delta = (dw, da, db, dbv, dbh)


    def sample_h(self, v, x):
        """Sample the hidden variables given observations v and context x.

        Returns a tuple (prob_h, h), where prob_h[n,i] is the
        probability that variable 'i' is one given the observations
        v[n,:], and h[n,i] is a sample from the posterior probability."""

        # The pre execution checks assume that v will give the input_dim but
        # this is not correct anymore due to the concatenation for execute.
        # Perhaps I should make two versions of the sample functions. One type
        # for the merged and one type for the separated input variables.
        #self._pre_execution_checks(v)
        return self._sample_h(v, x)

    def sample_v(self, h, x):
        """Sample the observed variables given hidden variables h and context.

        Returns a tuple (prob_v, v), where prob_v[n,i] is the
        probability that variable 'i' is one given the hidden variables
        h[n,:], and v[n,i] is a sample from that conditional probability."""

        #self._pre_inversion_checks(h)
        return self._sample_v(h, x)

    def _energy(self, v, h, x):
        ba = mult(x, self.a)
        bb = mult(x, self.b)
        ba += self.bv
        bb += self.bh
        if self._gaussian:
            return (((v - ba) ** 2).sum() / 2 - (h * bb).sum(axis=1) -
                    (mult(v, self.w) * h).sum(axis=1))
        else:
            return (-(v * ba).sum(axis=1) - (h * bb).sum(axis=1) -
                    (mult(v, self.w) * h).sum(axis=1))

    def energy(self, v, h, x):
        """Compute the energy of the RBM given observed variables state 'v' and
        hidden variables state 'h'."""
        return self._energy(v, h, x)

    def _execute(self, x, return_probs=True):
        """If 'return_probs' is True, returns the probability of the
        hidden variables h[n,i] being 1 given the observations v[n,:] and
        the context state x.
        If 'return_probs' is False, return a sample from that probability.
        """
        v, x = self._split_data(x)
        probs, h = self._sample_h(v, x)
        if return_probs:
            return probs
        else:
            return h

class CUDACRBMNode(ERBMNode):
    """Conditional Restricted Boltzmann Machine node. This type of
    RBM models the joint probability of the hidden and visible
    variables conditioned on a certain context variable. See the
    documentation of the RBMNode for more information.

    The context variables are expected to be concatendated to the
    input data. Note that the sample functions do however expect
    these types of variables as separated arguments. This has been
    done to allow for easier construction of flows while being
    able to specify context data on the fly as well.

    This cudamat version assumes that cudamat has been initialized!
    """

    def __init__(self, hidden_dim, visible_dim=None, context_dim=None,
                 gaussian=False, dtype=None, max_batch_size=1500):
        """
        Arguments:
            - hidden_dim: number of hidden variables
            - visible_dim: number of observed variables
            - context_dim: number of context variables
            - gaussian: use gaussian visible units (default is binary)
            - max_batch_size: number of datapoints to process simultaneously



        The max batch size is required to be able to optimize the adding of
        bias vectors by preallocating a vector of ones and taking the outer
        product with the bias vector. This is faster than directly adding it
        somehow.
        """
        super(RBMNode, self).__init__(hidden_dim, visible_dim + context_dim, dtype)
        self._input_dim = visible_dim + context_dim
        self._output_dim = hidden_dim

        self.context_dim = context_dim
        self.visible_dim = visible_dim
        self._initialized = False
        self._ones = cm.empty((1, max_batch_size))
        self._ones.assign(1)

        self._gaussian = gaussian

    def _init_weights(self):
        # weights and biases are initialized to small random values to
        # break the symmetry that might lead to degenerate solutions during
        # learning
        self._initialized = True

        # undirected weights
        self.w = self._refcast(randn(self.visible_dim, self.output_dim) * 0.01)
        # context to visible weights
        self.a = self._refcast(randn(self.context_dim, self.visible_dim) * 0.00)
        # context to hidden weights
        self.b = self._refcast(randn(self.context_dim , self.output_dim) * 0.01)
        # bias on the visible (input) units
        self.bv = self._refcast(randn(1, self.visible_dim) * 0.005)
        # bias on the hidden (output) units
        self.bh = self._refcast(randn(1, self.output_dim) * 0.005)

        # Copy to GPU

        self.wg = cm.CUDAMatrix(self.w)
        self.ag = cm.CUDAMatrix(self.a)
        self.bg = cm.CUDAMatrix(self.b)
        self.bvg = cm.CUDAMatrix(self.bv)
        self.bhg = cm.CUDAMatrix(self.bh)

        # Momentum terms

        self.mwg = cm.CUDAMatrix(np.zeros(self.w.shape))
        self.mag = cm.CUDAMatrix(np.zeros(self.a.shape))
        self.mbg = cm.CUDAMatrix(np.zeros(self.b.shape))
        self.mbvg = cm.CUDAMatrix(np.zeros(self.bv.shape))
        self.mbhg = cm.CUDAMatrix(np.zeros(self.bh.shape))

    def _split_data(self, x):
        # split data into visibles and context respectively.
        # TODO: Since this is probably more expensive than assigning it will be
        # perhaps be worth it to remove this whole splitting thing or to make
        # it store the values to preallocated memory instead.
        return cm.CUDAMatrix(x[:, :self.visible_dim]), cm.CUDAMatrix(x[:, self.visible_dim:])

    def _sample_h(self, v, x, sample=False, x_is_bias=False):
        # updates self.h
        #

        self.h = cm.empty((v.shape[0], self.output_dim))

        if x_is_bias: # Bias is precalculated
            self.h.assign(x)
        else:
            cm.dot(x, self.bg, self.h)

        self.h.add_dot(v, self.wg)

        # This is a 100 times faster than calling 'add_row_vec' to add biases.
        ones_cut = self._ones.get_col_slice(0, v.shape[0])
        self.h.add_dot(ones_cut.T, self.bhg)

        self.h.apply_sigmoid(self.h)

        if sample:
            # Sample random values
            sampled = cm.empty((v.shape[0], self.output_dim))
            sampled.fill_with_rand()
            # Sample values of hiddens
            sampled.less_than(self.h, self.h)

    def _sample_v(self, h, x, sample=False, x_is_bias=False):
        # updates self.v
        if x_is_bias: # Bias is precalculated
            self.v.assign(x)
        else:
            cm.dot(x, self.ag, self.v)
        self.v.add_dot(h, self.wg.T)

        # This is a 100 times faster than calling 'add_row_vec' to add biases.
        # Update: this might only be true for a cheap gpu.
        ones_cut = self._ones.get_col_slice(0, h.shape[0])
        self.v.add_dot(ones_cut.T, self.bvg)
        self.v.add_row_vec(self.bvg)

        if not self._gaussian:
            self.v.apply_sigmoid(self.v)
            if sample:
                # Sample random values in self.dbv
                sampled = cm.empty((self.v.shape[0], self.output_dim))
                sampled.fill_with_rand()
                # Sample values of visibles
                sampled.less_than(self.v, self.v)


    def train(self, x, n_updates=1, epsilon=0.1, decay=0., momentum=0.):
        """Update the parameters according to the input 'v' and context 'x'.
        The training is performed using Contrastive Divergence (CD).

            - v: a binary matrix having different variables on different columns and observations on the rows.
            - x: a matrix having different variables on different columns and observations on the rows.
            - n_updates: number of CD iterations. Default value: 1
            - epsilon: learning rate. Default value: 0.1
            - decay: weight decay term. Default value: 0.
            - momentum: momentum term. Default value: 0.
        """

        if not self.is_training():
            errstr = "The training phase has already finished."
            raise mdp.TrainingFinishedException(errstr)

        self._train_phase_started = True
        self._train(x, n_updates, epsilon, decay, momentum)

    def get_CD_gradient(self, x, n_updates=1):
        """Use Gibbs sampling to estimate the contrastive divergence gradient.

            - x: a cuda matrix having different variables on different columns and observations on the rows (context)
            - n_updates: number of CD iterations. Default value: 1

        Returns a tuple (dw, dbv, dbh, da, db) that contains the gradients of the
        weights and the biases of the visibles and the hidden respectively and
        the autoregressive gradients da and db.


        This is not the true gradient anymore as I didn't explicitly divide by
        n for the gradients that are based on sums over n datapoints.
        """

        # useful quantities
        n = x.shape[0]

        w, a, b, bv, bh = self.wg, self.ag, self.bg, self.bvg, self.bhg

        # Pre-calculate dynamic biases.
        dynamic_h = cm.empty((n, self.output_dim))
        dynamic_v = cm.empty((n, self.visible_dim))

        cm.dot(x, self.ag, dynamic_v)
        cm.dot(x, self.bg, dynamic_h)

        # first update of the hidden units for the data term
        self._sample_h(self.v, dynamic_h, sample=False, x_is_bias=True)
        # n updates of both v and h for the model term

        # TODO: I set things back to sutskever's way of sampling but should
        # really compare it to Ben's method some time.
        self.h_data = cm.empty(self.h.shape)
        self.v_data = cm.empty(self.v.shape)
        self.h_data.assign(self.h)
        self.v_data.assign(self.v)
        for i in range(n_updates):
            self._stochastic_h()
            self._sample_v(self.h, dynamic_v, x_is_bias=True)
            self._sample_h(self.v, dynamic_h, sample=False, x_is_bias=True)

        # Is preallocating really that "bad" for for example data_term?
        # find dw
        dw = cm.empty(self.w.shape)
        cm.dot(self.v_data.T, self.h_data, dw)
        dw.subtract_dot(self.v.T, self.h)

        # find da
        temp = cm.empty(self.v.shape) # TODO: perhaps this is inefficient...
        da = cm.empty(self.a.shape)
        self.v_data.subtract(self.v, temp)
        cm.dot(x.T, temp, da)

        # find db
        temp = cm.empty(self.h.shape) # TODO: perhaps this is inefficient...
        db = cm.empty(self.b.shape)
        self.h_data.subtract(self.h, temp)
        cm.dot(x.T, temp, db)

        # find dbv
        dbv = cm.empty((1, self.visible_dim))
        self.v_data.sum(axis=0, target=dbv)
        dbv.add_sums(self.v, axis=0, mult= -1.0) # Subtract sum

        # find dbh
        dbh = cm.empty((1, self.output_dim))
        self.h_data.sum(axis=0, target=dbh)
        dbh.add_sums(self.h, axis=0, mult= -1.0) # Subtract sum

        return (dw, dbv, dbh, da, db)

    def _train(self, x, n_updates=1, epsilon=0.1, decay=0., momentum=0.):
        """Update the parameters according to the input 'v' and context 'x'.
        The training is performed using Contrastive Divergence (CD).

            - x: a cuda matrix having different variables on different columns and observations on the rows.
            - n_updates: number of CD iterations. Default value: 1
            - epsilon: learning rate. Default value: 0.1
            - decay: weight decay term. Default value: 0.
            - momentum: momentum term. Default value: 0.
        """
        if not self._initialized:
            self._init_weights()


        # useful quantities
        n = x.shape[0]
        w, a, b, bv, bh = self.wg, self.ag, self.bg, self.bvg, self.bhg

        # old gradients for momentum term
        dw, da, db, dbv, dbh = self.mwg, self.mag, self.mbg, self.mbvg, self.mbhg

        # get the gradient
        dwt, dbvt, dbht, dat, dbt = self.get_CD_gradient(x, n_updates)


        # update w
        dw.mult(momentum)
        dw.add_mult(dwt, epsilon / n)
        dw.add_mult(w, -decay)
        w.add(dw)

        # update a
        da.mult(momentum)
        da.add_mult(dat, epsilon / n)
        da.add_mult(a, -decay)
        a.add(da)

        # update b
        db.mult(momentum)
        db.add_mult(dbt, epsilon / n)
        db.add_mult(b, -decay)
        b.add(db)

        # update bv
        dbv.mult(momentum)
        dbv.add_mult(dbvt, epsilon / n)
        bv.add(dbv)

        # update bh
        dbh.mult(momentum)
        dbh.add_mult(dbht, epsilon / n)
        bh.add(dbh)

    def _stochastic_h(self):
        sampled = cm.empty((self.h.shape[0], self.output_dim))
        sampled.fill_with_rand()
        # Sample values of hiddens
        sampled.less_than(self.h, self.h)

class CUDATRMNode(ERBMNode):
    """This model is essentially a crbm with reservoir states as context
    variables but it hold its own reservoir object and is able to train its
    weights using BPTT. So it is not exactly the TRM from Ben's paper.
    """

    def __init__(self, hidden_dim, visible_dim=None, context_dim=None,
                 gaussian=False, dtype=None, max_batch_size=500):
        """
        Arguments:
            - hidden_dim: number of hidden variables
            - visible_dim: number of observed variables
            - context_dim: number of context variables
            - gaussian: use gaussian visible units (default is binary)
            - max_batch_size: number of datapoints to process simultaneously

        The max batch size is required to be able to optimize the adding of
        bias vectors by preallocating a vector of ones and taking the outer
        product with the bias vector. This is faster than directly adding it
        somehow.
        """
        super(RBMNode, self).__init__(hidden_dim, visible_dim + context_dim, dtype)
        self._input_dim = visible_dim + context_dim
        self._output_dim = hidden_dim

        self.context_dim = context_dim
        self.visible_dim = visible_dim
        self._initialized = False
        self._ones = cm.empty((1, max_batch_size))
        self._ones.assign(1)

        self._gaussian = gaussian
        # TODO: Should probably use **kwargs for these arguments and figure out
        # whether to keep the leak_rate and bias in there for performance
        # reasons.
        self.reservoir = Oger.nodes.CUDAReservoirNode(visible_dim, context_dim,
                                                   spectral_radius=.01,
                                                   leak_rate=1,
                                                   input_scaling=.001)

    def _init_weights(self):
        # weights and biases are initialized to small random values to
        # break the symmetry that might lead to degenerate solutions during
        # learning
        self._initialized = True

        # undirected weights
        self.w = self._refcast(randn(self.visible_dim, self.output_dim) * 0.01)
        # context to visible weights
        self.a = self._refcast(randn(self.context_dim, self.visible_dim) * 0.01)
        # context to hidden weights
        self.b = self._refcast(randn(self.context_dim , self.output_dim) * 0.01)
        # bias on the visible (input) units
        self.bv = self._refcast(randn(1, self.visible_dim) * 0.01)
        # bias on the hidden (output) units
        self.bh = self._refcast(randn(1, self.output_dim) * 0.01)

        # Copy to GPU

        self.wg = cm.CUDAMatrix(self.w)
        self.ag = cm.CUDAMatrix(self.a)
        self.bg = cm.CUDAMatrix(self.b)
        self.bvg = cm.CUDAMatrix(self.bv)
        self.bhg = cm.CUDAMatrix(self.bh)

        # Momentum terms

        self.mwg = cm.CUDAMatrix(np.zeros(self.w.shape))
        self.mag = cm.CUDAMatrix(np.zeros(self.a.shape))
        self.mbg = cm.CUDAMatrix(np.zeros(self.b.shape))
        self.mbvg = cm.CUDAMatrix(np.zeros(self.bv.shape))
        self.mbhg = cm.CUDAMatrix(np.zeros(self.bh.shape))
        self.mw_resg = cm.CUDAMatrix(np.zeros(self.reservoir.w.shape))
        self.mw_ing = cm.CUDAMatrix(np.zeros(self.reservoir.w_in.shape))

    def _split_data(self, x):
        # split data into visibles and context respectively.
        # TODO: Since this is probably more expensive than assigning it will be
        # perhaps be worth it to remove this whole splitting thing or to make
        # it store the values to preallocated memory instead.
        return cm.CUDAMatrix(x[:, :self.visible_dim]), cm.CUDAMatrix(x[:, self.visible_dim:])

    def _stochastic_h(self):
        sampled = cm.empty((self.h.shape[0], self.output_dim))
        sampled.fill_with_rand()
        # Sample values of hiddens
        sampled.less_than(self.h, self.h)

    def _sample_h(self, v, x, sample=False, x_is_bias=False):
        # updates self.h
        #

        self.h = cm.empty((v.shape[0], self.output_dim))

        if x_is_bias: # Bias is precalculated
            self.h.assign(x)
        else:
            cm.dot(x, self.bg, self.h)

        self.h.add_dot(v, self.wg)

        # This is a 100 times faster than calling 'add_row_vec' to add biases.
        ones_cut = self._ones.get_col_slice(0, v.shape[0])
        self.h.add_dot(ones_cut.T, self.bhg)

        self.h.apply_sigmoid2(self.h)

        if sample:
            # Sample random values
            sampled = cm.empty((v.shape[0], self.output_dim))
            sampled.fill_with_rand()
            # Sample values of hiddens
            sampled.less_than(self.h, self.h)

    def _sample_v(self, h, x, sample=False, x_is_bias=False):
        # updates self.v
        if x_is_bias: # Bias is precalculated
            self.v.assign(x)
        else:
            cm.dot(x, self.ag, self.v)
        self.v.add_dot(h, self.wg.T)

        # This is a 100 times faster than calling 'add_row_vec' to add biases.
        ones_cut = self._ones.get_col_slice(0, h.shape[0])
        self.v.add_dot(ones_cut.T, self.bvg)
        self.v.add_row_vec(self.bvg)

        if not self._gaussian:
            self.v.apply_sigmoid2(self.v)
            if sample:
                # Sample random values in self.dbv
                sampled = cm.empty((self.v.shape[0], self.output_dim))
                sampled.fill_with_rand()
                # Sample values of visibles
                sampled.less_than(self.v, self.v)


    def train(self, x, n_updates=1, epsilon=0.1, decay=0., momentum=0.):
        """Update the parameters according to the input 'v' and context 'x'.
        The training is performed using Contrastive Divergence (CD).

            - v: a binary matrix having different variables on different columns and observations on the rows.
            - x: a matrix having different variables on different columns and observations on the rows.
            - n_updates: number of CD iterations. Default value: 1
            - epsilon: learning rate. Default value: 0.1
            - decay: weight decay term. Default value: 0.
            - momentum: momentum term. Default value: 0.
        """

        if not self.is_training():
            errstr = "The training phase has already finished."
            raise mdp.TrainingFinishedException(errstr)

        self._train_phase_started = True
        self._train(x, n_updates, epsilon, decay, momentum)

    def get_gradient(self, x, n_updates=1):
        """Use Gibbs sampling to estimate the contrastive divergence gradient.

            - x: a cuda matrix having different variables on different columns and observations on the rows (context)
            - n_updates: number of CD iterations. Default value: 1

        Returns a tuple (dw, dbv, dbh, da, db) that contains the gradients of the
        weights and the biases of the visibles and the hidden respectively and
        the autoregressive gradients da and db.


        This is not the true gradient anymore as I didn't explicitly divide by
        n for the gradients that are based on sums over n datapoints.

        The BPTT gradient with respect to the reservoir recurrent and input
        weight is computed as well.
        """

        # useful quantities
        n = x.shape[0]

        w, a, b, bv, bh = self.wg, self.ag, self.bg, self.bvg, self.bhg

        # Pre-calculate dynamic biases.
        dynamic_h = cm.empty((n, self.output_dim))
        dynamic_v = cm.empty((n, self.visible_dim))

        cm.dot(x, self.ag, dynamic_v)
        cm.dot(x, self.bg, dynamic_h)

        # first update of the hidden units for the data term
        self._sample_h(self.v, dynamic_h, sample=False, x_is_bias=True)
        # n updates of both v and h for the model term
        self.h_data = cm.empty(self.h.shape)
        self.v_data = cm.empty(self.v.shape)
        self.h_data.assign(self.h)
        self.v_data.assign(self.v)
        #self._sample_h(self.v, dynamic_h, sample=True, x_is_bias=True)
        for i in range(n_updates):
            self._stochastic_h()
            self._sample_v(self.h, dynamic_v, x_is_bias=True)
            self._sample_h(self.v, dynamic_h, sample=False, x_is_bias=True)


        # Is preallocating really that "bad" for for example data_term?
        # find dw
        dw = cm.empty(self.w.shape)
        cm.dot(self.v_data.T, self.h_data, dw)
        dw.subtract_dot(self.v.T, self.h)

        # find da
        d_v = cm.empty(self.v.shape) # TODO: perhaps this is inefficient...
        da = cm.empty(self.a.shape)
        self.v_data.subtract(self.v, d_v)
        cm.dot(x.T, d_v, da)

        # find db
        d_h = cm.empty(self.h.shape) # TODO: perhaps this is inefficient...
        # TODO: I should probably just compute the gradient with respect to the
        # biases once and use that for both updating matrix b and the biases
        # itself.
        db = cm.empty(self.b.shape)
        self.h_data.subtract(self.h, d_h)
        cm.dot(x.T, d_h, db)

        # find dbv
        dbv = cm.empty((1, self.visible_dim))
        self.v_data.sum(axis=0, target=dbv)
        dbv.add_sums(self.v, axis=0, mult= -1.0) # Subtract sum

        # find dbh
        dbh = cm.empty((1, self.output_dim))
        self.h_data.sum(axis=0, target=dbh)
        dbh.add_sums(self.h, axis=0, mult= -1.0) # Subtract sum

        #### BPTT code ####
        # TODO: Some of the computations above should be combined with the
        # gradient calculation here.

        d_reservoir = cm.empty((self.context_dim, n))

        # Do some transposes because get_col_slice is faster than get_row_slice.
        x_T = x.transpose()
        d_h_T = d_h.transpose()
        d_v_T = d_v.transpose()

        # Pre-calculate the tanh derivatives
        dtanh = cm.empty(x_T.shape)
        x_T.apply_dtanh(target=dtanh)


        # Last state gets no gradient information from the future
        drt = d_reservoir.get_col_slice(n - 1, n)
        drt.assign(0)

        # Main BPTT loop
        for i in range(n - 1, 0, -1):
            drt = d_reservoir.get_col_slice(i, i + 1)
            dr_pre_t = d_reservoir.get_col_slice(i - 1, i)
            d_vt = d_v_T.get_col_slice(i, i + 1)
            d_ht = d_h_T.get_col_slice(i, i + 1)

            # Add visible component
            # TODO: I could actually pre-calculate this outside the loop
            drt.add_dot(self.ag, d_vt)

            # Add hidden component
            drt.add_dot(self.bg, d_ht)

            # Mult with derivative
            drt.mult(dtanh.get_col_slice(i, i + 1))

            # Backpropagate
            cm.dot(self.reservoir.w.T, drt, dr_pre_t)

        d_vt = d_v_T.get_col_slice(0, 1)
        d_ht = d_h_T.get_col_slice(0, 1)
        dr_pre_t = d_reservoir.get_col_slice(0, 1)

        # Add visible component
        dr_pre_t.add_dot(self.ag, d_vt)

        # Add hidden component
        dr_pre_t.add_dot(self.bg, d_ht)

        # Mult with derivative
        dr_pre_t.mult(dtanh.get_col_slice(0, 1))

        # Compute weight derivatives
        dw_res = cm.empty(self.reservoir.w.shape)
        dw_res_in = cm.empty(self.reservoir.w_in.shape)

        # dw_res <- d_reservoir * x(t-1)
        # The first state has obviously no previous state so we can ignore it.
        cm.dot(d_reservoir.get_col_slice(1, n), x_T.get_col_slice(0, n - 1).T,
                                                               target=dw_res)
        # dw_res_in <- d_reservoir * v
        cm.dot(d_reservoir, self.v_data, target=dw_res_in)

        ###################


        return (dw, dbv, dbh, da, db, dw_res, dw_res_in)

    def _train(self, x, n_updates=1, epsilon=0.1, decay=0., momentum=0.):
        """Update the parameters according to the input 'v' and context 'x'.
        The training is performed using Contrastive Divergence (CD).

            - x: a cuda matrix having different variables on different columns and observations on the rows.
            - n_updates: number of CD iterations. Default value: 1
            - epsilon: learning rate. Default value: 0.1
            - decay: weight decay term. Default value: 0.
            - momentum: momentum term. Default value: 0.
        """
        if not self._initialized:
            self._init_weights()


        # useful quantities
        n = x.shape[0]
        w, a, b, bv, bh, w_res, w_res_in = (self.wg, self.ag, self.bg, self.bvg,
                                            self.bhg, self.reservoir.w,
                                            self.reservoir.w_in)

        # old gradients for momentum term
        dw, da, db, dbv, dbh, dw_res, dw_res_in = (self.mwg, self.mag,
                                                   self.mbg, self.mbvg,
                                                   self.mbhg, self.mw_resg,
                                                   self.mw_ing)

        # get the gradient
        dwt, dbvt, dbht, dat, dbt, dw_rest, dw_res_int = self.get_gradient(x,
                                                                     n_updates)

        # update w
        dw.mult(momentum)
        dw.add_mult(dwt, epsilon / n)
        dw.add_mult(w, -decay)
        w.add(dw)

        # update a
        da.mult(momentum)
        da.add_mult(dat, epsilon / n)
        da.add_mult(a, -decay)
        a.add(da)

        # update b
        db.mult(momentum)
        db.add_mult(dbt, epsilon / n)
        db.add_mult(b, -decay)
        b.add(db)

        # update bv
        dbv.mult(momentum)
        dbv.add_mult(dbvt, epsilon / n)
        bv.add(dbv)

        # update bh
        dbh.mult(momentum)
        dbh.add_mult(dbht, epsilon / n)
        bh.add(dbh)

        # update w_res
        dw_res.mult(momentum)
        dw_res.add_mult(dw_rest, epsilon / n)
        #dw_res.add_mult(dw_rest, epsilon)
        dw_res.add_mult(w_res, -decay)
        w_res.add(dw_res)

        # update w_res_in
        dw_res_in.mult(momentum)
        dw_res_in.add_mult(dw_res_int, epsilon / n)
        #dw_res_in.add_mult(dw_res_int, epsilon)
        dw_res_in.add_mult(w_res_in, -decay)
        w_res_in.add(dw_res_in)
