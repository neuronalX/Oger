import mdp
import Oger

if __name__ == "__main__":
    n_inputs = 1
    system_order = 30
    washout = 30

    inputs, outputs = Oger.datasets.narma30(sample_len=1000)
    #data = Dataset({'x':inputs, 'y':outputs})
    data = [[], zip(inputs, outputs)]

    # construct individual nodes
    reservoir = Oger.nodes.ReservoirNode(output_dim=100, input_scaling=0.1)
    readout = Oger.nodes.RidgeRegressionNode()

    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout])
    
    print "Simple training and testing (one fold, i.e. no cross-validation), training_fraction = 0.5."
    print "cross_validate_function = crossvalidation.train_test_only"
    errors = Oger.evaluation.validate(data, flow, Oger.utils.nrmse, cross_validate_function=Oger.evaluation.train_test_only, training_fraction=0.5)
    print errors
       
    print "5-fold cross-validation"
    print "cross_validate_function = crossvalidation.cross_validate"
    errors = Oger.evaluation.validate(data, flow, Oger.utils.nrmse, n_folds=5)
    print errors
    print

    print "Leave-one-out cross-validation"
    errors = Oger.evaluation.validate(data, flow, Oger.utils.nrmse, cross_validate_function=Oger.evaluation.leave_one_out)
    print errors
