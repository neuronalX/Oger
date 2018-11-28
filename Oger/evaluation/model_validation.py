import mdp.utils
import Oger
from copy import deepcopy

def train_test_only(n_samples, training_fraction, random=True):
    '''
    train_test_only(n_samples, training_fraction, random) -> train_indices, test_indices
    
    Return indices to do simple training and testing. Only one fold is created, 
    using training_fraction of the dataset for training and the rest for testing.
    The samples are selected randomly by default but this can be disabled.
    Two lists are returned, with 1 element each.
        - train_indices contains the indices of the dataset used for training
        - test_indices contains the indices of the dataset used for testing
    '''
    if random:
        # Shuffle the samples randomly
        perm = mdp.numx.random.permutation(n_samples)
    else:
        perm = range(n_samples)
    # Take a fraction training_fraction for training
    train_indices = [perm[0:int(round(n_samples * training_fraction))]]
    # Use the rest for testing
    test_indices = mdp.numx.array([mdp.numx.setdiff1d(perm, train_indices[-1])])
    return train_indices, test_indices


def leave_one_out(n_samples):
    '''
    leave_one_out(n_samples) -> train_indices, test_indices
    
    Return indices to do leave-one-out cross-validation. Per fold, one example is used for testing and the rest for training.
    Two lists are returned, with n_samples elements each.
        - train_indices contains the indices of the dataset used for training
        - test_indices contains the indices of the dataset used for testing
    '''
    train_indices, test_indices = [], []
    all_samples = range(n_samples)
    # Loop over all sample indices, using each one for testing once
    for test_index in all_samples:
        test_indices.append(mdp.numx.array([test_index]))
        train_indices.append(mdp.numx.setdiff1d(all_samples, [test_index]))
    return train_indices, test_indices


def n_fold_random(n_samples, n_folds):
    '''
    n_fold_random(n_samples, n_folds) -> train_indices, test_indices
    
    Return indices to do random n_fold cross_validation. Two lists are returned, with n_folds elements each.
        - train_indices contains the indices of the dataset used for training
        - test_indices contains the indices of the dataset used for testing
    '''

    if n_folds <= 1:
        raise Exception('Number of folds should be larger than one.')

    if n_folds > n_samples:
        raise Exception('Number of folds (%d) cannot be larger than the number of samples (%d).'%(n_folds, n_samples))


    # Create random permutation of number of samples
    randperm = mdp.numx.random.permutation(n_samples)
    train_indices, test_indices = [], []
    foldsize = mdp.numx.floor(float(n_samples) / n_folds)

    for fold in range(n_folds):
        # Select the sample indices used for testing
        test_indices.append(randperm[fold * foldsize:foldsize * (fold + 1)])
        # Select the rest for training
        train_indices.append(mdp.numx.array(mdp.numx.setdiff1d(randperm, test_indices[-1])))
    return train_indices, test_indices


def validate(data, flow, error_measure, cross_validate_function=n_fold_random, progress=True, internal_gridsearch_parameters=None, validation_suffix_flow=None, error_aggregation_function=mdp.numx.mean, *args, **kwargs):
    '''
    validate(data, flow, error_measure, cross_validate_function=n_fold_random, progress=True, *args, **kwargs) -> test_errors
    
    Perform  cross-validation on a flow, return the validation test_error for each fold. For every flow, the flow.train() method is called
    on the training data, and the flow.execute() function is called on the test data.
        - inputs and outputs are lists of arrays
        - flow is an mdp.Flow
        - error_measure is a function which should return a scalar
        - cross_validate_function is a function which determines the type of cross-validation
          Possible values are:
              - n_fold_random (default): split dataset in n_folds parts, for each fold train on n_folds-1 parts and test on the remainder
              - leave_one_out : do cross-validation with len(inputs) folds, using a single sample for testing in each fold and the rest for training
              - train_test_only : divide dataset into train- and testset, using training_fraction as the fraction of samples used for training
        - progress is a boolean to enable a progress bar (default True)
    '''
    test_error = []

    for validate_gen_result in validate_gen(data, flow, cross_validate_function, internal_gridsearch_parameters, error_measure, progress, validation_suffix_flow, *args, **kwargs):
        if internal_gridsearch_parameters is not None:
            f_copy, _, test_sample_list, f_untrained = validate_gen_result
        else:
            f_copy, _, test_sample_list = validate_gen_result
        # Empty list to store test errors for current fold
        fold_error = []

        # Add the suffix flow if requested
        if validation_suffix_flow is not None:
            f_val = f_copy + validation_suffix_flow
        else:
            f_val = f_copy

        # test on all test samples
        for test_sample in test_sample_list:
            if len(test_sample[-1][0]) == 1:
                t_i = test_sample[-1][0][0]
                t_t = test_sample[-1][0][0]
            else:
                t_i = test_sample[-1][0][0]
                t_t = test_sample[-1][0][-1]
            test = error_measure(f_val(t_i), t_t)
            fold_error.append(test)

        test_error.append(error_aggregation_function(fold_error))


    if internal_gridsearch_parameters is not None:
        return test_error, f_untrained
    else:
        return test_error



def validate_gen(data, flow, cross_validate_function=n_fold_random, internal_gridsearch_parameters=None, error_measure=None, progress=True, validation_suffix_flow=None, *args, **kwargs):
    '''
    validate_gen(data, flow, cross_validate_function=n_fold_random, progress=True, validation_suffix_flow=None, *args, **kwargs) -> test_output
    
    This generator performs cross-validation on a flow. It splits the data into folds according to the supplied cross_validate_function, and then for each fold, trains the flow and yields the trained flow, the training data, and a list of test data samples.
    Use it like this:
    >>> for flow, train_data, test_sample_list in validate_gen(...):
    ...
    See 'validate' for more information about the function signature.
    '''
    # Get the number of samples 
    n_samples = mdp.numx.amax(map(len, data))

    # Get the indices of the training and testing samples for each fold by calling the 
    # cross_validate_function hook
    train_samples, test_samples = cross_validate_function(n_samples, *args, **kwargs)

    if progress:
        print "Performing cross-validation using " + cross_validate_function.__name__
        iteration = mdp.utils.progressinfo(range(len(train_samples)), style='timer')
    else:
        iteration = range(len(train_samples))

    for fold in iteration:
        # Get the training data from the whole data set
        train_data = data_subset(data, train_samples[fold])

        # Copy the flow so we can re-train it for every fold
        f_copy = deepcopy(flow)

        if internal_gridsearch_parameters is not None:
            # We turn off parallelization for internal gridsearch
            active_extensions = mdp.get_active_extensions()
            mdp.deactivate_extension('parallel')
            opt = Oger.evaluation.Optimizer(internal_gridsearch_parameters, error_measure)
            opt.grid_search(train_data, flow, cross_validate_function=cross_validate_function, progress=False, validation_suffix_flow=validation_suffix_flow)
            f_copy = opt.get_optimal_flow()
            # Reactivate the parallel extension if needed
            mdp.activate_extensions(active_extensions)

        test_sample_list = [data_subset(data, [k]) for k in test_samples[fold]]

        if internal_gridsearch_parameters is not None:
            f_untrained = deepcopy(f_copy)

        f_copy.train(train_data)

        # Collect probe data if it is present
        for node in f_copy.flow:
            if hasattr(node, 'probe_data'):
                flow[f_copy.flow.index(node)].probe_data = node.probe_data

        if internal_gridsearch_parameters is not None:
            yield (f_copy, train_data, test_sample_list, f_untrained) # collect everything needed to evaluate this fold, including the untrained flow and return it.
        else:
            yield (f_copy, train_data, test_sample_list) # collect everything needed to evaluate this fold and return it.

def data_subset(data, data_indices):
    '''
    data_subset(data, data_indices) -> data_subset
    
    Return a subset of the examples in data given by data_indices.
    Data_indices can be a slice, a list of scalars or a numpy array.
    '''
    n_nodes = len(data)
    subset = []
    #print data_indices
    #reprint type(data_indices)
    for node in range(n_nodes):
        if isinstance(data_indices, slice) or isinstance(data_indices, int):
            subset.append(data[node].__getitem__(data_indices))
        else:
            tmp_data = []
            if not data[node] == []:
                for data_index in data_indices:
                    tmp_data.append(data[node][data_index])
            else:
                tmp_data.extend([])
            subset.append(tmp_data)
    return subset
