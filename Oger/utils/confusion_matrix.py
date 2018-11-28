import numpy as np

# Tools for working with confusion matrices, which are useful for the evaluation
# of classification tasks.
#
# The centerpiece is the ConfusionMatrix class, which encapsulates a rectangular
# numpy array. A number of error measures are implemented as properties. Some
# methods to construct this matrix from test data are also provided.
#
# The BinaryConfusionMatrix class extends the ConfusionMatrix class with some
# functionality that is specific to binary classification problems.
#
# The plot_conf function provides a quick way of visualising a confusion matrix.
# Balanced (row-normalised) matrices give the best results.
# 
#
# Usage example:
# 
#   >>> c = ConfusionMatrix.from_samples(num_classes, predictions, targets, labels)
#   >>> print "BER: %f" % c.ber
#   >>> print "Error rates per class: %s" % repr(c.error_rates) 
#   >>> plot_conf(c.balance())
#
# Note: a ConfusionMatrix wraps a numpy ARRAY, not a numpy MATRIX.

class ConfusionMatrix(object):
    """
    This class represents a confusion matrix, and encapsulates a rectangular 2D
    numpy array. It can be constructed from such an array by using the constructor,
    or from a set of predictions and target labels through the from_data and
    from_samples class methods.
    
    The class implements a number of commonly used accuracy and error measures
    based on confusion matrices as properties.
    """
    def __init__(self, input_array, labels=None):
        arr = np.asarray(input_array).astype(float)
        self._assert_dimensions(arr)
        self._arr = arr
        self._init_labels(len(arr), labels)
        
    def _init_labels(self, num_classes, labels):
        if labels is not None:
            self._assert_labels(num_classes, labels)
            self.labels = labels
        else:
            self.labels = range(0, num_classes) # default to labels 0, 1, 2, ..., N-1
        
    def _check_ndim(self, arr):
        # check if it is a 2D array
        return arr.ndim == 2
        
    def _check_shape(self, arr):
        # check if it is a rectangular array
        return arr.shape[0] == arr.shape[1]
        
    def _check_dimensions(self, arr):
        return self._check_ndim(arr) and self._check_shape(arr)
        
    def _assert_dimensions(self, arr): # like _check_dimensions, but with exceptions.
        if not self._check_ndim(arr):
            raise RuntimeError("ConfusionMatrix should be 2-dimensional - ndim is %d" % arr.ndim)
        elif not self._check_shape(arr):
            raise RuntimeError("ConfusionMatrix should be rectangular - shape is %dx%d" % arr.shape)
            
    def _check_labels_count(self, num_classes, labels):
        return len(labels) == num_classes
        
    def _check_labels_uniqueness(self, labels):
        return len(np.unique(labels)) == len(labels)
        
    def _check_labels(self, num_classes, labels):
        return self._check_labels_count(num_classes, labels) and self._check_labels_uniqueness(labels)
        
    def _assert_labels(self, num_classes, labels):
        if not self._check_labels_count(num_classes, labels):
            raise RuntimeError("Number of class labels does not equal number of classes - number of labels is %d, number of classes is %d" % (len(labels), num_classes)) # TODO: check happens very late, too late in case there are fewer labels than classes (then another, more obscure error will already have been raised). Resolving this issue isn't easy without code duplication.
        elif not self._check_labels_uniqueness(labels):
            raise RuntimeError("Class labels are not unique - labels are %s" % repr(labels))
            
    def __add__(self, other):
        return ConfusionMatrix(np.asarray(self) + np.asarray(other))
        
    def __repr__(self):
        return "ConfusionMatrix(%s)" % repr(self._arr)
            
    def __array__(self):
        return self._arr
        
    asarray = __array__
    array = __array__


    @classmethod
    def from_data(cls, num_classes, output_classes, target_classes, labels=None):
        """
        Create a confusion matrix from a set of target class labels and corresponding predictions.

        num_classes: the number of classes (also the size of the resulting matrix).
                     
        output_classes: predictions (MDP-style numpy array: 1 label per row)
        
        target_classes: target class labels (MDP-style numpy array: 1 label per row)
        
        labels: the class labels that were used. Defaults to 0, 1, 2, ..., num_classes-1.
        """
        if output_classes.shape != target_classes.shape:
            raise RuntimeError("Output and target data should have the same shape")
            
        if labels is None:
            labels = range(0, num_classes)
        
        conf = np.zeros((num_classes, num_classes))
        for output_label, target_label in zip(output_classes, target_classes):
            o = labels.index(output_label)
            t = labels.index(target_label)
            conf[t, o] += 1 # targets = rows, predictions = columns
           
        return ConfusionMatrix(conf, labels)
        
    @classmethod
    def from_samples(cls, num_classes, output_classes_list, target_classes_list, normalisation=True, labels=None):
        """
        Create a confusion matrix from a set of samples, where each sample is equally weighted
        irrespective of its length. ConfusionMatrix.from_data is applied on each sample. The
        resulting matrices are then normalised, and subsequently averaged (which results in a
        normalised matrix).
        
        num_classes: see ConfusionMatrix.from_data
        
        output_classes_list: iterable containing prediction data (see ConfusionMatrix.from_data for the correct format).
                            
        target_classes_list: iterable containing target class labels (see ConfusionMatrix.from_data for the correct format)
                             
        normalisation: defaults to True. If this parameter is set to False, no normalisation occurs and the confusion matrices corresponding to the samples are just added together.
                       
        labels: the class labels that were used. Defaults to 0, 1, 2, ..., num_classes-1.
        """
        matrices = [] # one matrix per sample
        for x, y in zip(output_classes_list, target_classes_list):
            nc = ConfusionMatrix.from_data(num_classes, x, y, labels)
            if normalisation:
                nc = nc.normalise()
                
            matrices.append(nc.asarray())
            
        if normalisation:
            matrix = np.array(matrices).mean(0)
        else:
            matrix = np.array(matrices).sum(0)
        
        return ConfusionMatrix(matrix)

    @classmethod
    def error_measure(cls, name, num_classes, labels=None, from_samples=False, normalisation=True):
        """
        Returns a function that constructs a confusion matrix and then computes the desired metric.

        For example:
            >>> ber = ConfusionMatrix.error_measure('ber', 2)
            >>> print ber(input_signal, target_signal)
        
        If from_samples is set to True, the returned function will use from_samples instead of
        from_data to construct the confusion matrix.
        """
            
        def f(input_signal, target_signal):
            if from_samples:
                cm = cls.from_samples(num_classes, input_signal, target_signal, labels=labels,
                                      normalisation=normalisation)
            else:
                cm = cls.from_data(num_classes, input_signal, target_signal, labels=labels)            
            return getattr(cm, name)
            
        return f
    
    def normalise(self):
        """
        Returns a confusion matrix in which the sum of all elements is 1.
        This is useful for percentage information, and for adding matrices together with equal weights.
        """
        return ConfusionMatrix(self._arr / self.total)
    
    def normalise_per_class(self):
        """
        Returns a balanced confusion matrix, in which all rows sum to 1.
        This is particularly useful for visualisation.
        """
        # in a CM normalised per class, classification accuracy and BCR are equivalent.
        truth = np.atleast_2d(self.truth).T
        normalised_arr = self._arr / truth # will have NaNs where truth == 0
        normalised_arr[np.isnan(normalised_arr)] = 0
        return ConfusionMatrix(normalised_arr)
        
    balance = normalise_per_class # in a CM normalised per class, accuracy and BCR,
    # error rate and BER, ... are equivalent; hence the name of this alias.
    # in a balanced confusion matrix, each class has the same weight.
    
    
    @property    
    def num_classes(self):
        """
        The number of different classes in this matrix, which also corresponds to its size.
        """      
        return len(self._arr)
        
    def __len__(self):
        return self.num_classes
    
    @property    
    def total(self):
        """
        The total amount of observations, corresponding to the sum of all elements of the matrix.
        For normalised matrices, this is always 1.
        """
        return self._arr.sum()
        
    @property
    def correct(self):
        """
        The amount of correct detections for each class, corresponding to the diagonal elements
        of the matrix.
        """
        return self._arr.diagonal()
        
    @property
    def incorrect(self):
        """
        The amount of incorrect detections for each class.
        """
        return self.truth - self.correct
    
    @property    
    def detections(self): # detections are column sums
        """
        The number of detections for each class, corresponding to the sums of the columns.
        """
        return self._arr.sum(0)
    
    @property    
    def ground_truth(self): # number of values of each class (row sums)
        """
        The number of entries for each class (ground truth), corresponding to the sums
        of the rows.
        """
        return self._arr.sum(1)
        
    truth = ground_truth
    
        
    def binary(self):
        """
        returns a list of binary confusion matrices for each class,
        where that class is considered 'positive' and the others are 'negative'.
        The result is returned as a list of 2x2 BinaryConfusionMatrices.
        """
        binary_confs = []

        for i in range(self.num_classes):
            # for each class, compute true positives, true negatives, false positives, false negatives
            not_i = range(0, i) + range(i+1, self.num_classes)
            tp = self._arr[i, i]
            fn = self._arr[i, not_i].sum()
            fp = self._arr[not_i, i].sum()
            
            # this is very dirty, but unfortunately conf[not_i, not_i] does not work as expected.
            tn_rows = not_i*len(not_i)
            tn_cols = list(tn_rows) # make a copy
            tn_cols.sort()
            tn = self._arr[tn_rows, tn_cols].sum()
            
            b = BinaryConfusionMatrix([[tp, fn], [fp, tn]])
            binary_confs.append(b)
            
        return binary_confs
        
    @property
    def classification_rates(self): 
        """
        The relative amount of correct classifications for each class.
        """
        r = self.correct / self.truth
        r[np.isnan(r)] = 0
        return r
    
    @property
    def error_rates(self): # compute the error rates per class
        """
        The relative amount of misclassifications for each class.
        """
        r = self.incorrect / self.truth
        r[np.isnan(r)] = 0
        return r
    
    @property    
    def balanced_error_rate(self):
        """
        The average relative amount of misclassifications, where each class
        is weighed equally. In a balanced matrix, this is the same as the
        classification error rate.
        """
        return self.error_rates.mean() # avg of error rates = BER = 1 - BCR
        
    ber = balanced_error_rate
    
    @property    
    def balanced_classification_rate(self):
        """
        The average relative amount of correct classifications, where each
        class is weighed equally. In a balanced matrix, this is the same as
        the classification accuracy.
        """
        return self.classification_rates.mean() # avg of classification rates = BCR = 1 - BER
        
    bcr = balanced_classification_rate
    
    @property    
    def classification_accuracy(self):
        """
        The relative amount of correct classifications.
        """
        return self.correct.sum() / self.total
        
    accuracy = classification_accuracy
    acc = accuracy
    
    @property
    def classification_error_rate(self):
        """
        The relative amount of misclassifications.
        """
        return self.incorrect.sum() / self.total
        
    error_rate = classification_error_rate
    err = error_rate
    
    

    
class BinaryConfusionMatrix(ConfusionMatrix):
    """
    This class represents a confusion matrix for a binary classification problem,
    in which one class is considered 'positive' and the other 'negative'. It extends
    the ConfusionMatrix class, so it provides all the methods and properties of this
    class. In addition, some error and accuracy measures specific to binary problems
    are provided as properties.
    """
    # additional error measures and functionality for binary confusion matrices,
    # where the first class is considered 'positive' and the second 'negative'.
    def _check_shape(self, arr):
        # check if this is a 2x2 array
        return (arr.shape[0] == 2) and (arr.shape[1] == 2)
        
    def _assert_dimensions(self, arr):
        if not self._check_ndim(arr):
            raise RuntimeError("ConfusionMatrix should be 2-dimensional - ndim is %d" % arr.ndim)
        elif not self._check_shape(arr):
            raise RuntimeError("ConfusionMatrix should have shape 2x2 - shape is %dx%d" % arr.shape)
            

        
    def __add__(self, other): # overridden to make sure that the resulting matrix is also binary
        return BinaryConfusionMatrix(np.asarray(self) + np.asarray(other))
        
    def __repr__(self):
        return "BinaryConfusionMatrix(%s)" % repr(self._arr)
    
    @classmethod
    def from_data(cls, output_classes, target_classes, labels=None):
        # By default '0' is the POSITIVE label, '1' is the NEGATIVE label!
        cf = super(BinaryConfusionMatrix, cls).from_data(2, output_classes, target_classes, labels)        
        return BinaryConfusionMatrix(cf)
        
    @classmethod
    def from_samples(cls, output_classes_list, target_classes_list, labels=None):
        # By default '0' is the POSITIVE label, '1' is the NEGATIVE label!
        cf = super(BinaryConfusionMatrix, cls).from_samples(2, output_classes_list, target_classes_list, labels)
        return BinaryConfusionMatrix(cf)
 
    @classmethod
    def error_measure(cls, name, labels=None, from_samples=False, normalisation=True):
        """
        Returns a function that constructs a confusion matrix and then computes the desired metric.

        For example:
            >>> recall = ConfusionMatrix.error_measure('recall')
            >>> print recall(input_signal, target_signal)
        
        If from_samples is set to True, the returned function will use from_samples instead of
        from_data to construct the confusion matrix.
        """
            
        def f(input_signal, target_signal):
            if from_samples:
                cm = cls.from_samples(input_signal, target_signal, labels=labels,
                                      normalisation=normalisation)
            else:
                cm = cls.from_data(input_signal, target_signal, labels=labels)            
            return getattr(cm, name)
            
        return f
    
    def normalise(self): # overridden to make sure that the resulting matrix is also binary
        """
        Returns a confusion matrix in which the sum of all elements is 1.
        This is useful for percentage information, and for adding matrices together with equal weights.
        """
        return BinaryConfusionMatrix(self._arr / self.total)   
 
 
    def normalise_per_class(self): # overridden to make sure that the resulting matrix is also binary
        """
        Returns a balanced confusion matrix, in which all rows sum to 1.
        This is particularly useful for visualisation.
        """
        # in a CM normalised per class, classification accuracy and BCR are equivalent.
        truth = np.atleast_2d(self.truth).T
        return BinaryConfusionMatrix(self._arr / truth)
        
    balance = normalise_per_class
    
    @property        
    def true_positives(self):
        """
        The amount of true positives (target label 0, predicted label 0).
        """
        return self._arr[0, 0]
    
    @property
    def false_positives(self):
        """
        The amount of false positives (target label 1, predicted label 0).
        """
        return self._arr[1, 0]
    
    @property    
    def true_negatives(self):
        """
        The amount of true negatives (target label 1, predicted label 1).
        """
        return self._arr[1, 1]
    
    @property    
    def false_negatives(self):
        """
        The amount of false negatives (target label 0, predicted label 0).
        """
        return self._arr[0, 1]
    
    tp = true_positives
    fp = false_positives
    tn = true_negatives
    fn = false_negatives
    
    @property
    def positives(self):
        """
        The amount of positive data (label 0).
        """
        return self.tp + self.fn
    
    @property
    def negatives(self):
        """
        The amount of negative data (label 1).
        """
        return self.tn + self.fp
        
    pos = positives
    neg = negatives
        
    @property
    def positive_detections(self):
        """
        The amount of positive detections (label 0).
        """
        return self.tp + self.fp
        
    @property
    def negative_detections(self):
        """
        The amount of negative detections (label 1).
        """
        return self.tn + self.fn
        
    pd = positive_detections
    nd = negative_detections
    
    
    @property    
    def recall(self):
        """
        Recall or sensitivity or true positive rate, defined as the amount of
        correct positive detections, divided by the total amount of positive data.
        """
        return self.tp / self.pos
    
    @property    
    def precision(self):
        """
        Precision, defined as the amount of correct positive detections,
        divided by the total number of positive detections.
        """
        return self.tp / self.pd
        
    r = recall
    p = precision
    
    def f_measure(self, beta=1): # not a property, because the beta parameter can be tuned
        """
        F measure, harmonic mean of recall and precision. The mean can optionally be
        weighted with a parameter beta. beta > 1 increases the importance of recall,
        beta < 1 increases the importance of precision. The default value is 1, resulting
        in the so-called F1-measure.
        """
        return (1 + beta**2) * ( (self.p*self.r) / ((beta**2) * self.p + self.r) )
    
    @property
    def f1(self): # for convenience
        """
        F1 measure (see BinaryConfusionMatrix.f_measure() for more information).
        """
        return self.f_measure(1)
        
    sensitivity = recall
    sens = sensitivity
    
    @property
    def specificity(self):
        """
        Specificity or true negative rate, defined as the amount of correct
        negative detections, divided by the total amount of negative data.
        """
        return self.tn / self.neg
        
    spec = specificity
    true_positive_rate = sensitivity
    true_negative_rate = specificity
    tp_rate = true_positive_rate
    tn_rate = true_negative_rate
    
    @property
    def positive_likelihood(self):
        """
        Positive likelihood, defined as sensitivity / (1 - specificity)
        """
        return self.sens / (1 - self.spec)
    
    @property    
    def negative_likelihood(self):
        """
        Negative likelihood, defined as specificity / (1 - sensitivity)
        """
        return self.spec / (1 - self.sens)
    
    @property    
    def youden_index(self):
        """
        Youden's index or Youden's J statistic, defined as sensitivity + specificity - 1
        """
        return self.sens + self.spec - 1
    
    @property    
    def matthews_correlation_coefficient(self):
        """
        Matthews Correlation Coefficient (MCC), a correlation coefficient between
        the observed and predicted binary classifications.
        """
        num = self.tp * self.tn + self.fp * self.fn
        denom = np.sqrt( self.pd * self.pos * self.neg * self.nd )
        return num / denom

    mcc = matthews_correlation_coefficient
    

# plotting
try:
    import matplotlib.pyplot as plt

    def plot_conf(conf):
        """
        Simple function to visualise a balanced confusion matrix.
        """
        res = plt.imshow(np.asarray(conf), cmap=plt.cm.jet, interpolation='nearest')
        # display correct detection percentages (only makes sense for CMs that are normalised per class (each row sums to 1)).
        for i, err in enumerate(conf.correct):
            err_percent = "%d%%" % round(err * 100)
            plt.text(i-.2, i+.1, err_percent, fontsize=14)

        cb = plt.colorbar(res)
        plt.show()
        
except ImportError:
    def plot_conf(conf):
        raise RuntimeError("Could not import matplotlib.pyplot, plotting not available.")
