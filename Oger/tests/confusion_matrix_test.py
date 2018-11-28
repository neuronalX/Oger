import mdp
import unittest
import Oger
import numpy as np
from Oger.utils import ConfusionMatrix, BinaryConfusionMatrix

# Python 2.6 unittest has no facilities to check exception messages, it seems.
# code for Python 2.7 with message checking is in the comments.

class ConfusionMatrixTest(unittest.TestCase):
    def setUp(self):
        pass
        
    def test_constructor_dimension_checks(self):
        # with self.assertRaises(RuntimeError) as cm:
        #    c = ConfusionMatrix([1, 2, 3])
        # self.assertEqual(cm.exception.message, "ConfusionMatrix should be 2-dimensional - ndim is 1")
        def func():
            c = ConfusionMatrix([1, 2, 3])
        self.assertRaises(RuntimeError, func)
                
        # with self.assertRaises(RuntimeError) as cm:
        #     c = ConfusionMatrix([[1, 2, 3],[4, 5, 6]])
        # self.assertEqual(cm.exception.message, "ConfusionMatrix should be rectangular - shape is 2x3")
        def func():
            c = ConfusionMatrix([[1, 2, 3], [4, 5, 6]])
        self.assertRaises(RuntimeError, func)
        
    def test_constructor_labels(self):
        c = ConfusionMatrix([[1, 2], [3, 4]])
        self.assertEqual(c.labels, [0, 1])
        
        c = ConfusionMatrix([[1, 2], [3, 4]], labels=['a', 'b'])
        self.assertEqual(c.labels, ['a', 'b'])
        
        # with self.assertRaises(RuntimeError) as cm:
        #     c = ConfusionMatrix([[1, 2], [3, 4]], labels=[1, 2, 3])
        # self.assertEqual(cm.exception.message, "Number of class labels does not equal number of classes - number of labels is 3, number of classes is 2")
        def func():
            c = ConfusionMatrix([[1, 2], [3, 4]], labels=[1, 2, 3])
        self.assertRaises(RuntimeError, func)
        
        # with self.assertRaises(RuntimeError) as cm:
        #     c = ConfusionMatrix([[1, 2], [3, 4]], labels=[1, 1])
        # self.assertEqual(cm.exception.message, "Class labels are not unique - labels are [1, 1]")
        def func():
            c = ConfusionMatrix([[1, 2], [3, 4]], labels=[1, 1])
        self.assertRaises(RuntimeError, func)
        
    def test_asarray(self):
        c = ConfusionMatrix([[1, 2], [3, 4]])
        self.assertTrue(np.all(c.asarray() == np.array([[1, 2], [3, 4]])))
        
    def test_add(self):
        c1 = ConfusionMatrix([[1, 2], [3, 4]])
        c2 = ConfusionMatrix([[3, 4], [1, 2]])
        s = c1 + c2
        t = ConfusionMatrix([[4, 6], [4, 6]])
        self.assertTrue(np.all(s.asarray() == t.asarray()))
    
    def test_normalise(self):
        c = ConfusionMatrix([[1, 2], [3, 4]])
        cn = c.normalise()
        self.assertTrue(np.all(cn.asarray() == np.array([[0.1, 0.2], [0.3, 0.4]])))
        
    def test_balance(self):
        c = ConfusionMatrix([[1, 3], [2, 2]])
        cb = c.balance()
        self.assertTrue(np.all(cb.asarray() == np.array([[0.25, 0.75], [0.5, 0.5]])))
        
    def test_properties(self):
        c = ConfusionMatrix([[1, 2], [3, 4]])
        cn = c.normalise()
        self.assertEqual(c.num_classes, 2)
        self.assertEqual(c.total, 10)
        self.assertEqual(cn.total, 1)
        
    def test_subsets(self):
        c = ConfusionMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertTrue(np.all(c.correct == [1, 5, 9]))
        self.assertTrue(np.all(c.incorrect == [5, 10, 15]))
        self.assertTrue(np.all(c.detections == [12, 15, 18]))
        self.assertTrue(np.all(c.ground_truth == [6, 15, 24]))
           
    def test_rates(self):
        c = ConfusionMatrix([[8, 2, 1], [1, 4, 3], [1, 2, 7]])
        self.assertTrue(np.all(c.classification_rates == [8.0/11, 0.5, 0.7]))
        self.assertTrue(np.all(c.error_rates == [3.0/11, 0.5, 0.3]))
        self.assertAlmostEqual(c.balanced_error_rate, np.mean([3.0/11, 0.5, 0.3]))
        self.assertAlmostEqual(c.balanced_classification_rate, np.mean([8.0/11, 0.5, 0.7]))
        self.assertAlmostEqual(c.classification_accuracy, 19.0 / 29.0)
        self.assertAlmostEqual(c.classification_error_rate, 10.0 / 29.0)

    def test_binary(self):
        c = ConfusionMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        bs = c.binary()
        self.assertTrue(np.all(bs[0].asarray() == np.array([[1, 5], [11, 28]])))
        self.assertTrue(np.all(bs[1].asarray() == np.array([[5, 10], [10, 20]])))
        self.assertTrue(np.all(bs[2].asarray() == np.array([[9, 15], [9, 12]])))
        for b in bs:
            # self.assertIsInstance(b, BinaryConfusionMatrix)
            self.assertTrue(isinstance(b, BinaryConfusionMatrix))

    def test_from_data(self):
        # with self.assertRaises(RuntimeError) as cm:
        #     i = np.atleast_2d(np.array([0, 1])).T
        #     t = np.atleast_2d(np.array([2])).T
        #     c = ConfusionMatrix.from_data(3, i, t)
        # self.assertEqual(cm.exception.message, "Output and target data should have the same shape")
        def func():
            i = np.atleast_2d(np.array([0, 1])).T
            t = np.atleast_2d(np.array([2])).T
            c = ConfusionMatrix.from_data(3, i, t)
        self.assertRaises(RuntimeError, func)
        
        i = np.atleast_2d(np.array([0, 0, 1, 2, 0, 0, 2, 2, 1, 2])).T
        t = np.atleast_2d(np.array([0, 0, 0, 1, 1, 0, 2, 2, 1, 1])).T
        c = ConfusionMatrix.from_data(3, i, t)
        
        self.assertTrue(np.all(c.truth == [4, 4, 2]))
        self.assertTrue(np.all(c.detections == [4, 2, 4]))
        self.assertTrue(np.all(c.correct == [3, 1, 2]))
        self.assertTrue(np.all(c.incorrect == [1, 3, 0]))

    def test_from_samples(self):
        i = [np.atleast_2d(np.array([0, 1])).T, np.atleast_2d(np.array([1, 2, 2])).T]
        t = [np.atleast_2d(np.array([0, 0])).T, np.atleast_2d(np.array([2, 2, 2])).T]
        c = ConfusionMatrix.from_samples(3, i, t)
        
        self.assertTrue(np.all(c.truth == [0.5, 0, 0.5]))
        self.assertTrue(np.all(c.detections == [0.5 / 2, (0.5 + 1.0/3)/ 2, (2.0/3)/2]))
        self.assertTrue(np.all(c.correct == [0.5/2, 0, (2.0/3)/2]))
        
        for result, truth in zip(c.incorrect, [0.5/2, 0, 1.0/6]):
            self.assertAlmostEqual(result, truth)
        
    def test_error_measure(self):
        ber = ConfusionMatrix.error_measure('ber', 3)
        i = np.atleast_2d(np.array([0, 0, 1, 2, 0, 0, 2, 2, 1, 2])).T
        t = np.atleast_2d(np.array([0, 0, 0, 1, 1, 0, 2, 2, 1, 1])).T
        self.assertAlmostEqual(ber(i, t), 1.0/3)
        
    def test_sparse(self):
        c = ConfusionMatrix([[5, 0, 0], [0, 0, 0], [1, 0, 3]])
        self.assertAlmostEqual(c.ber, 0.25/3)


        
class BinaryConfusionMatrixTest(unittest.TestCase):
    def setUp(self):
        pass
        
    def test_constructor_dimension_checks(self):
        # with self.assertRaises(RuntimeError) as cm:
        #     c = BinaryConfusionMatrix([1, 2, 3])
        # self.assertEqual(cm.exception.message, "ConfusionMatrix should be 2-dimensional - ndim is 1")
        def func():
            c = BinaryConfusionMatrix([1, 2, 3])
        self.assertRaises(RuntimeError, func)    
        
        # with self.assertRaises(RuntimeError) as cm:
        #     c = BinaryConfusionMatrix([[1, 2, 3], [4, 5, 6]])
        # self.assertEqual(cm.exception.message, "ConfusionMatrix should have shape 2x2 - shape is 2x3")
        def func():
            c = BinaryConfusionMatrix([[1, 2, 3], [4, 5, 6]])
        self.assertRaises(RuntimeError, func)        
        
    def test_normalise(self):
        c = BinaryConfusionMatrix([[1, 2], [3, 4]])
        cn = c.normalise()
        # self.assertIsInstance(cn, BinaryConfusionMatrix)
        self.assertTrue(isinstance(cn, BinaryConfusionMatrix))
        self.assertTrue(np.all(cn.asarray() == np.array([[0.1, 0.2], [0.3, 0.4]])))
        
    def test_balance(self):
        c = BinaryConfusionMatrix([[1, 3], [2, 2]])
        cb = c.balance()
        # self.assertIsInstance(cb, BinaryConfusionMatrix)
        self.assertTrue(isinstance(cb, BinaryConfusionMatrix))
        self.assertTrue(np.all(cb.asarray() == np.array([[0.25, 0.75], [0.5, 0.5]])))

    def test_add(self):
        c1 = BinaryConfusionMatrix([[1, 2], [3, 4]])
        c2 = BinaryConfusionMatrix([[3, 4], [1, 2]])
        s = c1 + c2
        t = BinaryConfusionMatrix([[4, 6], [4, 6]])
        # self.assertIsInstance(s, BinaryConfusionMatrix)
        self.assertTrue(isinstance(s, BinaryConfusionMatrix))
        self.assertTrue(np.all(s.asarray() == t.asarray()))
        
    def test_elements(self):
        c = BinaryConfusionMatrix([[1, 2], [3, 4]])
        self.assertEqual(c.true_positives, 1)
        self.assertEqual(c.false_negatives, 2)
        self.assertEqual(c.false_positives, 3)
        self.assertEqual(c.true_negatives, 4)
        
        self.assertEqual(c.positives, 3)
        self.assertEqual(c.negatives, 7)
        self.assertEqual(c.positive_detections, 4)
        self.assertEqual(c.negative_detections, 6)
        
    def test_measures(self):
        c = BinaryConfusionMatrix([[1, 2], [3, 4]])
        self.assertAlmostEqual(c.recall, 1.0/3)
        self.assertAlmostEqual(c.precision, 1.0/4)
        self.assertAlmostEqual(c.sensitivity, 1.0/3)
        self.assertAlmostEqual(c.specificity, 4.0/7)
        self.assertAlmostEqual(c.f1, 2.0/7)
        self.assertAlmostEqual(c.f_measure(2), 5.0/16)
        self.assertAlmostEqual(c.positive_likelihood, 7.0/9)
        self.assertAlmostEqual(c.negative_likelihood, 12.0/14)
        self.assertAlmostEqual(c.youden_index, -2.0/21)
        self.assertAlmostEqual(c.matthews_correlation_coefficient, 10.0/np.sqrt(504))

    def test_error_measure(self):
        i = np.atleast_2d(np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 0])).T
        t = np.atleast_2d(np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])).T
        recall = BinaryConfusionMatrix.error_measure('recall')
        self.assertAlmostEqual(recall(i, t), 2.0/3)
        
        
if __name__ == '__main__':
    unittest.main()

