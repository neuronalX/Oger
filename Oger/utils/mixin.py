import mdp
import inspect

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

import copy_reg
import types
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


def make_inspectable(baseclass):
    """
    This function makes a flow or node inspectable, i.e. it keeps the outputs produced by execute(). These can later be retrieved with the inspect() method.     
    """
    class InspectableClass():
        def _execute_seq(self, x, nodenr=None):
            """This code is a copy from the code from mdp.Flow, but with additional
            state tracking.
            """
            if not hasattr(self, '_states'):
                self._states = [[] for _ in range(len(self.flow))]

            flow = self.flow
            if nodenr is None:
                nodenr = len(flow) - 1
            for i in range(nodenr + 1):
                try:
                    x = flow[i].execute(x)
                    self._states[i].append(x)
                except Exception, e:
                    self._propagate_exception(e, i)
            return x

        def execute(self, iterable, *args, **kwargs):
            if hasattr(self, 'flow'):
                self._states = [[] for _ in range(len(self.flow))]
            else:
                self._states = []

            output = self.execute_no_inspect(iterable, *args, **kwargs)

            if hasattr(self, 'flow'):
                for i in range(len(self.flow)):
                    self._states[i] = mdp.numx.concatenate(self._states[i])
            else:
                self._states.append(output)

            return output

        def _inverse_seq(self, x):
            """This code is a copy from the code from mdp.Flow, but with additional
            state tracking.
            """
            if not hasattr(self, '_states'):
                self._states = [[] for _ in range(len(self.flow))]

            flow = self.flow
            for i in range(len(flow) - 1, -1, -1):
                try:
                    x = flow[i].inverse(x)
                    self._states[i].append(x)
                except Exception, e:
                    self._propagate_exception(e, i)
            return x

        def inverse(self, iterable):
            if hasattr(self, 'flow'):
                self._states = [list() for _ in range(len(self.flow))]

            output = self.inverse_no_inspect(iterable)

            if hasattr(self, 'flow'):
                for i in range(len(self.flow)):
                    self._states[i] = mdp.numx.concatenate(self._states[i])
            else:
                self._states = output

            return output

        def inspect(self, *args):
            """Return the state of the given node or node number in the flow.
            """
            if len(args) > 0:
                node_or_nr = args[0]
                if isinstance(node_or_nr, mdp.Node):
                    return self._states[self.flow.index(node_or_nr)]
                else:
                    return self._states[node_or_nr]
            else:
                return self._states

    if not inspect.isclass(baseclass):
        print 'Warning: make_inspectable() should be applied to classes, not to objects.'
        return

    if hasattr(baseclass, 'inspect'):
        print 'Class ' + baseclass.__name__ + ' is already inspectable.'
        return

    method_list = ['_execute_seq', 'execute', '_inverse_seq', 'inverse']

    for method in method_list:
        if hasattr(baseclass, method):
            setattr(baseclass, method + '_no_inspect', getattr(baseclass, method))
            setattr(baseclass, method, getattr(InspectableClass, method).im_func)

    setattr(baseclass, 'inspect', InspectableClass.inspect)

    baseclass.__bases__ = baseclass.__bases__ + (InspectableClass,)

def enable_washout(washout_class, washout=0, execute_washout=False):
    """
    This helper function injects additional code in the given class such
    that during training the first N timesteps are disregarded. This can be applied
    to all trainable nodes, both supervised and unsupervised.
    """

    if not isinstance(washout_class, type):
        raise Exception('Washout can only be enabled on classes.')

    if not hasattr(washout_class, "_train"):
        raise Exception('Object should have a _train method.')

    if hasattr(washout_class, "washout"):
        print ('Warning: washout already enabled.')
        return

    # helper washout class
    class Washout:
        def _train(self, x, *args, **kwargs):
            if len(args) > 0:
                self._train_no_washout(x[self.washout:, :], args[0][self.washout:, :], **kwargs)
            else:
                self._train_no_washout(x[self.washout:, :], **kwargs)

        def _execute(self, x):
            return self._execute_no_washout(x[self.washout:, :])

    # inject new methods
    setattr(washout_class, "_train_no_washout", washout_class._train)
    setattr(washout_class, "_train", Washout._train)
    setattr(washout_class, "_execute_no_washout", washout_class._execute)
    setattr(washout_class, "washout", washout)
    if execute_washout:
        setattr(washout_class, "_execute", Washout._execute)


    # add to base classes
    washout_class.__bases__ += (Washout,)

def disable_washout(washout_class):
    """
    Disable previously enabled washout.
    """

    if not isinstance(washout_class, type):
        raise Exception('Washout can only be enabled on classes.')

    if not hasattr(washout_class, "_train"):
        raise Exception('Object should have a _train method.')

    if not hasattr(washout_class, "washout"):
        raise Exception('Washout not enabled.')

    del washout_class._train
    del washout_class._execute
    del washout_class.washout

    setattr(washout_class, "_train", washout_class._train_no_washout)
    setattr(washout_class, "_execute", washout_class._execute_no_washout)
