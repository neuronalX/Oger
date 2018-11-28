"""
This subpackage contains support for parallel processing on a computing grid. 
"""

from parallel import (CondorGridScheduler, ParallelFlow)
import parallel_optimization

# clean up namespace
del parallel
del parallel_optimization
__all__ = ['CondorGridScheduler', 'ParallelFlow']
