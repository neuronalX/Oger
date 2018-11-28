import mdp.parallel
import subprocess
import getpass
from datetime import datetime

try:
    import pp
except ImportError:
    pass

class ParallelFlow(mdp.parallel.ParallelFlow):
    def __init__(self, flow, data, verbose=False, **kwargs):
        self._data = data
        super(ParallelFlow, self).__init__(flow, verbose=verbose, **kwargs)

    def _create_execute_task(self):
        """Create and return a single execution task.

        Returns None if data iterator end is reached.
        """
        try:
            dummy = self._exec_data_iterator.next()
            dummy[0].append(self._data)
            # TODO: check if forked task is forkable before enforcing caching
            return (dummy, None)
        except StopIteration:
            return None

class CondorGridScheduler(mdp.parallel.Scheduler):
    def __init__(self, ppserver=None, max_queue_length=1,
                 result_container=mdp.parallel.ListResultContainer(),
                 pp_cmd=None,
                 fsdomain = None,
                 ppservers=None,
                 verbose=False):
        super(GridScheduler, self).__init__(result_container=result_container,
                                          verbose=verbose)
        try:
            self.whoami = getpass.getuser()
        except:
            print 'Unable to determine username'
            raise

        # List of tuples instead of a dictionary because the order of the commands
        # is apparently important for condor
        self.condor_options = [
        ['cmd', pp_cmd], \
        ['args', ''], \
        ['requirements', 'OpSys=="*" || Arch=="*" || FileSystemDomain == ' + fsdomain], \
        ['output', outdir + 'out.$(cluster)'], \
        ['error', outdir + 'err.$(cluster).$(Process)'], \
        ['log', outdir + + 'log.$(cluster).$(Process)'], \
        ['universe', 'vanilla'], \
        ['environment', '"MPLCONFIGDIR=/tmp/"'], \
        ['Notification', 'Error'], \
        ['should_transfer_files', 'No'], \
        #('when_to_transfer_output', 'ON_EXIT'), \
        ['transfer_executable', 'False'], \
        ['InitialDir', '/mnt/snn_gluster/usr/' + self.whoami], \
        ]

        self.ppservers = ppservers

        self.reset()

    def reset(self):
        self.jobs = []
        self.job_id = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.ppserver = pp.Server(ncpus=0, ppservers=self.ppservers, secret=self.whoami + '_' + self.job_id)
        self.jobs_done = 0

    def execute(self):
        tempfile = open('condor_tmp_file.job', 'w')
        self.condor_options[1][1] = '-p 60002 -d -w 1 -t 3 -s ' + self.whoami + '_' + self.job_id
        self.condor_job_str = ''
        for option, value in self.condor_options:
            self.condor_job_str += option + ' = ' + value + '\n'

        self.condor_job_str += 'queue ' + str(len(self.jobs))
        tempfile.write(self.condor_job_str)
        tempfile.close()
        p = subprocess.Popen(['condor_submit', 'condor_tmp_file.job'], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        print self.condor_job_str
        stdout = p.communicate()[0]
        print(stdout)

    def _process_task(self, data, task_callable, task_index):
        """Non-blocking processing of tasks.

        Depending on the scheduler state this function is non-blocking or
        blocking. One reason for blocking can be a full task-queue.
        """
        task = (data, task_callable.fork(), task_index)
        def execute_task(task):
            """Call the first args entry and return the return value."""
            data, task_callable, task_index = task
            task_callable.setup_environment()
            return task_callable(data), task_index
        task_submitted = False
        while not task_submitted:
            # release lock to enable result storage
            self._lock.release()
            # the inner tuple is a trick to prevent introspection by pp
            # this forces pp to simply pickle the object
            self.jobs.append(self.ppserver.submit(execute_task, args=(task,),
                                 callback=self._pp_result_callback))
            task_submitted = True

    def get_results(self):
        self.execute()
        results = super(GridScheduler, self).get_results()
        self.reset()
        return results

    def _pp_result_callback(self, result):
        """Calback method for pp to unpack the result and the task id.

        This method then calls the normal _store_result method.
        """
        self.jobs_done += 1
        print "Jobs done: " + str(self.jobs_done) + "/" + str(len(self.jobs))
        self._store_result(result[0], result[1])

    def _shutdown(self):
        """Call destroy on the ppserver."""
        self.ppserver.destroy()
