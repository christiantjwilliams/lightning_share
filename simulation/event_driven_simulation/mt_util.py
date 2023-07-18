from typing import NoReturn, List
import numpy as np
import numpy.typing as npt
import simpy

class Request():
    def __init__(self, id: int, dnn: str, job_list: List[List[int]], max_ids) -> NoReturn:
        self.request_id = id
        self.dnn = dnn
        self.request_size = len(job_list)  # the number of layer jobs in this DNN inference request
        # TODO: rename, misleading 'set' var name
        max_job_id = max_ids['max_job_id']
        max_ids['max_job_id'] += self.request_size
        self.job_set = [Job(max_job_id+i, job_list[i], 0, max_ids) for i in range(self.request_size)]
        self.status = 0 

class Job():
    # TODO: rename size parameter, misleading
    def __init__(self, id: int, size: List[int], time: np.int64, max_ids) -> NoReturn:
        self.job_id = id
        self.job_size = sum(size)  # the sum of vector-vector product sizes for each Task in the Job
        # TODO: rename, misleading 'set' var name
        max_task_id = max_ids['max_task_id']
        max_ids['max_task_id'] += len(size)
        self.task_set = [Task(max_task_id+i, size[i], size[i]) for i in range(len(size))]
        self.time = time
        self.status = 0

class Task():
    def __init__(self, id: int, size: int, time: np.int64) -> NoReturn:
        self.id = id
        self.size = size  # the length of vector-vector product size
        self.time = time

    def __repr__(self):
        return 'Task ' + str(self.id) + ': ' + str(self.size)

class Core():
    def __init__(self, env, id):
        self.env = env
        self.action = env.process(self.run())

        self.id = id  # the id of core
        self.load_on_core = 0  # the load of each core is initialized to be 0
        self.tasks_on_core = []  # the list of tasks assigned on each core

    def add_task_to_core(self, task: Task):
        self.tasks_on_core.append(task)
        self.load_on_core += task.size
    
    def run(self):
        while self.tasks_on_core:
            current_task = self.tasks_on_core.pop(0)
            print('tasks on core:', self.tasks_on_core)
            print('Start task %s at %s' % (current_task.id, self.env.now))
            self.load_on_core -= current_task.time
            yield self.env.timeout(current_task.time)
            

    def __repr__(self):
        return 'Core ' + str(self.id) + ': ' + str(self.tasks_on_core)

class Processor():
    def __init__(self, env, cores, alg):
        self.env = env
        self.cores = cores
        self.alg = alg
        self.requests = []

    def add_request(self, request):
        self.requests.append(request)
    
    def simulate(self):
        while True:
            if self.requests:
                current_req = self.requests.pop(0)
                for job in current_req.job_set:
                    self.cores = self.alg(job.task_set, self.cores)
                    print('Core Assignments:\n', self.cores)
                    for core in self.cores:
                        core.run()