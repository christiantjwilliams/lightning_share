from typing import NoReturn, List
import numpy as np
import numpy.typing as npt

class MatrixProcess():
    def __init__(self) -> NoReturn:
        pass
    
    def ConvAsMatrixMultiply(conv_matrix: npt.ArrayLike, kernel_size: npt.ArrayLike, stride: npt.ArrayLike, padding: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        pass
    
# base class of an event
class Event():
    def __init__(self, id: int, event_type: str, size: int, t_start: np.int64) -> NoReturn:
        self.id = id
        self.type = event_type
        self.size = size
        self.t_start = t_start
        self.t_finish = None

# Task is a vector-vector product
class Task(Event):
    def __init__(self, id: int, size: int, t_start: np.int64, job_id: int, req_id: int) -> NoReturn:
        super().__init__(id, 'Task', size, t_start) # size: the length of vector-vector product size
        self.job_id = job_id
        self.req_id = req_id
        self.status = 0 # 0 means unfinished, 1 means finished

    def __repr__(self):
        return 'Task ' + str(self.id) + ', Job ' + str(self.job_id) + ', Req ' + str(self.req_id) + ': ' + str(self.size) 

    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self,other):
        return self.id == other.id and self.type == other.type

# Job is a fundamental unit on the DAG
class Job(Event):
    def __init__(self, id: int, tasks: List[Task], t_start: np.int64, req_id: int) -> NoReturn:
        super().__init__(id, 'Job', sum(task.size for task in tasks), t_start)
        self.tasks = tasks
        self.req_id = req_id
        # self.ct := number of completed tasks in job
        self.ct = 0
        self.status = 0
    
    def __repr__(self):
        return 'Job ' + str(self.id)
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self,other):
        return self.id == other.id and self.type == other.type

# Request is a DNN inference with a list of all of its layers
class Request(Event):
    def __init__(self, id: int, dnn: str, jobs: List[Job], t_start: np.int64) -> NoReturn:
        super().__init__(id, 'Request', sum(job.size for job in jobs), t_start)
        self.dnn = dnn
        self.jobs = jobs
        # self.cj := number of completed jobs in request
        self.cj = 0
        self.status = 0
    
    def __repr__(self):
        return 'Request ' + str(self.id)
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self,other):
        return self.id == other.id and self.type == other.type


class Core():
    def __init__(self, id: int) -> NoReturn:
        self.id = id  # the id of core
        self.load_on_core = 0  # the load of each core is initialized to be 0
        self.tasks_on_core = []  # the list of tasks assigned on each core
        self.active = True
    
    def __repr__(self):
        return 'Core ' + str(self.id) + ': ' + str(self.tasks_on_core)

    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self,other):
        return self.id == other.id and self.type == other.type

    def add_task_to_core(self, task: Task) -> NoReturn:
        self.tasks_on_core.append(task)
        self.load_on_core += task.size
        
    def compute(self, t_passed, current_time, jd, rd):
        '''
        t_passed: time that will pass during this compute step
        current_time: current time elapsed in sim
        jd: dictionary of jobs {job_id:Job()}
        rq: dictionary of requests {req_id:Request()}
        '''
        # make progress on current computation task

        # check for core resume condition, if necessary
        if not self.active:
            current_job_id = self.tasks_on_core[0].job_id
            #print('core id:', self.id)
            #print('job id:', current_job_id)
            if jd[current_job_id-1].status:
                self.active = True

        if self.active:
            # TODO: change from size == time representation
            self.load_on_core = max(0, self.load_on_core-t_passed)

            # complete tasks that will be completed in this compute step
            compute = True
            while compute:
                if self.tasks_on_core:
                    if t_passed >= self.tasks_on_core[0].size and self.active:

                        # 'start' the current task, job, or req if appropriate
                        # if not self.tasks_on_core[0].t_start:
                        #     self.tasks_on_core[0].t_start = current_time

                        #     current_job = jd[self.tasks_on_core[0].job_id]
                        #     if not current_job.t_start:
                        #         current_job.t_start = current_time
                        #         current_req = rd[current_job.req_id]
                        #         if not current_req.t_start:
                        #             current_req.t_start = current_time
                        
                        t_passed -= self.tasks_on_core[0].size
                        current_time += self.tasks_on_core[0].size

                        # complete the current task
                        self.tasks_on_core[0].t_finish = current_time
                        self.tasks_on_core[0].status = 1

                        current_job = jd[self.tasks_on_core[0].job_id]
                        current_job.ct += 1

                        # check to see if a job was completed
                        if current_job.ct == len(current_job.tasks):
                            # complete the current job
                            current_job.t_finish = current_time
                            current_job.status = 1
                            current_req = rd[current_job.req_id]
                            current_req.cj += 1
                            
                            # check to see if a request was completed
                            if current_req.cj == len(current_req.jobs):
                                current_req.t_finish = current_time
                                current_req.status = 1
                        
                        old_job_id, old_req_id = self.tasks_on_core[0].job_id, self.tasks_on_core[0].req_id        
                        self.tasks_on_core.pop(0)
                        # check to see if a core halt is necessary (to maintain DAG)
                        if self.tasks_on_core:
                            new_job_id, new_req_id = self.tasks_on_core[0].job_id, self.tasks_on_core[0].req_id
                            if new_job_id > old_job_id:
                                if new_req_id == old_req_id:
                                    if not jd[old_job_id].status:
                                        self.active = False
                    else:
                        compute = False
                else:
                    compute = False

            if self.active and self.tasks_on_core and t_passed > 0:
                self.tasks_on_core[0].t_start = current_time
                self.tasks_on_core[0].size -= t_passed

                # start the current job or req if appropriate
                # current_job = jd[self.tasks_on_core[0].job_id]
                # if not current_job.t_start:
                #     current_job.t_start = current_time
                #     current_req = rd[current_job.req_id]
                #     if not current_req.t_start:
                #         current_req.t_start = current_time
                    