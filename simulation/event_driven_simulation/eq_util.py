from typing import NoReturn, List
from heapq import merge
import numpy as np
from numpy import random
from tqdm import tqdm
import matplotlib.pyplot as plt
# import numpy.typing as npt
#import pickle
#import cPickle as pickle
import _pickle as cPickle
import gc
import time
from collections import deque
import concurrent.futures
from functools import partial

#TODO: taking dp latency into account only works for round robin

# class MatrixProcess():
#     def __init__(self) -> NoReturn:
#         pass
#     def conv_as_matrix_multiply(conv_matrix: npt.ArrayLike, kernel_size: npt.ArrayLike, \
#         stride: npt.ArrayLike, padding: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
#         pass

with open(f'built_models/model_layer_nums.p', 'rb') as f:
    MODEL_LAYER_NUMS = cPickle.load(f)

class Event():
    '''
    base class of an event
    '''
    def __init__(self, id: int, event_type: str, t_start: np.int64, \
        t_finish: np.int64) -> NoReturn:
        self.id = id
        self.type = event_type
        self.t_start = t_start
        self.t_finish = t_finish

class Task(Event):
    '''
    Task is a vector-vector product
    '''
    def __init__(self, id: int, req_id: int, dnn: str, size: int, t_start: np.int64, t_finish: np.int64, is_first_layer: bool) -> NoReturn:
        super().__init__(id, 'Task', t_start, t_finish) # size: vector-vector product size
        self.req_id = req_id
        self.dnn = dnn
        self.size = size
        self.is_first_layer = is_first_layer
        self.core = None

    def __repr__(self):
        return f'Task T-{self.t_start}'

    def set_core(self, core):
        self.core = core

class Job(Event):
    '''
    Job is a fundamental unit on the DAG
    '''
    def __init__(self, id: int, req_id: int, dnn: str, layer_num: int, previous_job_t_finish: int, t_start: np.int64, \
        t_finish: np.int64) -> NoReturn:
        super().__init__(id, 'Job', t_start, t_finish)
        self.tasks = None
        self.req_id = req_id
        self.dnn = dnn
        self.layer_num = layer_num
        self.previous_job_t_finish = previous_job_t_finish

    def __repr__(self):
        return f'Job T-{self.t_start}'

    def add_tasks(self, tasks):
        self.tasks = tasks
        self.size = sum(task.size for task in tasks)

class Request(Event):
    '''
    a DNN inference with a list of all of its layers
    '''
    def __init__(self, id: int, next_job_id: int, dnn: str, t_arrival: np.int64, t_start: np.int64, \
            t_finish: np.int64) -> NoReturn:
        super().__init__(id, 'Request', t_start, t_finish)
        self.next_job_id = next_job_id
        self.t_arrival = t_arrival # time that request arrives for computation
        self.dnn = dnn

    def __repr__(self):
        return f'{self.dnn} Request @ T-{self.t_arrival}, T-{self.t_start}'

class JobFinish(Event):
    '''
    A Job finish event which indicates that a job has been finished, thus another may be generated
    '''
    def __init__(self, id: int, job_id: int, req_id: int, dnn: str, t_finish: np.int64, layer_num: int, is_last: bool):
        super().__init__(id, 'JobFinish', t_finish, t_finish)
        self.job_id = job_id
        self.req_id = req_id
        self.dnn = dnn
        self.layer_num = layer_num
        self.is_last = is_last

class RequestFinish(Event):
    '''
    A Request finish event which indicates that a request has been finished, and we can log its total completion time
    '''
    def __init__(self, id: int, req_id: int, t_finish: np.int64):
        super().__init__(id, 'RequestFinish', t_finish, t_finish)
        self.req_id = req_id

class EventQueue():
    '''
    Queue interface that supports both merging and appending events
    '''
    def __init__(self) -> NoReturn:
        self.queue = deque([])

    def __bool__(self):
        return any(self.queue)

    def __repr__(self) -> NoReturn:
        return str(self.queue)

    def merge(self, events) -> NoReturn:
        '''
        merge events into event queue sorted by t_start
        '''
        # print("Merging {} into event queue", format(events))
        self.queue = deque(merge(sorted(events,key=lambda event:event.t_start), self.queue, key=lambda event:event.t_start))

    def append(self, event) -> NoReturn:
        '''
        append event to end of queue
        '''
        # print("append {} to the end of queue".format(event))
        self.queue.append(event)

    def pop(self):
        '''
        pop an event off the queue
        '''
        return self.queue.popleft()

class EventGenerator():
    '''
    Generates events in simulation
    '''
    def __init__(self):
        self.max_ids = {'max_task_id':0, 'max_job_id':0, 'max_req_id':0, 'max_job_finish_id':0, 'max_req_finish_id':0}

    #@jit(target_backend='cuda')
    def gen_request(self, model_type, sparsity, t_arrival, accelerator):
        '''
        generate a Request() event
        '''
        # specify DNN type to generate
        cores = accelerator.cores
        new_req = Request(self.max_ids['max_req_id'], self.max_ids['max_job_id'], model_type, t_arrival, max(t_arrival, cores[accelerator.last_scheduled_core].t_finish), None)
        self.max_ids['max_req_id'] += 1
        self.max_ids['max_job_id'] += 1
        return new_req

    def gen_job(self, accelerator, job_id, req_id, model_type, layer_num, previous_job_t_finish, gen_time):
        start = time.time()
        gc.disable()
        print(f'generating {model_type} job...')
        with open(f'built_models/{model_type}_layer_{layer_num}.p', 'rb') as f:
            new_job = cPickle.load(f)
        gc.enable()
        end = time.time()
        print(f'done, took {end - start} s')
        new_job.id = job_id
        new_job.req_id = req_id
        new_job.previous_job_t_finish = previous_job_t_finish
        for i in range(len(new_job.tasks)):
            task = new_job.tasks[i]
            task.id = self.max_ids['max_task_id']+i
            task.req_id = new_job.req_id
        start1 = time.time()
        print(f'scheduling {model_type} tasks...')
        parallel_round_robin(new_job.tasks, accelerator, new_job.previous_job_t_finish, gen_time)
        end1 = time.time()
        start2 = time.time()
        new_job = set_event_bounds(new_job)
        end2 = time.time()
        print(f'done, took {end1 - start1} and {end2 - start2} s')
        return new_job

    #TODO: deprecated
    def gen_tasks(self, job, accelerator, gen_time):
        start = time.time()
        gc.disable()
        with open(f'built_models/{job.dnn}_layer_{job.layer_num}.p', 'rb') as f:
            layer = cPickle.load(f)
        gc.enable()
        new_tasks = []
        end = time.time()
        time_to_open = end - start
        for i in range(len(layer)):
            task_size = layer[i]
            # add datapath latency to first layer tasks
            is_first_layer = True if job.layer_num == 0 else False
            new_task = Task(self.max_ids['max_task_id']+i, job.req_id, job.dnn, task_size, None, None, is_first_layer=is_first_layer)
            new_tasks.append(new_task)
        self.max_ids['max_task_id'] += len(layer)
        scheduled_tasks = round_robin(new_tasks, accelerator, job.previous_job_t_finish, gen_time)
        return new_tasks, time_to_open

    def gen_request_finish(self, job_finish):
        new_request_finish = RequestFinish(self.max_ids['max_req_finish_id'], job_finish.req_id, job_finish.t_finish)
        self.max_ids['max_req_finish_id'] += 1 
        return new_request_finish
        
    def gen_job_finish(self, job):
        is_last = True if MODEL_LAYER_NUMS[job.dnn] - 1 == job.layer_num else False
        new_job_finish = JobFinish(self.max_ids['max_job_finish_id']+1, job.id, job.req_id, job.dnn, job.t_finish, job.layer_num, is_last)
        self.max_ids['max_job_finish_id'] += 1
        return new_job_finish


class Core():
    '''
    a processor Core which simulates task computation
    '''
    def __init__(self, id: int) -> NoReturn:
        self.id = id  # the id of core
        self.load = 0  # the load of each core is initialized to be 0
        self.tasks = deque([])  # the list of tasks assigned on each core
        self.t_finish = 0 # time that core will finish all computation

    def __repr__(self):
        return 'Core ' + str(self.id)

    def __hash__(self):
        return hash(str(self))

    def add_task(self, task: Task) -> NoReturn:
        '''
        add a task to the Core's compute queue
        '''
        self.tasks.append(task)
        self.load += task.size
        self.t_finish = task.t_finish

    def reset_core(self) -> NoReturn:
        '''
        reset internal clock and task list of Core
        '''
        self.t_finish = 0
        self.tasks = 0

class Accelerator():
    '''
    a 'photonic' accelerator which holds processor Cores
    '''
    def __init__(self, id: int, cores: List[Core], dp_latency: int) -> NoReturn:
        self.id = id
        self.cores = cores
        self.last_scheduled_core = 0
        self.dp_latency = dp_latency

# scheduling algorithms
def set_event_bounds(event):
    '''
   set the t_start and t_finish for an event given its subevents
    '''
    min_t_start = float('inf')
    max_t_finish = 0
    if event.type == 'Job':
        subevents = event.tasks
    elif event.type == 'Request':
        subevents = event.jobs
    for subevent in subevents:
        min_t_start = min(min_t_start, subevent.t_start)
        max_t_finish = max(max_t_finish, subevent.t_finish)
    event.t_start, event.t_finish = min_t_start, max_t_finish
    return event

def long_flow_first(req, accelerator, gen_time):
    '''
    schedule request tasks to cores, longest flow to least load core
    '''
    previous_job_t_finish = 0
    cores = accelerator.cores
    for job in req.jobs:
        # sort at task level based on tasks's longest flow: large task first
        sorted_tasks = sorted(job.tasks, key=lambda t: t.size, reverse=True)
        # assign tasks on cores based on core weights
        #for t in tqdm(sorted_tasks, desc="Scheduling tasks..."):
        for task in sorted_tasks:
            min_load_core = min(cores, key=lambda core: core.load)
            # set t_start to latest of end of previous compute, end of previous job, or gen_time
            task.t_start = max(min_load_core.t_finish, previous_job_t_finish, gen_time)
            task.t_finish = task.t_start + task.size
            min_load_core.add_task(task)  # assign the longest flow to the least load core
        # set t_start and t_finish for current job
        job = set_event_bounds(job)
        previous_job_t_finish = job.t_finish

    # set t_start and t_finish for current request
    req = set_event_bounds(req)

def round_robin(tasks, accelerator, previous_job_t_finish, gen_time):
    '''
    schedule request tasks to cores, round robin
    '''
    cores = accelerator.cores
    dp_latency = accelerator.dp_latency
    # TODO: change to "next_schedule_core"
    i = accelerator.last_scheduled_core
    for task in tasks:
        if i >= len(cores):
            i = 0
        # set t_start to latest of end of previous compute, end of previous job, or gen_time
        task.t_start = max(cores[i].t_finish, previous_job_t_finish, gen_time)
        task.t_finish = task.t_start + task.size
        # account for dp latency if task is in first layer
        if task.is_first_layer:
            if cores[i].tasks:
                if cores[i].tasks[-1].req_id != task.req_id:
                    task.t_finish += dp_latency
            else:
                task.t_finish += dp_latency
        cores[i].add_task(task)
        task.set_core(cores[i])
        i += 1
    # set t_start and t_finish for current job
    accelerator.last_scheduled_core = i
    # set t_start and t_finish for current request
    return tasks

def add_task_to_core(task, core, previous_job_t_finish, gen_time, dp_latency):
    task.t_start = max(core.t_finish, previous_job_t_finish, gen_time)
    task.t_finish = task.t_start + task.size
    # account for dp latency if task is in first layer
    if task.is_first_layer:
        if core.tasks:
            if core.tasks[-1].req_id != task.req_id:
                task.t_finish += dp_latency
        else:
            task.t_finish += dp_latency
    core.add_task(task)
    task.set_core(core)

def parallel_round_robin(tasks, accelerator, previous_job_t_finish, gen_time):
    '''
    schedule request tasks to cores, round robin
    '''
    cores = accelerator.cores
    dp_latency = accelerator.dp_latency
    # TODO: change to "next_schedule_core"
    next_schedule_core = accelerator.last_scheduled_core
    last_task_ind = 0
    partial_function = partial(add_task_to_core, previous_job_t_finish=previous_job_t_finish, gen_time=gen_time, dp_latency=dp_latency)
    # schedule tasks in parallel, one pass through the cores at a time
    for _ in range(len(tasks)//len(cores)):
        task_indices = [last_task_ind + i for i in range(len(cores))]
        last_task_ind = task_indices[-1] + 1
        core_indices = []
        for core_ind in range(len(cores)):
            if core_ind >= len(cores):
                core_ind = 0
            core_indices.append(core_ind)
            core_ind += 1
        tasks_list = [tasks[i] for i in task_indices]
        cores_list = [cores[j] for j in core_indices]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(partial_function, tasks_list, cores_list)
    # schedule the remaining tasks
    remaining_tasks = len(tasks) % len(cores)
    task_indices = [last_task_ind + i for i in range(remaining_tasks)]
    core_indices = []
    for core_ind in range(next_schedule_core, next_schedule_core+remaining_tasks):
        if core_ind >= len(cores):
            core_ind = 0
        core_indices.append(core_ind)
        core_ind += 1
    tasks_list = [tasks[i] for i in task_indices]
    cores_list = [cores[j] for j in core_indices]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(partial_function, tasks_list, cores_list)
    # set t_start and t_finish for current job
    accelerator.last_scheduled_core = core_ind
    # set t_start and t_finish for current request
    return tasks

def lookup_table(req, accelerator, gen_time, table=None, permute=True):
    '''
    schedule request tasks to cores, using a snapshot-based lookup table
    '''
    previous_job_t_finish = 0
    cores = accelerator.cores
    table = {core:0 for core in cores} if not table else table
    for job in req.jobs:
        # sort at task level based on tasks's longest flow: large task first
        sorted_tasks = sorted(job.tasks, key=lambda t: t.size, reverse=True)
        table_keys = list(table.keys())
        if permute:
            permuted_table_keys = np.random.permutation(table_keys)
            table = {k: table[k] for k in permuted_table_keys}
        for task in sorted_tasks:
            min_load_core = min(table, key=lambda core: table[core])
            # set t_start to latest of end of previous compute, end of previous job, or gen_time
            task.t_start = max(min_load_core.t_finish, previous_job_t_finish, gen_time)
            task.t_finish = task.t_start + task.size
            min_load_core.add_task(task)  # assign the longest flow to the least load core
        # set t_start and t_finish for current job
        job = set_event_bounds(job)
        previous_job_t_finish = job.t_finish

    # set t_start and t_finish for current request
    req = set_event_bounds(req)

# VVP CDF

# for model_name in MODEL_DICT:
#     model = MODEL_DICT[model_name]
#     print(f'plotting {model_name}') 
#     if model_name == 'DLRM':
#         x = np.sort(model)
#         print('sorted vvps:', x)
#         y = 1. * np.arange(len(x)) / (len(x) - 1)
#         print('plotting...')
#         # plot CDF
#         plt.plot(x, y, label=model_name)
#         print('done')
#     else:
#         bucketed_model = []
#         print('bucketing...')
#         for i in range(200):
#             model_slice = model[int(len(model)/200)*i:int(len(model)/200)*(i+1)]
#             bucketed_model.append(round(float(np.mean(model_slice)), 2))
#         print('sorting...')
#         x = np.sort(bucketed_model)
#         print('sorted vvps:', x)
#         y = 1. * np.arange(len(x)) / (len(x) - 1)
#         print('plotting...')
#         # plot CDF
#         plt.plot(x, y, label=model_name)
#         print('done')

# plt.xlabel('VVP Compute Size')
# plt.ylabel('CDF')
# plt.title('')
# plt.legend()
# plt.show()
