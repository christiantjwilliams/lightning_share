from typing import NoReturn, List
import numpy as np
import numpy.typing as npt
import random
import simpy
from tqdm import tqdm
from mt_util import Request, Job, Task, Core, Processor

NUM_CORES = 4  # Number of cores on the processor
T_INTER = 7       # Create a request every ~7 minutes
SIM_TIME = 1     # Simulation time in minutes
GEN_CONSTRAINTS = {'task_size':random.randint(1,100), \
'job_size':random.randint(1,10), 'req_size':random.randint(1,5), \
'gen_time':random.randint(T_INTER-2,T_INTER+2)}

def long_flow_first(tasks, cores):
    print("Running long flow first")
    # sort at task level based on tasks's longest flow: large task first
    sorted_tasks = sorted(tasks, key=lambda t: t.size, reverse=True)
    # assign tasks on cores based on core weights
    for t in tqdm(sorted_tasks, desc="Scheduling tasks..."):
        min_load_core = min(cores, key=lambda core: core.load_on_core)
        min_load_core.add_task_to_core(t)  # assign the longest flow to the least load core    
    print('Core Assignments:\n', cores)
    return cores

def setup(env, num_cores, constraints):
    '''
    returns a new dnn Request(), and an update to the max_ids (tuple)
    size constraints for generation: 
    ['task_size':randint(), 'job_size':randint(), 'req_size':randint(),
    'gen_time': randint()]
    '''
    cores = []
    for i in range(num_cores):
        cores.append(Core(env, i))
    processor = Processor(env, cores, long_flow_first)
    max_ids = {'max_task_id':0, 'max_job_id':0, 'max_req_id':1}

    start_req = Request(0, 'start_req', [[10,15,20,25,30],[30,35,40,45,50],[50,55,60,65,70]], max_ids)
    processor.add_request(start_req)

    env.process(processor.simulate())
    while True:
        yield env.timeout(constraints['gen_time'])
        jobs = []
        for _ in range(constraints['req_size']):
            tasks = []
            for i in range(constraints['job_size']):
                task_size = constraints['task_size']
                tasks.append(Task(max_ids['max_task_id']+1, task_size, 0))
                max_ids['max_task_id'] += 1
            jobs.append(Job(max_ids['max_job_id']+1, tasks, 0))
            max_ids['max_job_id'] += 1
        new_req_id = max_ids['max_req_id']+1
        max_ids['max_req_id'] += 1
        new_req = Request(new_req_id,f'dnn{new_req_id}',jobs, max_ids)
        processor.add_request(new_req)
    
env = simpy.Environment()
env.process(setup(env, NUM_CORES, GEN_CONSTRAINTS))
print('Simulation start')
env.run(until=SIM_TIME)

