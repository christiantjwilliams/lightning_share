from tqdm import tqdm
from old_util import Event, Job, Task, Request, Core
from numpy import random
import numpy as np
import matplotlib.pyplot as plt

GPT2_LAYER_SIZES = [50257*1280]
for i in range(36):
    GPT2_LAYER_SIZES.append((1280*1280)+(1280*1280*3)+(1280*1280*4))
GPT2_LAYER_SIZES.append(1280*4*1280)

DLRM_LAYER_SIZES = [9980333,36084,17217,7378,20134,3,7112,1442,61, \
    9758201,1333352,313829,10,2208,11156,112,4,970,14,9994222,7267859, \
    9946608,415421,12420,101,36,
    13,512,256,415,512,512,256]

LENET_LAYER_SIZES = [784*300,300*100,100*10,10*10]

TOTAL_MODEL_COUNT = 3
SUM_MODEL_SIZES = sum(DLRM_LAYER_SIZES) + sum(GPT2_LAYER_SIZES) + sum(LENET_LAYER_SIZES)

# T = approx size of LeNet
T = 705600
SIM_TIME = 100*T
# scenario 1, generate request every T secs, where T=serve time of LeNet (266300)
GEN_CONSTRAINTS_1 = {'task_size':[1,7], 'job_size':[5,10], 'req_size':[1,10], 'gen_time':[T,T]}
# scenario 2, generate request every T/2 secs
GEN_CONSTRAINTS_2 = {'task_size':[1,7], 'job_size':[5,10], 'req_size':[1,10], 'gen_time':[T/2,T/2]}
# scenario 3, generate request randomly between T/2 and 2T seconds
GEN_CONSTRAINTS_3 = {'task_size':[1,7], 'job_size':[5,10], 'req_size':[1,10], 'gen_time':[T/2,2*T]}

# lookup table scenario, specifies a time interval for which to create a lookup table
GEN_CONSTRAINTS_4 = {'task_size':[1,7], 'job_size':[5,10], 'req_size':[1,10], 'gen_time':[T/2,2*T], 'update_table_time':5*T}

def long_flow_first(tasks, cores):
    # sort at task level based on tasks's longest flow: large task first
    sorted_tasks = sorted(tasks, key=lambda t: t.size, reverse=True)
    # assign tasks on cores based on core weights
    #for t in tqdm(sorted_tasks, desc="Scheduling tasks..."):
    for t in sorted_tasks:
        min_load_core = min(cores, key=lambda core: core.load_on_core)
        min_load_core.add_task_to_core(t)  # assign the longest flow to the least load core    
    return cores

def round_robin(tasks, cores):
    i = 0
    for t in tasks:
        if i >= len(cores):
            i = 0
        cores[i].add_task_to_core(t)
        i += 1
    return cores

def lookup_table(tasks, cores, lookup_table, permute=True):
    # sort at task level based on tasks's longest flow: large task first
    sorted_tasks = sorted(tasks, key=lambda t: t.size, reverse=True)
    lookup_table_keys = list(lookup_table.keys())
    if permute:
        permuted_lookup_table_keys = np.random.permutation(lookup_table_keys)
        lookup_table = {k: lookup_table[k] for k in permuted_lookup_table_keys}
    for t in sorted_tasks:
        min_load_core = min(lookup_table, key=lambda core: lookup_table[core])
        min_load_core.add_task_to_core(t)  # assign the longest flow to the least load core 
    return cores
            
class Simulator():
    def __init__(self, cores, alg, sim_time, gen_constraints, gen='RANDOM', poisson=None):
        self.cores = cores
        self.alg = alg
        self.sim_time = sim_time
        self.constraints = gen_constraints
        self.gen = gen
        self.poisson = poisson
        if self.constraints['gen_time'][0] == self.constraints['gen_time'][1]:
            self.gen_time = self.constraints['gen_time'][0]
        else:
            self.gen_time = random.randint(self.constraints['gen_time'][0], \
                self.constraints['gen_time'][1])
        self.lookup_table = {core:core.load_on_core for core in self.cores}
        if 'update_lookup_table' in self.constraints:
            self.update_table_time = self.constraints['update_lookup_table']
        else:
            self.update_table_time = float('inf')
        self.gen_times = [self.gen_time]
        self.max_ids = {'max_task_id':0, 'max_job_id':0, 'max_req_id':0}
        self.elapsed = 0
        # {job_id:Job()}
        self.jobs_dict = {}
        # {req_id:Request()}
        self.reqs_dict = {}

    def gen(self):
        '''
        returns a new randomized dnn Request()
        '''
        req_min, req_max = self.constraints['req_size'][0], self.constraints['req_size'][1]
        req_size = random.randint(req_min, req_max)
        jobs = []
        self.max_ids['max_req_id'] += 1
        new_req_id = self.max_ids['max_req_id']
        for _ in range(req_size):
            job_min, job_max = self.constraints['job_size'][0], self.constraints['job_size'][1]
            job_size = random.randint(job_min, job_max)
            tasks = []
            self.max_ids['max_job_id'] += 1
            new_job_id = self.max_ids['max_job_id']
            for i in range(job_size):
                task_min, task_max = self.constraints['task_size'][0], self.constraints['task_size'][1]
                task_size = random.randint(task_min, task_max)
                self.max_ids['max_task_id'] += 1
                new_task_id = self.max_ids['max_task_id']
                tasks.append(Task(new_task_id, task_size, self.elapsed, new_job_id, new_req_id))
            new_job = Job(new_job_id, tasks, self.elapsed, new_req_id)
            jobs.append(new_job)
            self.jobs_dict[new_job_id] = new_job
        new_req = Request(new_req_id,f'dnn{new_req_id}',jobs, self.elapsed)
        self.reqs_dict[new_req_id] = new_req
        return new_req
    
    def gen_vanilla(self, num_layers=3):
        '''
        returns a same-layer-size dnn request, sizes similar to LeNet
        '''
        req_id = self.max_ids['max_req_id']+1
        self.max_ids['max_req_id'] += 1
        lenet_jobs = []
        layer_sizes = [784*300,784*300,784*300]
        for i in range(1,num_layers+1):
            job_id = self.max_ids['max_job_id']+i
            new_task = Task(self.max_ids['max_task_id']+i,layer_sizes[i-1], self.elapsed, job_id, req_id)
            new_job = Job(job_id, [new_task], self.elapsed, req_id)
            self.jobs_dict[job_id] = new_job
            lenet_jobs.append(new_job)
        self.max_ids['max_task_id'] += num_layers
        self.max_ids['max_job_id'] += num_layers
        new_req = Request(req_id, 'LeNet', lenet_jobs, self.elapsed)
        self.reqs_dict[req_id] = new_req
        return new_req

    def gen_lenet(self, num_layers=len(LENET_LAYER_SIZES)):
        '''
        returns a LeNet-like dnn Request()
        J0:784*300 --> J1:300*100 --> J2:100x10 --> J3:10x10
        '''
        req_id = self.max_ids['max_req_id']+1
        self.max_ids['max_req_id'] += 1
        lenet_jobs = []
        layer_sizes = LENET_LAYER_SIZES
        for i in range(1,num_layers+1):
            job_id = self.max_ids['max_job_id']+i
            new_task = Task(self.max_ids['max_task_id']+i,layer_sizes[i-1], self.elapsed, job_id, req_id)
            new_job = Job(job_id, [new_task], self.elapsed, req_id)
            self.jobs_dict[job_id] = new_job
            lenet_jobs.append(new_job)
        self.max_ids['max_task_id'] += num_layers
        self.max_ids['max_job_id'] += num_layers
        new_req = Request(req_id, 'LeNet', lenet_jobs, self.elapsed)
        self.reqs_dict[req_id] = new_req
        return new_req

    def gen_gpt2(self, num_layers=len(GPT2_LAYER_SIZES)):
        '''
        returns a GPT2-like dnn Request()
        '''
        req_id = self.max_ids['max_req_id']+1
        self.max_ids['max_req_id'] += 1
        gpt2_jobs = []
        layer_sizes = GPT2_LAYER_SIZES
        for i in range(1,num_layers+1):
            job_id = self.max_ids['max_job_id']+i
            new_task = Task(self.max_ids['max_task_id']+i,layer_sizes[i-1], self.elapsed, job_id, req_id)
            new_job = Job(job_id, [new_task], self.elapsed, req_id)
            self.jobs_dict[job_id] = new_job
            gpt2_jobs.append(new_job)
        self.max_ids['max_task_id'] += num_layers
        self.max_ids['max_job_id'] += num_layers
        new_req = Request(req_id, 'GPT2', gpt2_jobs, self.elapsed)
        self.reqs_dict[req_id] = new_req
        return new_req
        
    
    def gen_dlrm(self,num_layers=len(DLRM_LAYER_SIZES)):
        '''
        returns a DLRM-like dnn Request()
        '''
        req_id = self.max_ids['max_req_id']+1
        self.max_ids['max_req_id'] += 1
        dlrm_jobs = []
        layer_sizes = DLRM_LAYER_SIZES
        for i in range(1,num_layers+1):
            job_id = self.max_ids['max_job_id']+i
            new_task = Task(self.max_ids['max_task_id']+i,layer_sizes[i-1], self.elapsed, job_id, req_id)
            new_job = Job(job_id, [new_task], self.elapsed, req_id)
            self.jobs_dict[job_id] = new_job
            dlrm_jobs.append(new_job)
        self.max_ids['max_task_id'] += num_layers
        self.max_ids['max_job_id'] += num_layers
        new_req = Request(req_id, 'DLRM', dlrm_jobs, self.elapsed)
        self.reqs_dict[req_id] = new_req
        return new_req
    
    def simulate(self, sim=True):
        # run simulation
        gen_times = []
        while sim:
            # move time forward wrt to task completion, until next scheduling event
            while self.elapsed < self.gen_time:
                if any(core.tasks_on_core for core in self.cores):
                    # check for cores with tasks
                    working_cores = [core for core in self.cores if core.tasks_on_core]
                    # find the smallest, subsequent compute task among working cores
                    min_time_core = min(working_cores, key=lambda core: core.tasks_on_core[0].size)
                    # TODO: currently time == size of vv product, change
                    min_time = min_time_core.tasks_on_core[0].size
                    #print('comp_time:', min_time)
                    if self.elapsed + min_time < self.sim_time:
                        if self.elapsed + min_time >= self.update_table_time:
                            self.lookup_table = {core:core.load_on_core for core in self.cores}                           
                        if self.elapsed + min_time <= self.gen_time:
                            self.elapsed += min_time
                            # make minimum progress on computation of each core
                            for core in working_cores:
                                core.compute(min_time, self.elapsed, self.jobs_dict, self.reqs_dict)
                        else:
                            self.elapsed = self.gen_time
                            for core in working_cores:
                                core.compute(self.gen_time-self.elapsed, self.elapsed, self.jobs_dict, self.reqs_dict)
                        #print('current core loads:', self.cores)
                    else:
                        self.elapsed = self.sim_time
                        break
                else:
                    self.elapsed = self.gen_time
            
            if self.elapsed < self.sim_time:
                past_gen_time = self.gen_time
                if self.poisson:
                    prev_gen_time = self.gen_time
                    self.gen_time += random.exponential(self.poisson,1)[0]
                    #print('interarrival: ', self.gen_time-prev_gen_time)
                elif self.constraints['gen_time'][0] == self.constraints['gen_time'][1]:
                    self.gen_time += self.constraints['gen_time'][0]
                else:
                    self.gen_time += random.randint(self.constraints['gen_time'][0], \
                        self.constraints['gen_time'][1])
                self.gen_times.append(self.gen_time - past_gen_time)
                # schedule new request to cores
                if self.gen == 'RANDOM':
                    next_request = self.gen()
                elif self.gen == 'VANILLA':
                    next_request = self.gen_vanilla()
                elif self.gen == 'VARIED':
                    model_ind = random.randint(2)
                    #next_request = self.gen_gpt2()
                    if model_ind == 0:
                        next_request = self.gen_lenet()
                    elif model_ind == 1:
                        next_request = self.gen_gpt2()
                    else:
                        next_request = self.gen_dlrm()
                for job in next_request.jobs:
                    self.cores = long_flow_first(job.tasks, self.cores)
                    #print('\nCore Assignments:\n', self.cores, '\n')
            else:
                sim = False

        # print completed jobs and requests, along with completion times
        finished_job_ids = sorted([job for job in self.jobs_dict if self.jobs_dict[job].status])
        finished_jobs = [self.jobs_dict[job] for job in finished_job_ids]
        finished_req_ids = sorted([req for req in self.reqs_dict if self.reqs_dict[req].status])
        finished_reqs = [self.reqs_dict[req] for req in finished_req_ids]
        # print average job completion time and average request completion time
        job_times = [job.t_finish - job.t_start for job in finished_jobs]
        req_times = [req.t_finish - req.t_start for req in finished_reqs]
        print('\nNUMBER OF CORES:', len(self.cores))
        print('number of jobs completed:', len(job_times))
        print('number of requests completed:', len(req_times))
        print('average job completion time:', round(float(np.mean(job_times)), 2))
        print('average request completion time:', round(float(np.mean(req_times)), 2))

        return np.sort(req_times), round(float(np.mean(self.gen_times))/T, 2)



# sim scenario 1
# sim = Simulator(cores=[Core(0)], alg=long_flow_first, sim_time=SIM_TIME,\
#  gen_constraints=GEN_CONSTRAINTS_1, gen='MNIST')
# sim1_req_times, avg_gen_time_1 = sim.simulate()
# #print('sim1_job_times:', sim1_job_times)

# # sim scenario 2
# sim = Simulator(cores=[Core(0)], alg=long_flow_first, sim_time=SIM_TIME,\
#  gen_constraints=GEN_CONSTRAINTS_2, gen='MNIST')
# sim2_req_times, avg_gen_time_2 = sim.simulate()

# #print('sim2_job_times:', sim2_job_times)

# # sim scenario 3.1
# sim = Simulator(cores=[Core(0)], alg=long_flow_first, sim_time=SIM_TIME,\
#  gen_constraints=GEN_CONSTRAINTS_3, gen='MNIST')
# sim3_1_req_times, avg_gen_time_3_1 = sim.simulate()

# # sim scenario 3.2
# sim = Simulator(cores=[Core(0)], alg=long_flow_first, sim_time=SIM_TIME,\
#  gen_constraints=GEN_CONSTRAINTS_3, gen='MNIST')
# sim3_2_req_times, avg_gen_time_3_2 = sim.simulate()

# # sim scenario 3.3
# sim = Simulator(cores=[Core(0)], alg=long_flow_first, sim_time=SIM_TIME,\
#  gen_constraints=GEN_CONSTRAINTS_3, gen='MNIST')
# sim3_3_req_times, avg_gen_time_3_3 = sim.simulate()

# # sim scenario 3.4
# sim = Simulator(cores=[Core(0)], alg=long_flow_first, sim_time=SIM_TIME,\
#  gen_constraints=GEN_CONSTRAINTS_3, gen='MNIST')
# sim3_4_req_times, avg_gen_time_3_4 = sim.simulate()

# print('sim3_job_times:', sim3_job_times)

# plot the CDFs for each scenario
# p1 = 1. * np.arange(len(sim1_req_times)) / (len(sim1_req_times) - 1)
# p2 = 1. * np.arange(len(sim2_req_times)) / (len(sim2_req_times) - 1)
# p3_1 = 1. * np.arange(len(sim3_1_req_times)) / (len(sim3_1_req_times) - 1)
# p3_2 = 1. * np.arange(len(sim3_2_req_times)) / (len(sim3_2_req_times) - 1)
# p3_3 = 1. * np.arange(len(sim3_3_req_times)) / (len(sim3_3_req_times) - 1)
# p3_4 = 1. * np.arange(len(sim3_4_req_times)) / (len(sim3_4_req_times) - 1)

# sim1_req_times = np.insert(sim1_req_times, 0, 0)
# sim2_req_times = np.insert(sim2_req_times, 0, 0)
# sim3_1_req_times = np.insert(sim3_1_req_times, 0, 0)
# sim3_2_req_times = np.insert(sim3_2_req_times, 0, 0)
# sim3_3_req_times = np.insert(sim3_3_req_times, 0, 0)
# sim3_4_req_times = np.insert(sim3_4_req_times, 0, 0)
# sim1_req_times = np.append(sim1_req_times, max(sim1_req_times)*100)
# sim2_req_times = np.append(sim2_req_times, max(sim2_req_times)*100)
# sim3_1_req_times = np.append(sim3_1_req_times, max(sim3_1_req_times))
# sim3_2_req_times = np.append(sim3_2_req_times, max(sim3_2_req_times))
# sim3_3_req_times = np.append(sim3_3_req_times, max(sim3_3_req_times))
# sim3_4_req_times = np.append(sim3_4_req_times, max(sim3_4_req_times))

# p1 = np.insert(p1, 0, 0)
# p2 = np.insert(p2, 0, 0)
# p3_1 = np.insert(p3_1, 0, 0)
# p3_2 = np.insert(p3_2, 0, 0)
# p3_3 = np.insert(p3_3, 0, 0)
# p3_4 = np.insert(p3_4, 0, 0)

# p1 = np.append(p1, 1)
# p2 = np.append(p2, 1)
# p3_1 = np.append(p3_1, 1)
# p3_2 = np.append(p3_2, 1)
# p3_3 = np.append(p3_3, 1)
# p3_4 = np.append(p3_4, 1)

# plt.plot(sim1_req_times, p1, label = 'scenario 1')
# plt.plot(sim2_req_times, p2, label = 'scenario 2')
# plt.plot(sim3_1_req_times, p3_1, label = 'scenario 3.1 \navg_gen_time=' + str(avg_gen_time_3_1) + 'T')
# plt.plot(sim3_2_req_times, p3_2, label = 'scenario 3.2\navg_gen_time=' + str(avg_gen_time_3_2) + 'T')
# plt.plot(sim3_3_req_times, p3_3, label = 'scenario 3.3\navg_gen_time=' + str(avg_gen_time_3_3) + 'T')
# plt.plot(sim3_4_req_times, p3_4, label = 'scenario 3.4\navg_gen_time=' + str(avg_gen_time_3_4) + 'T')

# plt.legend()
# #plt.title('scenario 1')
# plt.xlabel('request completion time')
# plt.ylabel('$p$')

# # ax2 = fig.add_subplot(132)
# # ax2.plot(sim2_req_times, p2)
# # ax2.set_title('scenario 2')
# # ax2.set_xlabel('request completion time')
# # ax2.set_ylabel('$p$')

# # ax3 = fig.add_subplot(133)
# # ax3.plot(sim3_req_times, p3)
# # ax3.set_title('scenario 3')
# # ax3.set_xlabel('request completion time')
# # ax3.set_ylabel('$p$')

# #fig.tight_layout()
# plt.show()


# sim scenario poisson 1/T
# sim = Simulator(cores=[Core(0)], alg=long_flow_first, sim_time=SIM_TIME,\
#  gen_constraints=GEN_CONSTRAINTS_3, gen='VANILLA', poisson=T)
# poisson_1_req_times, poisson_1_avg_gen_time = sim.simulate()

# p1 = 1. * np.arange(len(poisson_1_req_times)) / (len(poisson_1_req_times) - 1)

# poisson_1_req_times = np.insert(poisson_1_req_times, 0, 0)
# poisson_1_req_times = np.append(poisson_1_req_times, max(poisson_1_req_times)*100)

# p1 = np.insert(p1, 0, 0)
# p1 = np.append(p1, 1)
# plt.plot(poisson_1_req_times, p1, label = 'λ=1/T')

# # sim scenario poisson 2/T
# sim = Simulator(cores=[Core(0)], alg=long_flow_first, sim_time=SIM_TIME,\
#  gen_constraints=GEN_CONSTRAINTS_3, gen='VANILLA', poisson=T/2)
# poisson_1_req_times, poisson_1_avg_gen_time = sim.simulate()

# p1 = 1. * np.arange(len(poisson_1_req_times)) / (len(poisson_1_req_times) - 1)

# poisson_1_req_times = np.insert(poisson_1_req_times, 0, 0)
# poisson_1_req_times = np.append(poisson_1_req_times, max(poisson_1_req_times)*100)

# p1 = np.insert(p1, 0, 0)
# p1 = np.append(p1, 1)
# plt.plot(poisson_1_req_times, p1, label = 'λ=2/T')

# # sim scenario poisson 1/2T
# sim = Simulator(cores=[Core(0)], alg=long_flow_first, sim_time=SIM_TIME,\
#  gen_constraints=GEN_CONSTRAINTS_3, gen='VANILLA', poisson=2*T)
# poisson_1_req_times, poisson_1_avg_gen_time = sim.simulate()

# p1 = 1. * np.arange(len(poisson_1_req_times)) / (len(poisson_1_req_times) - 1)

# poisson_1_req_times = np.insert(poisson_1_req_times, 0, 0)
# poisson_1_req_times = np.append(poisson_1_req_times, max(poisson_1_req_times)*100)

# p1 = np.insert(p1, 0, 0)
# p1 = np.append(p1, 1)
# plt.plot(poisson_1_req_times, p1, label = 'λ=1/2T')

# plt.legend()
# plt.title('Poisson, Vanilla Generation Schema')
# plt.xlabel('request completion time')
# plt.ylabel('$p$')

# plt.show()

# sim DLRM, GPT-2, and LeNet with a variety of computation capacity

univ_gen_factor_1 = 50
univ_gen_factor_2 = 50
sim_time_factor = 100
sim_alg = long_flow_first
sim_poisson = None

# single-core

T = SUM_MODEL_SIZES/TOTAL_MODEL_COUNT
gen_constraints = {'gen_time':[T/univ_gen_factor_1,T/univ_gen_factor_2], 'update_table_time':5*T}

sim = Simulator(cores=[Core(0)], alg=sim_alg, sim_time=sim_time_factor*T,\
 gen_constraints=gen_constraints, gen='VARIED', poisson=sim_poisson)
varied_req_times, varied_avg_gen_time = sim.simulate()

if varied_req_times.any():
    p1 = 1. * np.arange(len(varied_req_times)) / (len(varied_req_times) - 1)

    varied_req_times = np.insert(varied_req_times, 0, 0)
    varied_req_times = np.append(varied_req_times, max(varied_req_times)*100)

    p1 = np.insert(p1, 0, 0)
    p1 = np.append(p1, 1)
    plt.plot(varied_req_times, p1, label = 'Single Core')

# four cores

NUM_CORES = 4
sim_cores = []
for i in range(NUM_CORES):
    sim_cores.append(Core(i))

#T = SUM_MODEL_SIZES/NUM_OF_MODELS
gen_constraints = {'gen_time':[T/univ_gen_factor_1,T/univ_gen_factor_2], 'update_table_time':5*T}

sim = Simulator(cores=sim_cores, alg=sim_alg, sim_time=sim_time_factor*T,\
 gen_constraints=gen_constraints, gen='VARIED', poisson=sim_poisson)
varied_req_times, varied_avg_gen_time = sim.simulate()

if varied_req_times.any():
    p1 = 1. * np.arange(len(varied_req_times)) / (len(varied_req_times) - 1)

    varied_req_times = np.insert(varied_req_times, 0, 0)
    varied_req_times = np.append(varied_req_times, max(varied_req_times)*100)

    p1 = np.insert(p1, 0, 0)
    p1 = np.append(p1, 1)
    plt.plot(varied_req_times, p1, label = 'Four Cores')

# fifty cores

NUM_CORES = 50
sim_cores = []
for i in range(NUM_CORES):
    sim_cores.append(Core(i))

#T = SUM_MODEL_SIZES
gen_constraints = {'gen_time':[T/univ_gen_factor_1,T/univ_gen_factor_2], 'update_table_time':5*T}

sim = Simulator(cores=sim_cores, alg=sim_alg, sim_time=sim_time_factor*T,\
 gen_constraints=gen_constraints, gen='VARIED', poisson=sim_poisson)
varied_req_times, varied_avg_gen_time = sim.simulate()

if varied_req_times.any():
    p1 = 1. * np.arange(len(varied_req_times)) / (len(varied_req_times) - 1)

    varied_req_times = np.insert(varied_req_times, 0, 0)
    varied_req_times = np.append(varied_req_times, max(varied_req_times)*100)

    p1 = np.insert(p1, 0, 0)
    p1 = np.append(p1, 1)
    plt.plot(varied_req_times, p1, label = '50 Cores')

# 100 cores

NUM_CORES = 100
sim_cores = []
for i in range(NUM_CORES):
    sim_cores.append(Core(i))

#T = SUM_MODEL_SIZES
gen_constraints = {'gen_time':[T/univ_gen_factor_1,T/univ_gen_factor_2], 'update_table_time':5*T}

sim = Simulator(cores=sim_cores, alg=sim_alg, sim_time=sim_time_factor*T,\
 gen_constraints=gen_constraints, gen='VARIED', poisson=sim_poisson)
varied_req_times, varied_avg_gen_time = sim.simulate()

if varied_req_times.any():
    p1 = 1. * np.arange(len(varied_req_times)) / (len(varied_req_times) - 1)

    varied_req_times = np.insert(varied_req_times, 0, 0)
    varied_req_times = np.append(varied_req_times, max(varied_req_times)*100)

    p1 = np.insert(p1, 0, 0)
    p1 = np.append(p1, 1)
    plt.plot(varied_req_times, p1, label = '100 Cores')

plt.legend()
plt.title('Gen Time T/10')
plt.xlabel('request completion time')
plt.ylabel('$p$')

#plt.xlim(-10**6, 10**10)

plt.show()