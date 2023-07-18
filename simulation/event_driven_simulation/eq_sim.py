from typing import NoReturn
from collections import defaultdict
import matplotlib.pyplot as plt
from numpy import random
import numpy as np
from tqdm import tqdm
from eq_util import SUM_MODEL_SIZES, TOTAL_MODEL_COUNT, DLRM_SIZE, ALEXNET_SIZE, Accelerator, Core, \
    EventGenerator, EventQueue, long_flow_first, round_robin, lookup_table

class Simulator():
    '''
    Simulator for processor core scheduling
    '''
    def __init__(self) -> NoReturn:
        self.event_queue = EventQueue()
        self.generator = EventGenerator()

    def gen(self, model_type, sparsity, alg, accelerators, gen_time, interarrival, poisson):
        '''
        generate and schedule a Request()
        '''
        # generate and schedule request
        req = self.generator.gen_request(model_type, sparsity, gen_time)
        alg(req, accelerators, gen_time)
        self.event_queue.merge([req])
        # update gen time
        if poisson:
            gen_time += random.exponential(interarrival,1)[0]
        else:
            gen_time += interarrival
        return gen_time

    def simulate(self, models, num_acc, num_cores, alg, num_gens, interarrival, gen_time=0, poisson=False):
        '''
        simulate DNN requests given the above constraints
        '''

        accelerators = [Accelerator(id=i, cores=[Core(id=j) for j in range(num_cores)]) for i in range(num_acc)]
        # index to choose next accelerator to schedule to (round robin)
        acc_ind = 0

        completed_reqs = []
        self.generator = EventGenerator()
        # start the simulation with a request
        model_type = random.choice(list(models.keys()))
        #TODO: do object attributes maintain in memory here?
        gen_time = self.gen(model_type, models[model_type], alg, accelerators[acc_ind], \
            gen_time, interarrival, poisson)
        acc_ind = acc_ind + 1 if acc_ind < num_acc - 1 else 0
        num_gens -= 1
        # run simulation
        pbar = tqdm(total=num_gens+1)
        while self.event_queue:
            event = self.event_queue.pop(0)
            # generate a new request
            if num_gens > 0:
                # gen next request if compute is fully finished
                if not self.event_queue:
                    model_type = random.choice(list(models.keys()))
                    gen_time = self.gen(model_type, models[model_type], alg, accelerators[acc_ind], \
                    gen_time, interarrival, poisson)
                    acc_ind = acc_ind + 1 if acc_ind < num_acc - 1 else 0
                    num_gens -= 1
                    pbar.update(1)
                # continue to generate requests until next compute
                while event.t_start >= gen_time:
                    model_type = random.choice(list(models.keys()))
                    gen_time = self.gen(model_type, models[model_type], alg, accelerators[acc_ind], \
                    gen_time, interarrival, poisson)
                    acc_ind = acc_ind + 1 if acc_ind < num_acc - 1 else 0
                    num_gens -= 1
                    pbar.update(1)
            if event.type == 'Request':
                self.event_queue.merge(event.jobs)
                completed_reqs.append(event)
            elif event.type == 'Job':
                self.event_queue.merge(event.tasks)
        pbar.update(1)
        pbar.close()
        return completed_reqs

def print_sim_metrics(reqs, num_cores):
    '''
    print relevant sim metrics to terminal
    '''
    print('\nnum_cores:', num_cores)
    print('sim T:', T)
    times = [req.t_finish - req.t_arrival for req in reqs]
    req_dict = defaultdict(int)
    for req in reqs:
        req_dict[req.dnn] += 1
    print('model types served:', dict(req_dict))
    for time in times:
        if time < 0:
            print('negative time of completion computed. Debug')
    print('num_reqs completed:', len(times))
    print('avg completion time:', round(float(np.mean(times)), 2))
    print('total sim time:', reqs[-1].t_finish)

sim = Simulator()
T = SUM_MODEL_SIZES/TOTAL_MODEL_COUNT
sim_models = {'AlexNet':None,'ResNet18':None,'VGG16':None,'VGG19':None,'BERT':None,'GPT2':None,'DLRM':None}
sim_num_gens = 100
sim_alg = round_robin
sim_poisson = True
sim_completed_reqs = []

load_factor = 0
min_load_t = 2*T
max_load_t = T/50
x, y = [], []

#[[processor type, num_cores, GHz]]
#processors = [['Lightning4GHz', 1000, 4],['Lightning10GHz', 1000, 10],['Lightning100GHz', 1000, 100],\
#    ['H100',18432*12,1.83],['A100',6912*8, 1.41],['P4', 2560*8, 1.114],['T4', 2560*8, 1.59]]

processors = [['Lightning10GHz', 1000, 10],['P4', 2560*8, 1]]

load_factors = [i*10 for i in range(11)]
load_factors.insert(0, 'Load Factor')
avg_comp_times = []

for enum in enumerate(processors):
    ind, processor = enum[0], enum[1]
    x, y = [], []
    while load_factor <= 1:
        sim_interarrival = min_load_t - ((min_load_t) - (max_load_t))*load_factor
        if not sim.event_queue:
            print(f'Load Factor {round(load_factor, 2)}')
            completed_reqs_i = sim.simulate(models=sim_models, num_acc=1, num_cores=processor[1],
            alg=sim_alg, num_gens=sim_num_gens, interarrival=sim_interarrival, poisson=sim_poisson)
        else:
            print("ERROR: Uncompleted requests in previous sim. Debug")
        avg_completion_time = sum([(req.t_finish - req.t_arrival) for req in completed_reqs_i]) \
            /len(completed_reqs_i)
        x.append(load_factor*100)
        y.append(avg_completion_time)
        load_factor += 0.1
    load_factor = 0
    y = [avg / processor[2] for avg in y]
    avg_comp_time = y.copy()
    avg_comp_time.insert(0, processor[0])
    avg_comp_times.append(avg_comp_time)
    # if ind == 4:
    #     plt.plot(x, y, '--', label=processor[ind][0], marker='.')
    # else:
    plt.plot(x, y, label=processor[0], marker='.')

np.savetxt('sim_data.csv', [p for p in zip(load_factors, avg_comp_times[0], avg_comp_times[1], avg_comp_times[2], \
    avg_comp_times[3], avg_comp_times[4], avg_comp_times[5], avg_comp_times[6])], delimiter=',', fmt='%s')

#plt.yscale("log")
#plt.xlabel('Load Factor (%)')
#plt.ylabel('Average Request Completion Time')
#plt.title('1000 Core Lightning Performance vs. ideal GPU')
#plt.legend()
#plt.savefig("thirdfig")
#plt.show()
# simple LeNet sim

# sim = Simulator()
# T = sum([784*300,300*100,100*10,10*10])
# sim_models = {'GPT2':0.8,'DLRM':0.8,'LENET':0.8}
# sim_num_gens = 10
# sim_alg = long_flow_first
# sim_interarrival = 2*T

# completed_reqs_lenet = sim.simulate(models=['LENET'], cores=[Core(0)], alg=sim_alg, \
#     num_gens=sim_num_gens, interarrival=sim_interarrival)

# print_sim_metrics(completed_reqs_lenet, 1)

# sim GPT-2, DLRM, and LeNet with a variety of computation capacity

# sim = Simulator()
# T = int(SUM_MODEL_SIZES/TOTAL_MODEL_COUNT)
# sim_models = {'GPT2':None,'DLRM':None}
# sim_num_gens = 100
# sim_alg = long_flow_first
# # sim_alg = round_robin
# # sim_alg = lookup_table
# sim_interarrival = T
# sim_poisson = True
# sim_completed_reqs = []

# print('Starting simulation...')
# # single-core
# completed_reqs_sc = sim.simulate(models=sim_models, num_acc=1, num_cores=1, \
#     alg=sim_alg, num_gens=sim_num_gens, interarrival=sim_interarrival, poisson=sim_poisson)

# sim_completed_reqs.append(completed_reqs_sc)

# # four cores
# four_cores = [Core(0), Core(1), Core(2), Core(3)]
# if not sim.event_queue:
#     completed_reqs_fc = sim.simulate(models=sim_models, num_acc=1, num_cores=4, \
#         alg=sim_alg, num_gens=sim_num_gens, interarrival=sim_interarrival, poisson=sim_poisson)
#     sim_completed_reqs.append(completed_reqs_fc)
# else:
#     print("ERROR: Uncompleted requests in previous sim. Debug")

# # 50 cores
# fifty_cores = [Core(i) for i in range(50)]
# if not sim.event_queue:
#     completed_reqs_ffc = sim.simulate(models=sim_models, num_acc=1, num_cores=50, \
#         alg=sim_alg, num_gens=sim_num_gens, interarrival=sim_interarrival, poisson=sim_poisson)
#     sim_completed_reqs.append(completed_reqs_ffc)
# else:
#     print("ERROR: Uncompleted requests in previous sim. Debug")

# # 100 cores
# hundred_cores = [Core(i) for i in range(100)]
# if not sim.event_queue:
#     completed_reqs_hc = sim.simulate(models=sim_models, num_acc=1, num_cores=100, \
#         alg=sim_alg, num_gens=sim_num_gens, interarrival=sim_interarrival, poisson=sim_poisson)
#     sim_completed_reqs.append(completed_reqs_hc)
# else:
#     print("ERROR: Uncompleted requests in previous sim. Debug")

# # print sim metrics
# num_cores_list = [1,4,50,100]
# for enum in enumerate(sim_completed_reqs):
#     print_sim_metrics(enum[1], num_cores_list[enum[0]])

# # graph results
# labels = ['Single Core', 'Four Cores', '50 Cores', '100 Cores']
# for enum in enumerate(sim_completed_reqs):
#     ind, data = enum[0], [req.t_finish - req.t_arrival for req in enum[1]]
#     # sort data
#     x = np.sort(data)
#     # calculate CDF values
#     y = 1. * np.arange(len(data)) / (len(data) - 1)
#     # plot CDF
#     plt.plot(x, y, label=labels[ind])

# plt.xlabel('request completion time (100 total reqs)')
# plt.ylabel('$p$')
# plt.title('Poisson Request Generation, λ=2/T (real-time LPT)')
# plt.legend()
# plt.show()

# generate plot of average req completion time across '01%->99%' load

# sim = Simulator()
# T = int(SUM_MODEL_SIZES/TOTAL_MODEL_COUNT)
# sim_models = {'GPT2':0.8,'DLRM':0.8,'LENET':0.8}
# sim_num_gens = 10000
# sim_algs = [long_flow_first, lookup_table, round_robin]
# sim_poisson = False

# labels = ['long_flow_first', 'lookup_table', 'round_robin']

# for enum in enumerate(sim_algs):
#     ind, sim_alg = enum
#     load_factor = 0.01
#     min_load_t = T
#     max_load_t = T/2
#     x, y = [], []
#     while load_factor <= 1:
#         sim_interarrival = min_load_t - ((min_load_t) - (max_load_t))*load_factor
#         if not sim.event_queue:
#             print(f'Load Factor {round(load_factor, 2)}')
#             completed_reqs_i = sim.simulate(models=sim_models, num_acc=1, num_cores=4,
#             alg=sim_alg, num_gens=sim_num_gens, interarrival=sim_interarrival, poisson=sim_poisson)
#         else:
#             print("ERROR: Uncompleted requests in previous sim. Debug")
#         avg_completion_time = sum([(req.t_finish - req.t_arrival) for req in completed_reqs_i]) \
#             /len(completed_reqs_i)
#         x.append(load_factor*100)
#         y.append(avg_completion_time)
#         load_factor += 0.01
#     plt.plot(x, y, label=labels[ind], marker='.')

# plt.xlabel('Load (%)')
# plt.ylabel('Average Request Completion Time')
# plt.title('Scheduling Algorithm Performance Across Load Factors')
# plt.legend()
# plt.show()

# sim large models with variance in # of accelerators and average request completion time

# sim = Simulator()
# T = int(SUM_MODEL_SIZES/TOTAL_MODEL_COUNT)
# sim_models = {'GPT2':None, 'DLRM':None, 'VGG16':None, 'VGG19':None}
# sim_num_gens = 100
# sim_alg = long_flow_first
# sim_poisson = True
# sim_interarrival = T/20


# ghz = [1,4,10,100]
# labels = ['GPU', 'US_4GHz', 'US_10GHz', 'US_100GHz']

# for i in range(4):
#     x, y = [], []
#     for sim_num_cores in range(1, 101):
#         if not sim.event_queue:
#             completed_reqs = sim.simulate(models=sim_models, num_acc=1, num_cores=sim_num_cores, alg=sim_alg,
#             num_gens=sim_num_gens, interarrival=sim_interarrival, poisson=sim_poisson)
#         else:
#             print ("ERROR: Uncompleted requests in previous sim. Debug")
#         avg_completion_time = sum([(req.t_finish - req.t_arrival) for req in completed_reqs]) \
#                 /len(completed_reqs)
#         x.append(sim_num_cores)
#         y.append(avg_completion_time)
#     y = [avg / ghz[i] for avg in y]
#     if i == 0:
#         plt.plot(x, y, '--', label=labels[i], marker='.')
#     else:
#         plt.plot(x, y, label=labels[i], marker='.')

# plt.xlabel('# of Cores')
# plt.ylabel('Average Request Completion Time')
# plt.title('Computation Performance Across Number of Cores and Clock Frequencies')
# plt.legend()
# plt.savefig("firstfig")
# plt.show()

# plt.clf()

# sim = Simulator()
# T = int(SUM_MODEL_SIZES/TOTAL_MODEL_COUNT)
# sim_models = {'GPT2':None, 'DLRM':None, 'VGG16':None, 'VGG19':None}
# sim_num_gens = 100
# sim_alg = long_flow_first
# sim_poisson = True
# sim_interarrival = T/20

# num_cores_list = [1, 2, 5, 10]

# for sim_num_cores in num_cores_list:
#     x, y = [], []
#     for sim_num_acc in range(1,101):
#         if not sim.event_queue:
#             completed_reqs = sim.simulate(models=sim_models, num_acc=sim_num_acc, num_cores=sim_num_cores, alg=sim_alg,
#             num_gens=sim_num_gens, interarrival=sim_interarrival, poisson=sim_poisson)
#         else:
#             print ("ERROR: Uncompleted requests in previous sim. Debug")
#         avg_completion_time = sum([(req.t_finish - req.t_arrival) for req in completed_reqs]) \
#              /len(completed_reqs)
#         x.append(sim_num_acc)
#         y.append(avg_completion_time)
#     plt.plot(x, y, label=f'{sim_num_cores} Cores', marker='.')

# plt.xlabel('# of Accelerators')
# plt.ylabel('Average Request Completion Time')
# plt.title('Computation Performance Across Number of Cores and Accelerators')
# plt.legend()
# plt.savefig("secondfig")
# plt.show()

# plt.clf()

# sim = Simulator()
# T = int(SUM_MODEL_SIZES)
# sim_models = {'ResNet18':None,'AlexNet':None,'VGG16':None,'VGG19':None,'BERT':None,'GPT2':None,'DLRM':None}
# sim_num_gens = 100
# sim_alg = long_flow_first
# sim_poisson = True
# sim_interarrival = 100*T


# ghz = [1,4,10,100]
# labels = ['GPU', 'Photonic 4GHz', 'Photonic 10GHz', 'Photonic 100GHz']

# for i in range(4):
#     x, y = [], []
#     for sim_num_cores in range(5, 101):
#         if not sim.event_queue:
#             completed_reqs = sim.simulate(models=sim_models, num_acc=1, num_cores=sim_num_cores, alg=sim_alg,
#             num_gens=sim_num_gens, interarrival=sim_interarrival, poisson=sim_poisson)
#         else:
#             print ("ERROR: Uncompleted requests in previous sim. Debug")
#         avg_completion_time = sum([(req.t_finish - req.t_arrival) for req in completed_reqs]) \
#                 /len(completed_reqs)
#         x.append(sim_num_cores)
#         y.append(avg_completion_time)
#     y = [avg / ghz[i] for avg in y]
#     if i == 0:
#         plt.plot(x, y, '--', label=labels[i], marker='.')
#     else:
#         plt.plot(x, y, label=labels[i], marker='.')

# plt.xlabel('# of Cores')
# plt.ylabel('Average Request Completion Time')
# plt.title('Computation Performance Across Number of Cores and Clock Frequencies')
# plt.legend()
# plt.savefig("thirdfig")
# plt.show()

# plt.clf()

# sim = Simulator()
# T = int(SUM_MODEL_SIZES/TOTAL_MODEL_COUNT)
# sim_models = {'ResNet18':None,'AlexNet':None,'VGG16':None,'VGG19':None,'BERT':None,'GPT2':None,'DLRM':None}
# sim_num_gens = 100
# sim_alg = long_flow_first
# sim_poisson = True
# sim_interarrival = 100*T

# num_cores_list = [1, 2, 5, 10]

# for sim_num_cores in num_cores_list:
#     x, y = [], []
#     for sim_num_acc in range(1,101):
#         if not sim.event_queue:
#             completed_reqs = sim.simulate(models=sim_models, num_acc=sim_num_acc, num_cores=sim_num_cores, alg=sim_alg,
#             num_gens=sim_num_gens, interarrival=sim_interarrival, poisson=sim_poisson)
#         else:
#             print ("ERROR: Uncompleted requests in previous sim. Debug")
#         avg_completion_time = sum([(req.t_finish - req.t_arrival) for req in completed_reqs]) \
#              /len(completed_reqs)
#         x.append(sim_num_acc)
#         if sim_num_cores == 6912:
#            y.append(avg_completion_time)
#         else:
#             y.append(avg_completion_time/36.57)
#     if sim_num_cores == 6912:
#         plt.plot(x, y, label='GPU', marker='.')
#     else:
#         plt.plot(x, y, label=f'{sim_num_cores} Photonic Cores', marker='.')

# plt.xlabel('# of Accelerators')
# plt.ylabel('Average Request Completion Time')
# plt.title('Computation Performance Across Number of Cores and Accelerators')
# plt.legend()
# plt.savefig("fifthfig")
# plt.show()

# sim = Simulator()
# T = int(SUM_MODEL_SIZES/TOTAL_MODEL_COUNT)
# sim_models = {'ResNet18':None,'AlexNet':None,'VGG16':None,'VGG19':None,'BERT':None,'GPT2':None,'DLRM':None}
# sim_num_gens = 100
# sim_alg = long_flow_first
# sim_poisson = True
# sim_interarrival = 100*T

# num_cores_list = [6912, 1, 5, 10, 200]

# for sim_num_cores in num_cores_list:
#     x, y = [], []
#     for sim_num_acc in range(1,101):
#         if not sim.event_queue:
#             completed_reqs = sim.simulate(models=sim_models, num_acc=sim_num_acc, num_cores=sim_num_cores, alg=sim_alg,
#             num_gens=sim_num_gens, interarrival=sim_interarrival, poisson=sim_poisson)
#         else:
#             print ("ERROR: Uncompleted requests in previous sim. Debug")
#         avg_completion_time = sum([(req.t_finish - req.t_arrival) for req in completed_reqs]) \
#              /len(completed_reqs)
#         x.append(sim_num_acc)
#         if sim_num_cores == 200:
#            y.append(avg_completion_time/36.57)
#         else:
#             y.append(avg_completion_time)
#     if sim_num_cores == 6912:
#         plt.plot(x, y, label='GPU', marker='.')
#     else:
#         plt.plot(x, y, label=f'{sim_num_cores} Photonic Cores', marker='.')

# plt.xlabel('# of Accelerators')
# plt.ylabel('Average Request Completion Time')
# plt.title('Computation Performance Across Number of Cores and Accelerators')
# plt.legend()
# plt.savefig("fourthfig")
# plt.show()

# Comparing scheduling algorithms

# sim = Simulator()
# T = DLRM_SIZE
# sim_models = {'DLRM':None}
# sim_num_gens = 10000
# sim_algs = [long_flow_first, round_robin]
# sim_poisson = True
# sim_interarrival = T/100
# sim_num_cores = 4
# sim_completed_reqs = []

# for sim_alg in sim_algs:
#     if not sim.event_queue:
#         completed_reqs = sim.simulate(models=sim_models, num_acc=1, num_cores=sim_num_cores, \
#             alg=sim_alg, num_gens=sim_num_gens, interarrival=sim_interarrival, poisson=sim_poisson)
#         sim_completed_reqs.append(completed_reqs)
#     else:
#         print("ERROR: Uncompleted requests in previous sim. Debug")

# # graph results
# labels = ['long flow first', 'round robin']
# for enum in enumerate(sim_completed_reqs):
#     ind, data = enum[0], [req.t_finish - req.t_arrival for req in enum[1]]
#     # sort data
#     x = np.sort(data)
#     print(f'avg completion time {labels[ind]}:', round(float(np.mean(data)), 2))
#     # calculate CDF values
#     y = 1. * np.arange(len(data)) / (len(data) - 1)
#     # plot CDF
#     plt.plot(x, y, label=labels[ind])

# plt.xlabel('request completion time (100 total reqs)')
# plt.ylabel('$p$')
# plt.title('Long Flow First vs. Round Robin DLRM Request Completion Time')
# plt.legend()
# plt.show()

# sim = Simulator()
# T = DLRM_SIZE
# sim_models = {'DLRM':None}
# sim_num_gens = 10000
# sim_algs = [long_flow_first, round_robin]
# sim_poisson = True
# sim_interarrival = T
# sim_num_cores = 1
# sim_completed_reqs = []

# for sim_alg in sim_algs:
#     if not sim.event_queue:
#         completed_reqs = sim.simulate(models=sim_models, num_acc=1, num_cores=sim_num_cores, \
#             alg=sim_alg, num_gens=sim_num_gens, interarrival=sim_interarrival, poisson=sim_poisson)
#         sim_completed_reqs.append(completed_reqs)
#     else:
#         print("ERROR: Uncompleted requests in previous sim. Debug")

# # graph results
# labels = ['long flow first', 'round robin']
# for enum in enumerate(sim_completed_reqs):
#     ind, data = enum[0], [req.t_finish - req.t_arrival for req in enum[1]]
#     # sort data
#     x = np.sort(data)
#     print(f'avg completion time {labels[ind]}:', round(float(np.mean(data)), 2))
#     # calculate CDF values
#     y = 1. * np.arange(len(data)) / (len(data) - 1)
#     # plot CDF
#     plt.plot(x, y, label=labels[ind])

# plt.xlabel('request completion time (100 total reqs)')
# plt.ylabel('$p$')
# plt.title('Long Flow First vs. Round Robin DLRM Request Completion Time')
# plt.legend()
# plt.show()
