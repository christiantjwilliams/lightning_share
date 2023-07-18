from collections import defaultdict
import matplotlib.pyplot as plt
from numpy import random
import numpy as np
from tqdm import tqdm
from eq_util import Accelerator, Core, Job, Task, \
    EventGenerator, EventQueue, long_flow_first, round_robin, lookup_table
import time
import argparse
import pickle
import os
import csv

def ParseOpt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--proc_id', type=int, help='processor id of simulation')
    parser.add_argument('--traffic', type=int, help='gbps of traffic')
    parser.add_argument('--inter', type=int, help='interarrival of simulation')
    parser.add_argument('--time', type=int, help='runtime of simulation in terms of T')
    parser.add_argument('--core_util', type=bool, help='turn on core utilization tracking')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    return opt

opt = ParseOpt()

processors = [
            ['Lightning-1-200-100', 300, 1], 
            ['Brainwave', 96000, 0.6],
            ['A100', 6912*8 + 432*3, 1.41],
            ['P4', 2560*8, 1.114],
            ['DPU', 6912*8, 1.41]
        ]

job_times = [] # list of times for each job (used for plotting average job completion time)

class Simulator():
    '''
    Simulator for processor core scheduling
    '''
    def __init__(self):
        self.event_queue = EventQueue()
        self.generator = EventGenerator()

    def gen(self, model_type, sparsity, alg, accelerators, gen_time, interarrival, dp_latency, poisson):
        '''
        generate and schedule a Request()
        '''
        # generate and schedule request
        req = self.generator.gen_request(model_type, sparsity, gen_time, accelerators)
        #req = alg(req, accelerators, gen_time, dp_latency)
        self.event_queue.merge([req])
        # update gen time
        if poisson:
            gen_time += random.exponential(interarrival,1)[0]
        else:
            gen_time += interarrival
        return req, gen_time

    def log_request(self, req_type, index, request, utilized_cores=None, num_cores=None):
        '''
        log request data in memory in case simulation does not run to completion
        '''
        
        file_path = f'{req_type}_reqs/proc_{opt.proc_id}_traffic_{opt.traffic}_{req_type[0:3]}_reqs.csv'
        if index == 0 and os.path.isfile(file_path):
            os.remove(file_path)

        core_utilization = utilized_cores/num_cores if utilized_cores else None
        completion_time = request.t_finish - request.t_arrival if request.t_finish else None

        with open(file_path, "a", newline="") as csvf:
            fieldnames = ["req_ID", "utilized_cores", "num_cores", "core_utilization", "req_t_arrival", "req_t_finish", "completion_time", "req_dnn"]
            writer = csv.DictWriter(csvf, fieldnames=fieldnames)
            if index == 0:
                writer.writeheader()
            writer.writerow({
                "req_ID": request.id,
                "utilized_cores": utilized_cores,
                "num_cores": num_cores,
                "core_utilization": core_utilization,
                "req_t_arrival": request.t_arrival,
                "req_t_finish": request.t_finish,
                "completion_time": completion_time,
                "req_dnn": request.dnn
                })
        return index + 1

    def simulate(self, models, num_acc, num_cores, alg, interarrival, dp_latency, end_time=0, poisson=False):
        '''
        simulate DNN requests given the above constraints
        '''

        accelerators = [Accelerator(id=i, cores=[Core(id=j) for j in range(num_cores)], dp_latency=dp_latency) for i in range(num_acc)]
        # index to choose next accelerator to schedule to (round robin) TODO: just does one acc for now
        acc_ind = 0
        gen_time = 0
        gen_index = 0
        req_index = 0
        opt = ParseOpt()
        completed_reqs = {}
        generated_reqs = {}
        self.generator = EventGenerator()
        # start the simulation with a request
        model_type = random.choice(list(models.keys()))
        gen_req, gen_time = self.gen(model_type, models[model_type], alg, accelerators[acc_ind], gen_time, interarrival, dp_latency, poisson)
        generated_reqs[gen_req.id] = gen_req
        gen_index = self.log_request('generated', gen_index, gen_req)
        acc_ind = acc_ind + 1 if acc_ind < num_acc - 1 else 0
        # run simulation
        continue_sim = True
        requests_in_sim = 1
        print('requests in sim: ', requests_in_sim)
        last_event_type = None
        start_tasks = time.time()
        while continue_sim:
            
            # get next event off the event queue
            event = self.event_queue.pop()

            if event.type != last_event_type:
                end_tasks = time.time()
                print(f'time to handle {last_event_type}(s): ', end_tasks - start_tasks)
                if last_event_type == 'Job':
                    job_times.append(end_tasks - start_tasks)
                print(f'currently handling {event.type}(s)...')
                start_tasks = time.time()
                last_event_type = event.type

            start = time.time()
            # generate a new request
            if gen_time <= end_time:
                # gen next request if compute is fully finished
                if not self.event_queue:
                    model_type = random.choice(list(models.keys()))
                    gen_req, gen_time = self.gen(model_type, models[model_type], alg, accelerators[acc_ind], gen_time, interarrival, dp_latency, poisson)
                    generated_reqs[gen_req.id] = gen_req
                    gen_index = self.log_request('generated', gen_index, gen_req)
                    acc_ind = acc_ind + 1 if acc_ind < num_acc - 1 else 0
                    requests_in_sim += 1
                    print('requests in sim: ', requests_in_sim)
                # continue to generate requests until next compute
                while event.t_start >= gen_time:
                    model_type = random.choice(list(models.keys()))
                    gen_req, gen_time = self.gen(model_type, models[model_type], alg, accelerators[acc_ind], gen_time, interarrival, dp_latency, poisson)
                    generated_reqs[gen_req.id] = gen_req
                    gen_index = self.log_request('generated', gen_index, gen_req)
                    acc_ind = acc_ind + 1 if acc_ind < num_acc - 1 else 0
                    requests_in_sim += 1
                    print('requests in sim: ', requests_in_sim)
            end = time.time()
            block_1_time = end - start
            
            start = time.time()
            if event.type == 'Request':
                gen_job = self.generator.gen_job(accelerators[acc_ind], event.next_job_id, event.id, event.dnn, 0, 0, event.t_arrival) 
                self.event_queue.merge([gen_job])
                generated_reqs[event.id] = event
            end = time.time()
            block_2_time = end - start

            start = time.time()
            if event.type == 'Job':
                gen_job_finish = self.generator.gen_job_finish(event)
                self.event_queue.merge(event.tasks)
                self.event_queue.merge([gen_job_finish])
                if gen_job_finish.is_last:
                    gen_request_finish = self.generator.gen_request_finish(gen_job_finish)
                    self.event_queue.merge([gen_request_finish])
            end = time.time()
            block_3_time = end - start

            start = time.time()
            if event.type == 'Task':
                event.core.tasks.popleft()
            end = time.time()
            block_4_time = end - start

            start = time.time()
            if event.type == 'JobFinish':
                if not event.is_last:
                    gen_job = self.generator.gen_job(accelerators[acc_ind], event.job_id+1, event.req_id, event.dnn, event.layer_num+1, event.t_finish, event.t_finish)
                    self.event_queue.merge([gen_job])
            end = time.time()
            block_5_time = end - start

            start = time.time()
            if event.type == 'RequestFinish':
                requests_in_sim -= 1
                print('requests in sim: ', requests_in_sim)
                completed_reqs[event.req_id] = generated_reqs[event.req_id]
                completed_reqs[event.req_id].t_finish = event.t_finish 
                completed_req = completed_reqs[event.req_id]
                utilized_cores = 0
                if opt.core_util:
                    for accelerator in accelerators:
                        for core in accelerator.cores:
                            if core.tasks:
                                utilized_cores += 1
                else:
                    utilized_cores, core_utilization = None, None

                req_index = self.log_request('completed', req_index, completed_req, utilized_cores, num_cores)
            end = time.time()
            block_6_time = end - start

            if block_1_time > 1:
                print(f'block 1 stall, took {block_1_time} s')
            if block_2_time > 1:
                print(f'block 2 stall, took {block_2_time} s')
            if block_3_time > 1:
                print(f'block 3 stall, took {block_3_time} s')
            if block_4_time > 1:
                print(f'block 4 stall, took {block_4_time} s')
            if block_5_time > 1:
                print(f'block 5 stall, took {block_5_time} s')
            if block_6_time > 1:
                print(f'block 6 stall, took {block_6_time} s')

            if not self.event_queue:
                continue_sim = False
        
        return completed_reqs.values(), generated_reqs.values()

def print_sim_metrics(reqs):
    '''
    print relevant sim metrics to terminal
    '''
    times = [req.t_finish - req.t_arrival for req in reqs]
    req_dict = defaultdict(int)
    for req in reqs:
        req_dict[req.dnn] += 1
    print('model types served:', dict(req_dict))
    for time in times:
        if time < 0:
            print('negative time of completion computed. Debug')
    print('num_reqs completed:', len(times))

def run_sim(opt):
    start = time.time()
    sim = Simulator()
    #T = SIMULATION_T
    # T = 10000*(4800*(10**3))/opt.traffic  # imagenet data is 4800Bytes, so this translates to the total amount of time in nanoseconds to transmit a image through a 100 Gbps ethernet
    # T = 1184 + 344
    T = opt.time
    #T = 100000000000
    #sim_models = {'AlexNet':None,'ResNet18':None,'VGG16':None,'VGG19':None,'BERT':None,'GPT2':None,'ChatGPT':None,'DLRM':None}
    sim_models = {'LeNet':None}
    sim_alg = round_robin
    sim_poisson = True
    load_factor = 0
    gpu_datapath_latency = 1549000  # in nanoseconds
    brainwave_datapath_latency = 344  # in nanoseconds
    lightning_datapath_latency = 344  # in nanoseconds
    # sim_interarrival = T/opt.inter 
    sim_interarrival = opt.inter
    processor = processors[opt.proc_id]
    
    if 'Lightning' in processor[0]:
        sim_dp_latency = lightning_datapath_latency
        print("Adding datapath latency {}ns to {}".format(sim_dp_latency, processor[0]))
    elif 'Brainwave' in processor[0]:
        sim_dp_latency = brainwave_datapath_latency
        print("Adding datapath latency {}ns to {}".format(sim_dp_latency, processor[0]))
    elif 'DPU' in processor[0]:
        sim_dp_latency = brainwave_datapath_latency
        print("Adding datapath latency {}ns to {}".format(sim_dp_latency, processor[0]))
    else:
        sim_dp_latency = gpu_datapath_latency
        print("Adding datapath latency {}ns to {}".format(sim_dp_latency, processor[0]))
    
    if not sim.event_queue:
        completed_reqs, generated_reqs = sim.simulate(models=sim_models, num_acc=1, num_cores=processor[1], alg=sim_alg, interarrival=sim_interarrival, dp_latency=sim_dp_latency, poisson=sim_poisson, end_time=opt.time*T)
    else:
        print("ERROR: Uncompleted requests in previous sim. Debug")
    comp_avg_completion_time = sum([((req.t_finish - req.t_arrival)/processor[2]) for req in completed_reqs])/len(completed_reqs) 
    end = time.time()
    sim_time = end - start
    print('avg completion time:', comp_avg_completion_time)    
    print('sim_time: ', sim_time, ' s')
    print_sim_metrics(completed_reqs)
    return comp_avg_completion_time, sim_time 

if __name__ == "__main__":
    random.seed(1)
    # comp_avg_completion_time, sim_time = run_sim(opt)
    # pickle.dump(comp_avg_completion_time, open("sim_data/proc_{}_traffic_{}.p".format(opt.proc_id, opt.traffic), 'wb'))
    # pickle.dump(sim_time, open("sim_times/proc_{}_traffic_{}_sim_times.p".format(opt.proc_id,  opt.traffic), 'wb'))
    opt.time = 1184 + 344
    for interarrival in range(500,2000,100):
        opt.inter = interarrival
        run_sim(opt) # hanging return values currently (FIX THIS)
        if len(job_times) > 0: # shouldn't be the case
            job_avg_completion_time = sum(job_times)/len(job_times)
            pickle.dump(job_avg_completion_time, open("sim_data/{}_interarrival_avg_job_completion.p".format(interarrival), 'wb'))
            job_times = []