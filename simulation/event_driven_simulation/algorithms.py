import argparse
import matplotlib.pyplot as plt
import pickle
from termcolor import colored
from tetrismatrix import *
from nextrun import *
import time
import random
import math
import seaborn as sns
from tqdm import tqdm
import random
import binpacking
import pickle
import numpy as np
import copy
import os
import os.path


label = {
    0: "PerFlowLongFlowFirst", 
    1: "PerFlowShortFlowFirst", 
    2: "PerTaskLongFlowFirst", 
    3: "PerTaskShortFlowFirst", 
    4: "PerTaskRandomFlowFirst", 
    5: "PerTaskPerCoreRectangleShortFlowFirst",
    6: "PerTaskPerCoreRectangleStatefulShortFlowFirst",
    7: "PerTaskAllCoreRectangleStatefulShortFlowFirst",
    8: "PerTaskAllCoreStatefulRectangleFlowFirst",
    9: "PerTaskAllCoreStatefulRectangleFlowFirst2",
    10: "PerTaskAllCoreStatefulRectangleFlowPreserveStateFirst",
    11: "RoundRobinFIFO",
    12: "RoundRobinSJF",
    13: "LookupTableSJFMax",
    14: "LookupTableSJFSum",
    15: "LookupTableFIFO",
    16: "LookupTableShiftedFIFO",
    17: "LookupTablePermutedFIFO",
    18: "LongestFlowFirstFIFO",
    }


## data structure:
## [cores] is a map, core index -> set of global flow index
## [flows] is a map, global flow index -> flow size
## [tasks] is a map, task index -> set of global flow index


## visualize scheduling via stacked bars
def VisualizeScheduling(dir, tasks, flows, cores, title, savefig=False):
    print("Visulizing {} scheduling outcomes...".format(title))
    palette = sns.color_palette(None, len(tasks))
    tasks_completion_time = {}
    fig, ax = plt.subplots(figsize=(20,len(cores)//3), facecolor="white")
    ax.set_facecolor("white")
    task_on_core = {}
        
    for t in tqdm(tasks, desc="Visualizing tasks..."):
        task_on_core[t] = {}
        max_task_completion_time = 0
        for c in cores:
            task_on_core[t][c] = [(0, 0) for x in range(len(cores[c]))]
            for f in tasks[t]:
                if f in cores[c]:  # task is on this core
                    task_already_on_core = 0
                    for i in range(0, cores[c].index(f)):
                        task_already_on_core += flows[cores[c][i]]
                    task_tuple = (task_already_on_core, flows[f])
                    task_on_core[t][c][cores[c].index(f)] = task_tuple

                    if max_task_completion_time < task_already_on_core + flows[f]:
                        max_task_completion_time = task_already_on_core + flows[f]

            ax.broken_barh(task_on_core[t][c], (c+0.6, 0.8), facecolor=palette[t], label="Task "+str(t)) 
            for h in range(len(task_on_core[t][c])):
                ax.text(x=task_on_core[t][c][h][0]+task_on_core[t][c][h][1]/2, y=c+1, s=task_on_core[t][c][h][1], ha='center', va='center', color='white')        
            
        tasks_completion_time[t] = max_task_completion_time


    ax.set_ylabel('Optical MAC cores')
    ax.set_xlabel('Time slots')
    plt.title(title)

    if savefig:
        figname = "{}.png".format(title)
        plt.savefig(os.path.join(dir, figname), dpi=500)
    else:
        plt.show()

    return tasks_completion_time


def VisualizeScheduleingMatrix(dir, tasks, flows, cores, colormap, title, savefig=False):
    # convert task/flow/core language into a matrix
    tail_tasks_completion_time = 0
    tasks_completion_time = {}
    task_on_core = {}
    for t in tqdm(tasks, desc="Visualizing tasks..."):
        task_on_core[t] = {}
        max_task_completion_time = 0
        for c in cores:
            task_on_core[t][c] = [(0, 0) for x in range(len(cores[c]))]
            for f in tasks[t]:
                if f in cores[c]:  # task is on this core
                    task_already_on_core = 0
                    for i in range(0, cores[c].index(f)):
                        task_already_on_core += flows[cores[c][i]]
                    task_tuple = (task_already_on_core, flows[f])
                    task_on_core[t][c][cores[c].index(f)] = task_tuple

                    if max_task_completion_time < task_already_on_core + flows[f]:
                        max_task_completion_time = task_already_on_core + flows[f]
        
        tasks_completion_time[t] = max_task_completion_time

    # if we want to visualize scheduling or not
    ###################################
    #     if tail_tasks_completion_time < max_task_completion_time:
    #         tail_tasks_completion_time = max_task_completion_time

    # visulize_matrix = np.zeros((len(cores), tail_tasks_completion_time))
    
    # for t in tasks:
    #     for c in cores:
    #         for i in range(len(task_on_core[t][c])):
    #             visulize_matrix[c, task_on_core[t][c][i][0] : task_on_core[t][c][i][0]+task_on_core[t][c][i][1]] = [t+3 for i in range(task_on_core[t][c][i][1])]

    # plt.figure()
    # plt.imshow(visulize_matrix, origin="lower", cmap=colormap)
    # plt.title(title)

    # plt.colorbar()
    # if savefig:
    #     figname = "{}.png".format(title)
    #     plt.savefig(os.path.join(dir, figname), dpi=500)
    # else:
    #     plt.show()

    return tasks_completion_time


def PerFlowLongFlowFirst(flows, core_num):
    print("Running PerFlowLongFlowFirst")
    cores = {}
    for c in range(core_num):
        cores[c] = []
    core_weights = [0 for i in range(core_num)]

    # sort at flow level based: small flow first
    sorted_flows = {k: v for k, v in sorted(flows.items(), key=lambda item: item[1], reverse=True)}

    # assign tasks on cores based on core weights
    for f in tqdm(sorted_flows, desc="Scheduling tasks' flows..."):
        min_core_weight = min(core_weights)
        min_core_weight_index = core_weights.index(min_core_weight)
        cores[min_core_weight_index].append(f)  # assign the longest flow to the least weight core
        core_weights[min_core_weight_index] += sorted_flows[f]

    return cores


def PerFlowShortFlowFirst(flows, core_num):  ## first means we want to choose the least weight core first
    print("Running PerFlowShortFlowFirst")
    cores = {}
    for c in range(core_num):
        cores[c] = []
    core_weights = [0 for i in range(core_num)]
    
    # sort at flow level based: large flow first
    sorted_flows = {k: v for k, v in sorted(flows.items(), key=lambda item: item[1], reverse=False)}

    # assign tasks on cores based on core weights
    for f in tqdm(sorted_flows, desc="Scheduling tasks' flows..."):
        min_core_weight = min(core_weights)
        min_core_weight_index = core_weights.index(min_core_weight)
        cores[min_core_weight_index].append(f)  # assign the longest flow to the least weight core
        core_weights[min_core_weight_index] += sorted_flows[f]

    return cores


def PerTaskLongFlowFirst(tasks, flows, core_num):
    print("Running PerTaskLongFlowFirst")
    cores = {}
    for c in range(core_num):
        cores[c] = []
    core_weights = [0 for i in range(core_num)]

    # get the task's longest flow size
    task_flowsize = {}
    for t in tasks:
        task_flowsize[t] = []
        for f in tasks[t]:
            task_flowsize[t].append(flows[f])
    task_longest_flows = {}
    for t in task_flowsize:
        task_longest_flows[t] = max(task_flowsize[t])

    # sort at task level based on tasks's longest flow: small task first
    sorted_tasks = {k: v for k, v in sorted(task_longest_flows.items(), key=lambda item: item[1], reverse=True)}
    
    # assign tasks on cores based on core weights
    for t in tqdm(sorted_tasks, desc="Scheduling tasks..."):
        for f in tasks[t]:
            min_core_weight = min(core_weights)
            min_core_weight_index = core_weights.index(min_core_weight)
            cores[min_core_weight_index].append(f)  # assign the longest flow to the least weight core
            core_weights[min_core_weight_index] += flows[f]
    
    return cores


def PerTaskShortFlowFirst(tasks, flows, core_num):
    print("Running PerTaskShortFlowFirst")
    cores = {}
    for c in range(core_num):
        cores[c] = []
    core_weights = [0 for i in range(core_num)]

    # get the task's longest flow size
    task_flowsize = {}
    for t in tasks:
        task_flowsize[t] = []
        for f in tasks[t]:
            task_flowsize[t].append(flows[f])
    task_longest_flows = {}
    for t in task_flowsize:
        task_longest_flows[t] = max(task_flowsize[t])

    # sort at task level based on tasks's longest flow: large task first
    sorted_tasks = {k: v for k, v in sorted(task_longest_flows.items(), key=lambda item: item[1], reverse=False)}
    
    # assign tasks on cores based on core weights
    for t in tqdm(sorted_tasks, desc="Scheduling tasks..."):
        flowsizes = {f: flows[f] for f in tasks[t]}
        sorted_flows = {k: v for k, v in sorted(flowsizes.items(), key=lambda item: item[1], reverse=True)}
        for f in sorted_flows:
        # for f in tasks[t]:
            min_core_weight = min(core_weights)
            min_core_weight_index = core_weights.index(min_core_weight)
            cores[min_core_weight_index].append(f)  # assign the longest flow to the least weight core
            core_weights[min_core_weight_index] += flows[f]

    return cores


def PerTaskRandomFlowFirst(tasks, flows, core_num):
    print("Running PerTaskRandomFlowFirst")
    cores = {}
    for c in range(core_num):
        cores[c] = []
    core_weights = [0 for i in range(core_num)]

    # get the task's longest flow size
    task_flowsize = {}
    for t in tasks:
        task_flowsize[t] = []
        for f in tasks[t]:
            task_flowsize[t].append(flows[f])
    task_longest_flows = {}
    for t in task_flowsize:
        task_longest_flows[t] = max(task_flowsize[t])
    
    ## randomize task order
    random.seed(10)
    random.shuffle(list(task_longest_flows.keys()))
    sorted_tasks = {k: task_longest_flows[k] for k in task_longest_flows}

    # assign tasks on cores based on core weights
    for t in tqdm(sorted_tasks, desc="Scheduling tasks..."):
        for f in tasks[t]:
            min_core_weight = min(core_weights)
            min_core_weight_index = core_weights.index(min_core_weight)
            cores[min_core_weight_index].append(f)  # assign the longest flow to the least weight core
            core_weights[min_core_weight_index] += flows[f]
    
    return cores


## first transform each task tetric block into rectangle in a per-core manner preserving the task's completion time, then run schdeuling based on shortest task first
def PerTaskPerCoreRectangleShortFlowFirst(tasks, flows, core_num, wave_num):
    print("Running PerTask PerCore Rectangle Short Flow First Scheduling")
    cores = {}
    output_tasks = {}
    output_flows = {}
    for c in range(core_num):
        cores[c] = []
    core_weights = [0 for i in range(core_num)]

    # get the task's longest flow size
    task_flowsize = {}
    for t in tasks:
        task_flowsize[t] = []
        for f in tasks[t]:
            task_flowsize[t].append(flows[f])
    task_longest_flows = {}
    for t in task_flowsize:
        task_longest_flows[t] = max(task_flowsize[t])

    # bandwidth (core count) reallocation towards rectangle for each flows within a coflow, not arbitrarily splittable
    global_flow_index = 0
    for t in tqdm(tasks, desc="Scheduling tasks..."): 
        ## first calculate if each flow of this task can be extended to reduce utilized wavelengths (equavalently bandwidth)
        utilized_wave_relative_core = {}
        for f in tasks[t]:
            if task_longest_flows[t] // flows[f] > 1:  # we can extend the time needed and cut the number of required wavelengths 
                utilized_wave_relative_core[f] = math.ceil(wave_num / (task_longest_flows[t] // flows[f]))  # the core's wavelengths are partially utilized
            else:
                utilized_wave_relative_core[f] = wave_num  # the core's wavelengths are fully utilized
        # re-grouping flows into cores
        regroup_task = binpacking.to_constant_volume(utilized_wave_relative_core, wave_num)
        output_tasks[t] = []
        for g_task in regroup_task:
            gf_size = 0
            for gf in g_task:
                gf_size += flows[gf]
            output_flows[global_flow_index] = gf_size
            output_tasks[t].append(global_flow_index)
            global_flow_index += 1
    
    # scheduling based on regrouped tasks
    output_task_flowsize = {}
    for t in tasks:
        output_task_flowsize[t] = []
        for f in tasks[t]:
            output_task_flowsize[t].append(flows[f])
    output_task_longest_flows = {}
    for t in output_task_flowsize:
        output_task_longest_flows[t] = max(output_task_flowsize[t])
    
    # sort at task level based on tasks's longest flow: small task first
    sorted_output_tasks = {k: v for k, v in sorted(output_task_longest_flows.items(), key=lambda item: item[1], reverse=False)} 

    # assign tasks on cores based on core weights
    for t in tqdm(sorted_output_tasks, desc="Scheduling tasks..."):
        for f in output_tasks[t]:
            min_core_weight = min(core_weights)
            min_core_weight_index = core_weights.index(min_core_weight)
            cores[min_core_weight_index].append(f)  # assign the longest flow to the least weight core
            core_weights[min_core_weight_index] += output_flows[f]
    
    return output_tasks, output_flows, cores


## first transform each task tetric block into rectangle in a per-core manner preserving the task's completion time, then run schdeuling based on STATEFUL shortest task first
def PerTaskPerCoreRectangleStatefulShortFlowFirst(tasks, flows, core_num, wave_num):
    print("Running PerTask PerCore Rectangle Stateful Short Flow First Rectangle Scheduling")
    output_tasks = {}
    output_flows = {}

    # get the task's longest flow size
    task_flowsize = {}
    for t in tasks:
        task_flowsize[t] = []
        for f in tasks[t]:
            task_flowsize[t].append(flows[f])
    task_longest_flows = {}
    for t in task_flowsize:
        task_longest_flows[t] = max(task_flowsize[t])

    # bandwidth (core count) reallocation towards rectangle for each flows within a coflow, not arbitrarily splittable
    global_flow_index = 0
    for t in tqdm(tasks, desc="Scheduling tasks..."): 
        ## first calculate if each flow of this task can be extended to reduce utilized wavelengths (equavalently bandwidth)
        utilized_wave_relative_core = {}
        for f in tasks[t]:
            if task_longest_flows[t] // flows[f] > 1:  # we can extend the time needed and cut the number of required wavelengths 
                utilized_wave_relative_core[f] = math.ceil(wave_num / (task_longest_flows[t] // flows[f]))  # the core's wavelengths are partially utilized
            else:
                utilized_wave_relative_core[f] = wave_num  # the core's wavelengths are fully utilized
        # re-grouping flows into cores
        regroup_task = binpacking.to_constant_volume(utilized_wave_relative_core, wave_num)
        output_tasks[t] = []
        for g_task in regroup_task:
            gf_size = 0
            for gf in g_task:
                gf_size += flows[gf]
            output_flows[global_flow_index] = gf_size
            output_tasks[t].append(global_flow_index)
            global_flow_index += 1
    
    # scheduling based on regrouped tasks
    output_task_flowsize = {}
    for t in tasks:
        output_task_flowsize[t] = []
        for f in tasks[t]:
            output_task_flowsize[t].append(flows[f])
    output_task_longest_flows = {}
    for t in output_task_flowsize:
        output_task_longest_flows[t] = max(output_task_flowsize[t])

    # scheduling based on regrouped tasks (rectangles)， sorting based on dynamic "water levels", sort first on tasks, small task first
    sorted_output_tasks = {k: v for k, v in sorted(output_task_longest_flows.items(), key=lambda item: item[1], reverse=False)} 
    cores = {}
    for c in range(core_num):
        cores[c] = []
    core_weights = [0 for i in range(core_num)] 

    while(len(sorted_output_tasks) > 0):
        t = list(sorted_output_tasks.items())[0][0]  # the current first task
        # place the first task
        for f in output_tasks[t]:
            min_core_weight = min(core_weights)
            min_core_weight_index = core_weights.index(min_core_weight)
            cores[min_core_weight_index].append(f)  # assign the longest flow to the least weight core
            core_weights[min_core_weight_index] += output_flows[f]
        # remove the first task from the task map
        sorted_output_tasks.pop(t)

        # calculate virtual task water levels based on current state
        stateful_output_task_longest_flows = {}
        for tt in sorted_output_tasks:
            virtual_core_weights = [core_weights[i] for i in range(core_num)]
            task_flow_stateful_completion_time = []
            for f in output_tasks[tt]:
                min_core_weight = min(virtual_core_weights)
                min_core_weight_index = virtual_core_weights.index(min_core_weight)
                virtual_core_weights[min_core_weight_index] += output_flows[f]
                task_flow_stateful_completion_time.append(virtual_core_weights[min_core_weight_index])
            
            stateful_output_task_longest_flows[tt] = max(task_flow_stateful_completion_time)
        
        sorted_stateful_output_tasks = {k: v for k, v in sorted(stateful_output_task_longest_flows.items(), key=lambda item: item[1], reverse=False)} 
        sorted_output_tasks = sorted_stateful_output_tasks
       
    return output_tasks, output_flows, cores


## first transform each task tetric block into rectangle in a GLOBAL-CORE manner preserving the task's completion time, then run schdeuling based on STATEFUL shortest task first
def PerTaskAllCoreRectangleStatefulShortFlowFirst(tasks, flows, core_num, wave_num):
    print("Running PerTask AllCore Rectangle Stateful Short Flow First Scheduling")
    output_tasks = {}
    output_flows = {}

    # get the task's longest flow size
    task_flowsize = {}
    for t in tasks:
        task_flowsize[t] = []
        for f in tasks[t]:
            task_flowsize[t].append(flows[f])

    # make each task (tetris block) to be rectangle
    global_flow_index = 0
    for t in tqdm(tasks, desc="Scheduling tasks..."): 
        # run bin packing with bin size to be the largest flow of the task
        regroup_task = binpacking.to_constant_volume(task_flowsize[t], max(task_flowsize[t]))
        output_tasks[t] = []
        regroup_flowsize = []
        for g_task in regroup_task:
            regroup_flowsize.append(sum(g_task))
        regroup_flowsize.sort(reverse=True)
        for regroup_f in regroup_flowsize :
            output_flows[global_flow_index] = regroup_f
            output_tasks[t].append(global_flow_index)
            global_flow_index += 1
    
    output_task_flowsize = {}
    for t in tasks:
        output_task_flowsize[t] = []
        for f in tasks[t]:
            output_task_flowsize[t].append(flows[f])
    output_task_longest_flows = {}
    for t in output_task_flowsize:
        output_task_longest_flows[t] = max(output_task_flowsize[t])
    
    # scheduling based on regrouped tasks (rectangles)， sorting based on dynamic "water levels", sort first on tasks, small task first
    sorted_output_tasks = {k: v for k, v in sorted(output_task_longest_flows.items(), key=lambda item: item[1], reverse=False)} 
    cores = {}
    for c in range(core_num):
        cores[c] = []
    core_weights = [0 for i in range(core_num)] 

    while(len(sorted_output_tasks) > 0):
        t = list(sorted_output_tasks.items())[0][0]  # the current first task
        # place the first task
        for f in output_tasks[t]:
            min_core_weight = min(core_weights)
            min_core_weight_index = core_weights.index(min_core_weight)
            cores[min_core_weight_index].append(f)  # assign the longest flow to the least weight core
            core_weights[min_core_weight_index] += output_flows[f]
        # remove the first task from the task map
        sorted_output_tasks.pop(t)

        # calculate virtual task water levels based on current state
        stateful_output_task_longest_flows = {}
        for tt in sorted_output_tasks:
            virtual_core_weights = [core_weights[i] for i in range(core_num)]
            task_flow_stateful_completion_time = []
            for f in output_tasks[tt]:
                min_core_weight = min(virtual_core_weights)
                min_core_weight_index = virtual_core_weights.index(min_core_weight)
                virtual_core_weights[min_core_weight_index] += output_flows[f]
                task_flow_stateful_completion_time.append(virtual_core_weights[min_core_weight_index])
            
            stateful_output_task_longest_flows[tt] = max(task_flow_stateful_completion_time)
        
        sorted_output_tasks = {k: v for k, v in sorted(stateful_output_task_longest_flows.items(), key=lambda item: item[1], reverse=False)} 
       
    return output_tasks, output_flows, cores


## run scheduling based on STATEFUL shortest task first while creating rectangle water level
def PerTaskAllCoreStatefulRectangleFlowFirst(tasks, flows, core_num, wave_num):
    print("Running PerTask AllCore Stateful Rectangle Flow First Scheduling")
    # get the task's longest flow size
    task_flowsize = {}
    for t in tasks:
        task_flowsize[t] = []
        for f in tasks[t]:
            task_flowsize[t].append(flows[f])
    task_longest_flows = {}
    for t in task_flowsize:
        task_longest_flows[t] = max(task_flowsize[t])

    # sort at task level based on tasks's longest flow: large task first
    sorted_tasks = {k: v for k, v in sorted(task_longest_flows.items(), key=lambda item: item[1], reverse=False)}
    
    cores = {}
    cores_cache = {}
    for c in range(core_num):
        cores_cache[c] = []
    core_weights = [0 for i in range(core_num)]

    while(len(sorted_tasks) > 0):
        t = list(sorted_tasks.keys())[0]  # the current first task
        # place the first task
        core_weights_dict = {-c-1 : core_weights[c] for c in range(core_num) if core_weights[c] > 0} # distinguish existing core load with new task's flows
        task_flowsize_dict = {f : flows[f] for f in tasks[t]}
        flows_topack_dict = {**core_weights_dict, **task_flowsize_dict}
        regroup_task_dictlist = binpacking.to_constant_bin_number(flows_topack_dict, core_num) 
        for c in range(core_num):
            core_weights[c] = sum(regroup_task_dictlist[c].values())
            cores[c] = []
            keys = list(regroup_task_dictlist[c].keys())
            keys.sort()
            for f in keys:
                if f > -1:  # this load is new task
                    cores[c].append(f)
                else:  # this load is existing core load
                    cores[c] += cores_cache[-f-1]
        # remove the first task from the task map
        for c in range(core_num):
            cores_cache[c] = cores[c].copy()

        sorted_tasks.pop(t)

        # calculate virtual task water levels based on current state
        stateful_task_longest_flows = {}
        for tt in tqdm(sorted_tasks, desc="Scheduling tasks..."):
            virtual_core_weights = [core_weights[i] for i in range(core_num)]
            task_flow_stateful_completion_time = []
            
            virtual_core_weights_dict = {-c-1 : virtual_core_weights[c] for c in range(core_num)} # distinguish existing core load with new task's flows
            virtual_task_flowsize_dict = {f : flows[f] for f in tasks[tt]}
            virtual_flows_topack_dict = {**virtual_core_weights_dict, **virtual_task_flowsize_dict}
            virtual_regroup_task_dictlist = binpacking.to_constant_bin_number(virtual_flows_topack_dict, core_num) 
            for c in range(core_num):
                task_flow_stateful_completion_time.append(sum(virtual_regroup_task_dictlist[c].values()))
            
            stateful_task_longest_flows[tt] = max(task_flow_stateful_completion_time)
        
        sorted_tasks = {k: v for k, v in sorted(stateful_task_longest_flows.items(), key=lambda item: item[1], reverse=False)} 
       
    return cores


## run scheduling based on STATEFUL shortest task first while creating rectangle water level
def PerTaskAllCoreStatefulRectangleFlowFirst2(tasks, flows, core_num, wave_num):
    print("Running PerTask AllCore Stateful Rectangle Flow First Scheduling")
    # get the task's longest flow size
    task_flowsize = {}
    for t in tasks:
        task_flowsize[t] = []
        for f in tasks[t]:
            task_flowsize[t].append(flows[f])
    task_longest_flows = {}
    for t in task_flowsize:
        task_longest_flows[t] = max(task_flowsize[t])

    # sort at task level based on tasks's longest flow: large task first
    sorted_tasks = {k: v for k, v in sorted(task_longest_flows.items(), key=lambda item: item[1], reverse=False)}
    
    cores = {}
    cores_cache = {}
    for c in range(core_num):
        cores_cache[c] = []
    core_weights = [0 for i in range(core_num)]

    while(len(sorted_tasks) > 0):
        t = list(sorted_tasks.keys())[0]  # the current first task
        # place the first task
        core_weights_dict = {-c-1 : core_weights[c] + 100000000000 for c in range(core_num) if core_weights[c] > 0}  # make existing flow size large so that it does not get disturbed
        task_flowsize_dict = {f : flows[f] for f in tasks[t]}
        flows_topack_dict = {**core_weights_dict, **task_flowsize_dict}
        regroup_task_dictlist = binpacking.to_constant_bin_number(flows_topack_dict, core_num) 
        for subdict in regroup_task_dictlist:
            for i in subdict:
                if subdict[i] > 100000000000:
                    subdict[i] -= 100000000000

        for c in range(core_num):
            core_weights[c] = sum(regroup_task_dictlist[c].values())
            cores[c] = []
            keys = list(regroup_task_dictlist[c].keys())
            keys.sort()
            for f in keys:
                if f > -1:  # this load is new task
                    cores[c].append(f)
                else:  # this load is existing core load
                    cores[c] += cores_cache[-f-1]
        # remove the first task from the task map
        for c in range(core_num):
            cores_cache[c] = cores[c].copy()

        sorted_tasks.pop(t)

        # calculate virtual task water levels based on current state
        stateful_task_longest_flows = {}
        for tt in tqdm(sorted_tasks, desc="Scheduling tasks..."):
            virtual_core_weights = [core_weights[i] for i in range(core_num)]
            task_flow_stateful_completion_time = []
            
            virtual_core_weights_dict = {-c-1 : virtual_core_weights[c] + 100000000000 for c in range(core_num)} # distinguish existing core load with new task's flows
            virtual_task_flowsize_dict = {f : flows[f] for f in tasks[tt]}
            virtual_flows_topack_dict = {**virtual_core_weights_dict, **virtual_task_flowsize_dict}
            virtual_regroup_task_dictlist = binpacking.to_constant_bin_number(virtual_flows_topack_dict, core_num) 
            for c in range(core_num):
                task_flow_stateful_completion_time.append(sum(virtual_regroup_task_dictlist[c].values()))
            
            stateful_task_longest_flows[tt] = max(task_flow_stateful_completion_time) - 100000000000
        
        sorted_tasks = {k: v for k, v in sorted(stateful_task_longest_flows.items(), key=lambda item: item[1], reverse=False)} 
       
    return cores


def FlowCoreMapping(func_cores, flows, core_weights, current_task):
    # current_task is a list of flow index, order is from the longest flow to shortest flow
    flow_core = {}
    for f in current_task:
        min_core_weight = min(core_weights)
        min_core_weight_index = core_weights.index(min_core_weight)
        func_cores[min_core_weight_index].append(f)  # assign the longest flow to the least weight core, assume that task is already sorted based on longest flow
        core_weights[min_core_weight_index] += flows[f]
        flow_core[f] = min_core_weight_index

    max_completion_time = max(core_weights)  # note here that if there is a case where new task can fit into the valley of the previous task, then this is not the completion time of the new task, but rarely happen
    current_task_flowsize = {}
    for f in current_task:
        current_task_flowsize[f] = flows[f]

    sorted_current_task_flowsize = {k: v for k, v in sorted(current_task_flowsize.items(), key=lambda item: item[1], reverse=False)}  # small flows first
    reverse_current_task = list(sorted_current_task_flowsize.keys())  # shortest flow comes first

    ## form a rectngle
    for fr in reverse_current_task:  # in a reverse way
        core_weight_dict = {c: core_weights[c] for c in range(len(core_weights))}
        sorted_core_weight_dict = {k: v for k, v in sorted(core_weight_dict.items(), key=lambda item: item[1], reverse=True)}  # large core weight first
        sorted_core_weight_keys = list(sorted_core_weight_dict.keys())
        sorted_core_weight_values = list(sorted_core_weight_dict.values())
   
        find_a_spot = -1
        target_core_relative_index = 0
        while(find_a_spot == -1 and target_core_relative_index < len(func_cores)):
            if flows[fr] + sorted_core_weight_values[target_core_relative_index] > max_completion_time:  # this is not feasible, change to another
                target_core_relative_index += 1
            elif flows[fr] + sorted_core_weight_values[target_core_relative_index] > core_weights[flow_core[fr]]:
                find_a_spot = sorted_core_weight_keys[target_core_relative_index]
            else:
                target_core_relative_index += 1
        
        # move the flow to the new spot that can accommodate it
        if find_a_spot >= 0:
            core_weights[find_a_spot] += flows[fr]
            func_cores[find_a_spot].append(fr)

            core_weights[flow_core[fr]] -= flows[fr]
            func_cores[flow_core[fr]].remove(fr)


def PerTaskAllCoreStatefulRectangleFlowPreserveStateFirst(tasks, flows, core_num, wave_num):
    print("Running PerTask AllCore Stateful Rectangle Flow First Preserving Past State Scheduling")
    cores = {}
    for c in range(core_num):
        cores[c] = []
    core_weights = [0 for i in range(core_num)]

    # get the task's longest flow size
    task_flowsize = {}
    for t in tasks:
        task_flowsize[t] = []
        for f in tasks[t]:
            task_flowsize[t].append(flows[f])
    task_longest_flows = {}
    for t in task_flowsize:
        task_longest_flows[t] = max(task_flowsize[t])

    # sort at task level based on tasks's longest flow: large task first
    sorted_tasks = {k: v for k, v in sorted(task_longest_flows.items(), key=lambda item: item[1], reverse=False)}

    ## stateful create rectangle and run scheduling
    while(len(sorted_tasks) > 0):
        t = list(sorted_tasks.items())[0][0]  # the current first task
        # place the first task
        FlowCoreMapping(cores, flows, core_weights, tasks[t])
        # remove the first task from the task map
        sorted_tasks.pop(t)

        # re-sort based on new waterlevel, calculate virtual task water levels based on current state
        stateful_task_longest_flows = {}
        for tt in tqdm(sorted_tasks, desc="Scheduling tasks..."):
            # virtual_core_weights = [core_weights[i] for i in range(core_num)]
            # virtual_cores = {i: j for i, j in cores.items()}
            virtual_core_weights = copy.deepcopy(core_weights)
            virtual_cores = copy.deepcopy(cores)
            FlowCoreMapping(virtual_cores, flows, virtual_core_weights, tasks[tt])
            stateful_task_longest_flows[tt] = max(virtual_core_weights)
        
        sorted_tasks = {k: v for k, v in sorted(stateful_task_longest_flows.items(), key=lambda item: item[1], reverse=False)} 

    return cores    


def RoundRobinFIFO(tasks, flows, core_num):
    print("Running round robin")
    
    cores = {}
    for c in range(core_num):
        cores[c] = []
    
    i = 0
    for t in tasks:
        for f in tasks[t]:
            cores[i % core_num].append(f)
            i = i + 1

    return cores


def RoundRobinSJF(tasks, flows, core_num):
    print("Running round robin SJB")
    task_flowsize = {}
    for t in tasks:
        task_flowsize[t] = []
        for f in tasks[t]:
            task_flowsize[t].append(flows[f])
    task_longest_flows = {}
    for t in task_flowsize:
        task_longest_flows[t] = max(task_flowsize[t])  # max of all flows for sorting

    # sort at task level based on tasks's longest flow: large task first
    sorted_tasks = {k: v for k, v in sorted(task_longest_flows.items(), key=lambda item: item[1], reverse=False)}
    
    cores = {}
    for c in range(core_num):
        cores[c] = []
    
    i = 0
    for t in sorted_tasks:
        for f in tasks[t]:
            cores[i % core_num].append(f)
            i = i + 1

    return cores


def CreateLookupTable(task_single, task_name_single, flows, core_num):
    print("creating lookup tables for {}".format(task_name_single))
    ## longest flow first to least load core, relative flow index
    cores = {}
    for c in range(core_num):
        cores[c] = []
    core_weights = [0 for i in range(core_num)]
    # assume a single task
    current_task_flow = list(task_single.values())[0]   # list of current flows of global index
    flow_index_size = {x: flows[current_task_flow[x]] for x in range(len(current_task_flow))}   # map relative flow index to flow size
    sorted_flow_index_size = {k: v for k, v in sorted(flow_index_size.items(), key=lambda item: item[1], reverse=True)}   # sort, long flow first

    for fi in sorted_flow_index_size:   # relative flow index
        min_core_weight = min(core_weights)
        min_core_weight_index = core_weights.index(min_core_weight)
        cores[min_core_weight_index].append(fi)  # assign the longest flow to the least weight core, relative index
        core_weights[min_core_weight_index] += flows[current_task_flow[fi]]

    # here is relative flow index
    pickle.dump(cores, open("lookuptable/{}_core_{}.p".format(task_name_single, core_num), "wb"))

    return cores


def LookupTableSJFMax(tasks, task_names, flows, core_num):
    print("Running Look up tables")
    cores_table_set = []
    # get the task's longest flow size
    task_flowsize = {}
    for t in tasks:
        task_flowsize[t] = []
        for f in tasks[t]:
            task_flowsize[t].append(flows[f])
    task_longest_flows = {}
    for t in task_flowsize:
        task_longest_flows[t] = max(task_flowsize[t])  # max of all flows for sorting

    # sort at task level based on tasks's longest flow: large task first
    sorted_tasks = {k: v for k, v in sorted(task_longest_flows.items(), key=lambda item: item[1], reverse=False)}
    
    # sorted_tasks: sorted task ID -> included flow IDs        
    for i in sorted_tasks:
        if os.path.exists("lookuptable/{}_core_{}.p".format(task_names[i], core_num)):
            cores_table = pickle.load(open("lookuptable/{}_core_{}.p".format(task_names[i], core_num), "rb"))
        else:
            cores_table = CreateLookupTable({list(tasks.keys())[i]: list(tasks.values())[i]}, task_names[i], flows, core_num)
        cores_table_set.append(cores_table)

    cores = {}
    for c in range(core_num):
        cores[c] = []
    
    for k in range(len(cores_table_set)):
        cores_table = cores_table_set[k]
        for c in cores_table:
            for fr in cores_table[c]:
                cores[c].append(tasks[k][fr])

    return cores


def LookupTableSJFSum(tasks, task_names, flows, core_num):
    print("Running Look up tablesall")
    cores_table_set = []
    # get the task's longest flow size
    task_flowsize = {}
    for t in tasks:
        task_flowsize[t] = []
        for f in tasks[t]:
            task_flowsize[t].append(flows[f])
    task_longest_flows = {}
    for t in task_flowsize:
        task_longest_flows[t] = sum(task_flowsize[t])  # sum of all flows for sorting

    # sort at task level based on tasks's longest flow: large task first
    sorted_tasks = {k: v for k, v in sorted(task_longest_flows.items(), key=lambda item: item[1], reverse=False)}
    
    # sorted_tasks: sorted task ID -> included flow IDs        
    for i in sorted_tasks:
        if os.path.exists("lookuptable/{}_core_{}.p".format(task_names[i], core_num)):
            cores_table = pickle.load(open("lookuptable/{}_core_{}.p".format(task_names[i], core_num), "rb"))
        else:
            cores_table = CreateLookupTable({list(tasks.keys())[i]: list(tasks.values())[i]}, task_names[i], flows, core_num)
        cores_table_set.append(cores_table)

    cores = {}
    for c in range(core_num):
        cores[c] = []
    
    for k in range(len(cores_table_set)):
        cores_table = cores_table_set[k]
        for c in cores_table:
            for fr in cores_table[c]:
                cores[c].append(tasks[k][fr])

    return cores


def GetLongestFlowCore(cores_table, flows, core_num):
    core_load = [0 for i in range(core_num)]
    for c in cores_table:
        for f in cores_table[c]:
            core_load[c] += flows[f]

    maxvalue = max(core_load)
    maxnindex = core_load.index(maxvalue)

    minvalue = max(core_load)
    minnindex = core_load.index(minvalue)

    return maxnindex, minnindex
    


def LookupTableEqualShiftedFIFO(tasks, task_names, flows, core_num):
    print("Running Look up tables FIFO")
    cores_table_set = []
    
    # sorted_tasks: sorted task ID -> included flow IDs        
    for i in tasks:
        if os.path.exists("lookuptable/{}_core_{}.p".format(task_names[i], core_num)):
            cores_table = pickle.load(open("lookuptable/{}_core_{}.p".format(task_names[i], core_num), "rb"))
        else:
            cores_table = CreateLookupTable({list(tasks.keys())[i]: list(tasks.values())[i]}, task_names[i], flows, core_num)
        cores_table_set.append(cores_table)

    cores = {}
    for c in range(core_num):
        cores[c] = []
    
    for k in range(len(cores_table_set)):
        cores_table = cores_table_set[k]
        for c in cores_table:
            for fr in cores_table[c]:
                cores[(c+k)%core_num].append(tasks[k][fr])
        
    return cores


def LookupTablePermutedFIFO(tasks, task_names, flows, core_num):
    print("Running Look up tables FIFO")
    cores_table_set = []
    
    # sorted_tasks: sorted task ID -> included flow IDs        
    for i in tasks:
        if os.path.exists("lookuptable/{}_core_{}.p".format(task_names[i], core_num)):
            cores_table = pickle.load(open("lookuptable/{}_core_{}.p".format(task_names[i], core_num), "rb"))
        else:
            cores_table = CreateLookupTable({list(tasks.keys())[i]: list(tasks.values())[i]}, task_names[i], flows, core_num)
        core_table_keys = list(cores_table.keys())
        permuted_core_table_keys = np.random.permutation(core_table_keys)
        permuted_cores_table = {k: cores_table[k] for k in permuted_core_table_keys}
        cores_table_set.append(permuted_cores_table)

    cores = {}
    for c in range(core_num):
        cores[c] = []
    
    for k in range(len(cores_table_set)):
        cores_table = cores_table_set[k]
        for c in cores_table:
            for fr in cores_table[c]:
                cores[c].append(tasks[k][fr])

    return cores


def LookupTableFIFO(tasks, task_names, flows, core_num):
    print("Running Look up tables FIFO")
    cores_table_set = []
    
    # sorted_tasks: sorted task ID -> included flow IDs        
    for i in tasks:
        if os.path.exists("lookuptable/{}_core_{}.p".format(task_names[i], core_num)):
            cores_table = pickle.load(open("lookuptable/{}_core_{}.p".format(task_names[i], core_num), "rb"))
        else:
            cores_table = CreateLookupTable({list(tasks.keys())[i]: list(tasks.values())[i]}, task_names[i], flows, core_num)
        cores_table_set.append(cores_table)

    cores = {}
    for c in range(core_num):
        cores[c] = []
    
    for k in range(len(cores_table_set)):
        cores_table = cores_table_set[k]
        for c in cores_table:
            for fr in cores_table[c]:
                cores[c].append(tasks[k][fr])

    return cores


def LongestFlowFirstFIFO(tasks, flows, core_num):
    cores = {}
    for c in range(core_num):
        cores[c] = []
    core_weights = [0 for i in range(core_num)] 

    # assign tasks on cores based on core weights
    for t in tqdm(tasks, desc="Scheduling tasks..."):
        flowsizes = {f: flows[f] for f in tasks[t]}
        sorted_flows = {k: v for k, v in sorted(flowsizes.items(), key=lambda item: item[1], reverse=True)}
        for f in sorted_flows:
            min_core_weight = min(core_weights)
            min_core_weight_index = core_weights.index(min_core_weight)
            cores[min_core_weight_index].append(f)  # assign the longest flow to the least weight core
            core_weights[min_core_weight_index] += flows[f]

    return cores


def LoadPickleModels(model_list):
    print(colored("- Load pickle models...", "blue"))
    all_compressed_tasks = []
    all_compressed_tasks_names = []
    for model in tqdm(model_list):
        if  model == "lenet":
            all_lenet_compressed_tasks = []
            lenet_lin_1 = pickle.load(open("../../data/saved_models/torch_models/lenet/compressed_fc_1.p", "rb"))
            lenet_lin_2 = pickle.load(open("../../data/saved_models/torch_models/lenet/compressed_fc_2.p", "rb"))
            lenet_lin_3 = pickle.load(open("../../data/saved_models/torch_models/lenet/compressed_fc_3.p", "rb"))
            all_lenet_compressed_tasks = [lenet_lin_1, lenet_lin_2, lenet_lin_3]
            all_compressed_tasks += all_lenet_compressed_tasks
            all_compressed_tasks_names += ["lenet_1", "lenet_2", "lenet_3"]

        if model == "bert":
            all_bert_compressed_tasks = []
            all_bert_compressed_tasks_names = []
            for i in range(12):  # bert has 12 fully connected layers
                locals()["compressed_aln_lin_"+str(i)+"_1"] = pickle.load(open("../../data/saved_models/torch_models/bert/compressed_80_aln_lin_"+str(i)+"_1.p", "rb"))
                locals()["compressed_aln_lin_"+str(i)+"_2"] = pickle.load(open("../../data/saved_models/torch_models/bert/compressed_80_aln_lin_"+str(i)+"_2.p", "rb"))
                locals()["compressed_aln_lin_"+str(i)+"_3"] = pickle.load(open("../../data/saved_models/torch_models/bert/compressed_80_aln_lin_"+str(i)+"_3.p", "rb"))
                all_bert_compressed_tasks.append(locals()["compressed_aln_lin_"+str(i)+"_1"])
                all_bert_compressed_tasks.append(locals()["compressed_aln_lin_"+str(i)+"_2"])
                all_bert_compressed_tasks.append(locals()["compressed_aln_lin_"+str(i)+"_3"])
                all_bert_compressed_tasks_names.append("bert_{}_1".format(i))
                all_bert_compressed_tasks_names.append("bert_{}_2".format(i))
                all_bert_compressed_tasks_names.append("bert_{}_3".format(i))

            compressed_final_lin = pickle.load(open("../../data/saved_models/torch_models/bert/compressed_80_final_lin.p", "rb" ))
            all_bert_compressed_tasks.append(compressed_final_lin)
            all_bert_compressed_tasks_names.append("bert_final")

            r = random.randrange(len(all_bert_compressed_tasks)) 
            all_compressed_tasks.append(all_bert_compressed_tasks[r])
            all_compressed_tasks_names.append(all_bert_compressed_tasks_names[r])

        if model == "alexnet":
            all_alexnet_compressed_tasks = []
            alexnet_lin_1 = pickle.load(open("/home/zhizhenz/optical-inference/data/saved_models/torch_models/alexnet/compressed_80_aln_lin_1.p", "rb"))
            alexnet_lin_2 = pickle.load(open("/home/zhizhenz/optical-inference/data/saved_models/torch_models/alexnet/compressed_80_aln_lin_2.p", "rb"))
            alexnet_lin_3 = pickle.load(open("/home/zhizhenz/optical-inference/data/saved_models/torch_models/alexnet/compressed_80_aln_lin_3.p", "rb"))
            all_alexnet_compressed_tasks = [alexnet_lin_1, alexnet_lin_2, alexnet_lin_3]
            
            r = random.randrange(len(all_alexnet_compressed_tasks)) 
            all_compressed_tasks.append(all_alexnet_compressed_tasks[r])
            all_compressed_tasks_names.append(["alexnet_1", "alexnet_2", "alexnet_3"][r])
        
        if model == "densenet":
            all_densenet_compressed_tasks = []
            densenet_lin_1 = pickle.load(open("../../data/saved_models/torch_models/densenet/compressed_80_aln_lin_1.p", "rb"))
            all_densenet_compressed_tasks = [densenet_lin_1]
            all_compressed_tasks += all_densenet_compressed_tasks
            all_compressed_tasks_names += ["densenet_1"]

        if model == "inception":
            all_inception_compressed_tasks = []
            inception_lin_1 = pickle.load(open("../../data/saved_models/torch_models/inception/compressed_80_aln_lin_1.p", "rb"))
            inception_lin_2 = pickle.load(open("../../data/saved_models/torch_models/inception/compressed_80_aln_lin_2.p", "rb"))
            all_inception_compressed_tasks = [inception_lin_1, inception_lin_2]

            r = random.randrange(len(all_inception_compressed_tasks)) 
            all_compressed_tasks.append(all_inception_compressed_tasks[r])
            all_compressed_tasks_names.append(["inception_1", "inception_2"][r])

        if model == "resnet18":
            all_resnet18_compressed_tasks = []
            resnet18_lin_1 = pickle.load(open("/home/zhizhenz/optical-inference/data/saved_models/torch_models/resnet18/compressed_80_aln_lin_1.p", "rb"))
            all_resnet18_compressed_tasks = [resnet18_lin_1]
            all_compressed_tasks += all_resnet18_compressed_tasks
            all_compressed_tasks_names += ["resnet_1"]

        if model == "vgg16":
            all_vgg16_compressed_tasks = []
            vgg16_lin_1 = pickle.load(open("../../data/saved_models/torch_models/vgg16/compressed_80_aln_lin_1.p", "rb"))
            vgg16_lin_2 = pickle.load(open("../../data/saved_models/torch_models/vgg16/compressed_80_aln_lin_2.p", "rb"))
            vgg16_lin_3 = pickle.load(open("../../data/saved_models/torch_models/vgg16/compressed_80_aln_lin_3.p", "rb"))
            all_vgg16_compressed_tasks = [vgg16_lin_1, vgg16_lin_2, vgg16_lin_3]

            r = random.randrange(len(all_vgg16_compressed_tasks)) 
            all_compressed_tasks.append(all_vgg16_compressed_tasks[r])
            all_compressed_tasks_names.append(["vgg_1", "vgg_2", "vgg_3"][r])

        if model == "dlrm":
            all_compressed_tasks.append("dlrm")  # a special note about DLRM, we will fill in its tetris size directly later
            all_compressed_tasks_names.append("dlrm")
    
    print("Load pickle models finish")

    return all_compressed_tasks, all_compressed_tasks_names


## apply placement to get flow-core assignment
def Scheduling(dir, tasks, task_names, flows, core_num, wave_num, scheduler_id, savefig, compact, compact_cmap, repetitive):
    sum_flows = len(flows)  # total number of flows
    cores = {}
    scheduling_runtime = {}

    if sum_flows > core_num:
        print(colored("- {} Running {} scheduling".format(scheduler_id, label[scheduler_id]), "green"))
        print("Cores are Insufficient for flows, need to stack the flows.")

        if scheduler_id == 0:
            start_time = time.time()
            cores = PerFlowLongFlowFirst(flows, core_num)
            scheduling_runtime[scheduler_id] = time.time() - start_time
            if compact:
                tasks_completion_time = VisualizeScheduleingMatrix(dir, tasks, flows, cores, colormap=compact_cmap, title=label[scheduler_id], savefig=savefig)
            else:
                tasks_completion_time = VisualizeScheduling(dir, tasks, flows, cores, title=label[scheduler_id], savefig=savefig)
            # print("tasks_completion_time", tasks_completion_time)
        
        elif scheduler_id == 1:
            start_time = time.time()
            cores = PerFlowShortFlowFirst(flows, core_num)
            scheduling_runtime[scheduler_id] = time.time() - start_time
            if compact:
                tasks_completion_time = VisualizeScheduleingMatrix(dir, tasks, flows, cores, colormap=compact_cmap, title=label[scheduler_id], savefig=savefig)
            else:
                tasks_completion_time = VisualizeScheduling(dir, tasks, flows, cores, title=label[scheduler_id], savefig=savefig)
            # print("tasks_completion_time", tasks_completion_time)
        
        elif scheduler_id == 2:
            start_time = time.time()
            cores = PerTaskLongFlowFirst(tasks, flows, core_num)
            scheduling_runtime[scheduler_id] = time.time() - start_time
            if compact:
                tasks_completion_time = VisualizeScheduleingMatrix(dir, tasks, flows, cores, colormap=compact_cmap, title=label[scheduler_id], savefig=savefig)
            else:
                tasks_completion_time = VisualizeScheduling(dir, tasks, flows, cores, title=label[scheduler_id], savefig=savefig)
            # print("tasks_completion_time", tasks_completion_time)
        
        elif scheduler_id == 3:
            start_time = time.time()
            cores = PerTaskShortFlowFirst(tasks, flows, core_num)
            scheduling_runtime[scheduler_id] = time.time() - start_time
            if compact:
                tasks_completion_time = VisualizeScheduleingMatrix(dir, tasks, flows, cores, colormap=compact_cmap, title=label[scheduler_id], savefig=savefig)
            else:
                tasks_completion_time = VisualizeScheduling(dir, tasks, flows, cores, title=label[scheduler_id], savefig=savefig)
            # print("tasks_completion_time", tasks_completion_time)
        
        elif scheduler_id == 4:
            start_time = time.time()
            cores = PerTaskRandomFlowFirst(tasks, flows, core_num)
            scheduling_runtime[scheduler_id] = time.time() - start_time
            if compact:
                tasks_completion_time = VisualizeScheduleingMatrix(dir, tasks, flows, cores, colormap=compact_cmap, title=label[scheduler_id], savefig=savefig)
            else:
                tasks_completion_time = VisualizeScheduling(dir, tasks, flows, cores, title=label[scheduler_id], savefig=savefig)
            # print("tasks_completion_time", tasks_completion_time)
        
        elif scheduler_id == 5:
            start_time = time.time()
            r_tasks, r_flows, cores = PerTaskPerCoreRectangleShortFlowFirst(tasks, flows, core_num, wave_num=wave_num)
            scheduling_runtime[scheduler_id] = time.time() - start_time
            if compact:
                tasks_completion_time = VisualizeScheduleingMatrix(dir, r_tasks, r_flows, cores, colormap=compact_cmap, title=label[scheduler_id], savefig=savefig)
            else:
                tasks_completion_time = VisualizeScheduling(dir, r_tasks, r_flows, cores, title=label[scheduler_id], savefig=savefig)
            # print("tasks_completion_time", tasks_completion_time)

        elif scheduler_id == 6:
            start_time = time.time()
            r_tasks, r_flows, cores = PerTaskPerCoreRectangleStatefulShortFlowFirst(tasks, flows, core_num, wave_num=wave_num)
            scheduling_runtime[scheduler_id] = time.time() - start_time
            if compact:
                tasks_completion_time = VisualizeScheduleingMatrix(dir, r_tasks, r_flows, cores, colormap=compact_cmap, title=label[scheduler_id], savefig=savefig)
            else:
                tasks_completion_time = VisualizeScheduling(dir, r_tasks, r_flows, cores, title=label[scheduler_id], savefig=savefig)
            # print("tasks_completion_time", tasks_completion_time)

        elif scheduler_id == 7:
            start_time = time.time()
            r_tasks, r_flows, cores = PerTaskAllCoreRectangleStatefulShortFlowFirst(tasks, flows, core_num, wave_num=wave_num)
            scheduling_runtime[scheduler_id] = time.time() - start_time
            if compact:
                tasks_completion_time = VisualizeScheduleingMatrix(dir, r_tasks, r_flows, cores, colormap=compact_cmap, title=label[scheduler_id], savefig=savefig)
            else:
                tasks_completion_time = VisualizeScheduling(dir, r_tasks, r_flows, cores, title=label[scheduler_id], savefig=savefig)
            # print("tasks_completion_time", tasks_completion_time)

        elif scheduler_id == 8:
            start_time = time.time()
            cores = PerTaskAllCoreStatefulRectangleFlowFirst(tasks, flows, core_num, wave_num=wave_num)
            scheduling_runtime[scheduler_id] = time.time() - start_time
            if compact:
                tasks_completion_time = VisualizeScheduleingMatrix(dir, tasks, flows, cores, colormap=compact_cmap, title=label[scheduler_id], savefig=savefig)
            else:
                tasks_completion_time = VisualizeScheduling(dir, tasks, flows, cores, title=label[scheduler_id], savefig=savefig)
            # print("tasks_completion_time", tasks_completion_time)

        elif scheduler_id == 9:
            start_time = time.time()
            cores = PerTaskAllCoreStatefulRectangleFlowFirst2(tasks, flows, core_num, wave_num=wave_num)
            scheduling_runtime[scheduler_id] = time.time() - start_time
            if compact:
                tasks_completion_time = VisualizeScheduleingMatrix(dir, tasks, flows, cores, colormap=compact_cmap, title=label[scheduler_id], savefig=savefig)
            else:
                tasks_completion_time = VisualizeScheduling(dir, tasks, flows, cores, title=label[scheduler_id], savefig=savefig)
            # print("tasks_completion_time", tasks_completion_time)     

        elif scheduler_id == 10:
            start_time = time.time()
            cores = PerTaskAllCoreStatefulRectangleFlowPreserveStateFirst(tasks, flows, core_num, wave_num=wave_num)
            scheduling_runtime[scheduler_id] = time.time() - start_time
            if compact:
                tasks_completion_time = VisualizeScheduleingMatrix(dir, tasks, flows, cores, colormap=compact_cmap, title=label[scheduler_id], savefig=savefig)
            else:
                tasks_completion_time = VisualizeScheduling(dir, tasks, flows, cores, title=label[scheduler_id], savefig=savefig)
            # print("tasks_completion_time", tasks_completion_time)     

        elif scheduler_id == 11:
            start_time = time.time()
            cores = RoundRobinFIFO(tasks, flows, core_num)
            scheduling_runtime[scheduler_id] = time.time() - start_time
            if compact:
                tasks_completion_time = VisualizeScheduleingMatrix(dir, tasks, flows, cores, colormap=compact_cmap, title=label[scheduler_id], savefig=savefig)
            else:
                tasks_completion_time = VisualizeScheduling(dir, tasks, flows, cores, title=label[scheduler_id], savefig=savefig)
            # print("tasks_completion_time", tasks_completion_time) 
            
        elif scheduler_id == 12:
            start_time = time.time()
            cores = RoundRobinSJF(tasks, flows, core_num)
            scheduling_runtime[scheduler_id] = time.time() - start_time
            if compact:
                tasks_completion_time = VisualizeScheduleingMatrix(dir, tasks, flows, cores, colormap=compact_cmap, title=label[scheduler_id], savefig=savefig)
            else:
                tasks_completion_time = VisualizeScheduling(dir, tasks, flows, cores, title=label[scheduler_id], savefig=savefig)
            # print("tasks_completion_time", tasks_completion_time)     

        elif scheduler_id == 13:
            start_time = time.time()
            cores = LookupTableSJFMax(tasks, task_names, flows, core_num)
            scheduling_runtime[scheduler_id] = time.time() - start_time
            if compact:
                tasks_completion_time = VisualizeScheduleingMatrix(dir, tasks, flows, cores, colormap=compact_cmap, title=label[scheduler_id], savefig=savefig)
            else:
                tasks_completion_time = VisualizeScheduling(dir, tasks, flows, cores, title=label[scheduler_id], savefig=savefig)
            # print("tasks_completion_time", tasks_completion_time)     

        elif scheduler_id == 14:
            start_time = time.time()
            cores = LookupTableSJFSum(tasks, task_names, flows, core_num)
            scheduling_runtime[scheduler_id] = time.time() - start_time
            if compact:
                tasks_completion_time = VisualizeScheduleingMatrix(dir, tasks, flows, cores, colormap=compact_cmap, title=label[scheduler_id], savefig=savefig)
            else:
                tasks_completion_time = VisualizeScheduling(dir, tasks, flows, cores, title=label[scheduler_id], savefig=savefig)
            # print("tasks_completion_time", tasks_completion_time) 

        elif scheduler_id == 15:
            start_time = time.time()
            cores = LookupTableFIFO(tasks, task_names, flows, core_num)
            scheduling_runtime[scheduler_id] = time.time() - start_time
            if compact:
                tasks_completion_time = VisualizeScheduleingMatrix(dir, tasks, flows, cores, colormap=compact_cmap, title=label[scheduler_id], savefig=savefig)
            else:
                tasks_completion_time = VisualizeScheduling(dir, tasks, flows, cores, title=label[scheduler_id], savefig=savefig)
            # print("tasks_completion_time", tasks_completion_time) 
       
        elif scheduler_id == 16:
            start_time = time.time()
            cores = LookupTableEqualShiftedFIFO(tasks, task_names, flows, core_num)
            scheduling_runtime[scheduler_id] = time.time() - start_time
            if compact:
                tasks_completion_time = VisualizeScheduleingMatrix(dir, tasks, flows, cores, colormap=compact_cmap, title=label[scheduler_id], savefig=savefig)
            else:
                tasks_completion_time = VisualizeScheduling(dir, tasks, flows, cores, title=label[scheduler_id], savefig=savefig)
            # print("tasks_completion_time", tasks_completion_time) 

        elif scheduler_id == 17:
            start_time = time.time()
            cores = LookupTablePermutedFIFO(tasks, task_names, flows, core_num)
            scheduling_runtime[scheduler_id] = time.time() - start_time
            if compact:
                tasks_completion_time = VisualizeScheduleingMatrix(dir, tasks, flows, cores, colormap=compact_cmap, title=label[scheduler_id], savefig=savefig)
            else:
                tasks_completion_time = VisualizeScheduling(dir, tasks, flows, cores, title=label[scheduler_id], savefig=savefig)
            # print("tasks_completion_time", tasks_completion_time) 

        elif scheduler_id == 18:
            start_time = time.time()
            cores = LongestFlowFirstFIFO(tasks, flows, core_num)
            scheduling_runtime[scheduler_id] = time.time() - start_time
            if compact:
                tasks_completion_time = VisualizeScheduleingMatrix(dir, tasks, flows, cores, colormap=compact_cmap, title=label[scheduler_id], savefig=savefig)
            else:
                tasks_completion_time = VisualizeScheduling(dir, tasks, flows, cores, title=label[scheduler_id], savefig=savefig)
            # print("tasks_completion_time", tasks_completion_time) 

    else:
        print("Cores are Sufficient for flows, first fit assignment")
        tasks_completion_time = {}
        scheduling_runtime = {}

    req_num = len(tasks)
    pickle.dump(tasks_completion_time, open(os.path.join(dir, "data/{}_{}_{}_{}_completion_time.p".format(core_num, req_num, scheduler_id, repetitive)), "wb"))
    pickle.dump(cores, open(os.path.join(dir, "data/{}_{}_{}_{}_cores.p".format(core_num, req_num, scheduler_id, repetitive)), "wb"))
    pickle.dump(scheduling_runtime, open(os.path.join(dir, "data/{}_{}_{}_{}_runtime.p".format(core_num, req_num, scheduler_id, repetitive)), "wb"))

    print("{} results saved in {}".format(label[scheduler_id], dir))
    print(colored("Done with {} scheduling".format(label[scheduler_id]), "red"))

    return tasks_completion_time, scheduling_runtime


def ParseOpt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='directory to store the results')
    parser.add_argument('--readwritepickle', type=int, default=0, help='read from pickle tasks')
    # parser.add_argument('--dnn', action='append', help='list of pruned DNNs to be scheduled')
    parser.add_argument('--requestnum', type=int, help='number of DNN requests')
    # parser.add_argument('--modelstatistics', type=bool, default=False, help='generate model statistics')
    parser.add_argument('--corenum', type=int, default=1000, help='number of optical computing cores')
    parser.add_argument('--wavenum', type=int, default=1, help='number of WDM wavelengths in a optical computing core')
    parser.add_argument('--schedulerid', type=int, default=0, help='ID for scheduling algorithms')
    parser.add_argument('--savefig', type=bool, default=True, help='if we save result figures into files')
    parser.add_argument('--compact', type=bool, default=True, help='generate compact scheduling figure')
    parser.add_argument('--cmap', type=str, default="viridis", help='color map')
    parser.add_argument('--repetitive', type=int, help='repetitive exp')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    return opt


def main(opt):
    dir = opt.dir
    print(dir)
    with open(os.path.join(dir, 'arguments.txt'), 'w') as f:
        f.writelines(["dir", opt.dir, "\n"])
        # f.writelines(["dnn", str(opt.dnn), "\n"])
        f.writelines(["requestnum", str(opt.requestnum), "\n"])
        # f.writelines(["readwritepickle", str(opt.readwritepickle)], "\n")
        # f.writelines(["modelstatistics", str(opt.modelstatistics), "\n"])
        f.writelines(["corenum", str(opt.corenum), "\n"])
        f.writelines(["wavenum", str(opt.wavenum), "\n"])
        f.writelines(["schedulerid", str(opt.schedulerid), "\n"])
        f.writelines(["savefig", str(opt.savefig), "\n"])
        f.writelines(["compact", str(opt.compact), "\n"])
        f.writelines(["cmap", opt.cmap])

    if opt.readwritepickle == 0:
        # model_list = opt.dnn
        all_models = ["alexnet", "resnet18", "bert", "densenet", "vgg16", "inception", "dlrm"]
        model_list = []
        dlrm_location = random.randrange(opt.requestnum) 
        for i in range(opt.requestnum):
            if i == dlrm_location:
                model_list.append("dlrm")
            else:
                r = random.randrange(len(all_models)) 
                model_list.append(all_models[r])
        print("Scheduling {}".format(model_list))

        ## get the matrices from models
        all_compressed_tasks, all_compressed_tasks_names = LoadPickleModels(model_list)

        # print(opt.modelstatistics)
        # ## plot all matrix statistic
        # if opt.modelstatistics:
        #     vector_lengths = MatrixListStatistics(all_compressed_tasks)
        #     vector_lengths.sort()
        #     plt.plot(vector_lengths, marker="o")
        #     plt.xlabel("Row vector of matrices")
        #     plt.ylabel("Number of on-zero values")
        #     plt.show()
        #     plt.savefig(dir+"/all_matrix_statistic.png")

        ## translate ML model matrices into tasks/flows
        tasks, flows = Matrix2Task(all_compressed_tasks, wdm_ch_num=opt.wavenum)

        pickle.dump(tasks, open(os.path.join(dir, "data/tasks_{}_{}.p".format(opt.requestnum, opt.repetitive)), "wb"))
        pickle.dump(flows, open(os.path.join(dir, "data/flows_{}_{}.p".format(opt.requestnum, opt.repetitive)), "wb"))
        pickle.dump(all_compressed_tasks_names, open(os.path.join(dir, "data/namestolookup_{}_{}.p".format(opt.requestnum, opt.repetitive)), "wb"))
    
    else:
        tasks = pickle.load(open(os.path.join(dir, "data/tasks_{}_{}.p".format(opt.requestnum, opt.repetitive)), "rb"))
        flows = pickle.load(open(os.path.join(dir, "data/flows_{}_{}.p".format(opt.requestnum, opt.repetitive)), "rb"))
        all_compressed_tasks_names = pickle.load(open(os.path.join(dir, "data/namestolookup_{}_{}.p".format(opt.requestnum, opt.repetitive)), "rb"))

        ## run scheduling algorithm
        Scheduling(dir, tasks, all_compressed_tasks_names, flows, core_num=opt.corenum, wave_num=opt.wavenum, scheduler_id=opt.schedulerid, savefig=opt.savefig, compact=opt.compact, compact_cmap=opt.cmap, repetitive=opt.repetitive)
        

if __name__ == "__main__":
    opt = ParseOpt()
    main(opt)    