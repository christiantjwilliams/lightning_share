from typing import NoReturn, List
from heapq import merge
import numpy as np
from numpy import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import _pickle as cPickle
# import numpy.typing as npt
import argparse

#def ParseOpt(known=False):
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--gran', type=int, help='granularity of sim')
#
#    opt = parser.parse_known_args()[0] if known else parser.parse_args()
#
#    return opt
#
#opt = ParseOpt()

# TODO: make granularity reduction method rigorous to all models, currently only works for this specific VGG issue
#min_task_size = opt.gran
#resnet_task_size = 1000000
#anet_task_size = 1000

def calc_linear_vector_mult(input_size, output_size):
    return input_size*output_size

def calc_conv2d_vector_mult(input_channel, kernel_size, output_shape):
    num_vector_mult = input_channel*output_shape[1]*output_shape[2]*output_shape[3]
    vector_len = kernel_size[0] * kernel_size[1]
    # print("Number of vector multiplication needed: ", num_vector_mult)
    # print("Length of the vector: ", vector_len)
    return num_vector_mult, vector_len

## LeNet
LeNet = []
LeNet.append([784]*300)
LeNet.append([300]*100)
LeNet.append([100]*10)

## AlexNet (Mingran)
num_mult_list_alexnet = []
num_mult_list_alexnet.append(calc_conv2d_vector_mult(3, [11, 11], [-1, 64, 55, 55])[0]) # layer 1
num_mult_list_alexnet.append(calc_conv2d_vector_mult(64, [5, 5], [-1, 192, 27, 27])[0]) # layer 4
num_mult_list_alexnet.append(calc_conv2d_vector_mult(192, [3, 3], [-1, 192, 13, 13])[0]) # layer 7
num_mult_list_alexnet.append(calc_conv2d_vector_mult(384, [3, 3], [-1, 256, 13, 13])[0]) # layer 9
num_mult_list_alexnet.append(calc_conv2d_vector_mult(256, [3, 3], [-1, 256, 13, 13])[0]) # layer 11

#AlexNet = [[min_task_size]*int(num_mult/(min_task_size/9)+1) for num_mult in tqdm(num_mult_list_alexnet, desc='building AlexNet...')]
#AlexNet = [[anet_task_size]*int(num_mult/(anet_task_size/9)+1) for num_mult in num_mult_list_alexnet]

#AlexNet.append([min_task_size]*int(4096/(min_task_size/9216)+1))
#AlexNet.append([anet_task_size]*int(4096/(anet_task_size/4096)+1))
#AlexNet.append([anet_task_size]*int(1000/(anet_task_size/4096)+1))

AlexNet = [[9]*num_mult for num_mult in tqdm(num_mult_list_alexnet, desc='building AlexNet...')]

AlexNet.append([9216]*4096)
AlexNet.append([4096]*4096)
AlexNet.append([4096]*1000)

num_mult_list_resnet18 = []
num_mult_list_resnet18.append(calc_conv2d_vector_mult(3, [7, 7], [-1, 64, 224, 224])[0])
num_mult_list_resnet18.append(calc_conv2d_vector_mult(64, [7, 7], [-1, 64, 224, 224])[0])
num_mult_list_resnet18.append(calc_conv2d_vector_mult(64, [7, 7], [-1, 64, 224, 224])[0])
num_mult_list_resnet18.append(calc_conv2d_vector_mult(64, [7, 7], [-1, 64, 224, 224])[0])
num_mult_list_resnet18.append(calc_conv2d_vector_mult(64, [7, 7], [-1, 64, 224, 224])[0])
num_mult_list_resnet18.append(calc_conv2d_vector_mult(64, [7, 7], [-1, 128, 112, 112])[0])
num_mult_list_resnet18.append(calc_conv2d_vector_mult(128, [7, 7], [-1, 128, 112, 112])[0])
num_mult_list_resnet18.append(calc_conv2d_vector_mult(64, [7, 7], [-1, 128, 112, 112])[0])
num_mult_list_resnet18.append(calc_conv2d_vector_mult(128, [7, 7], [-1, 128, 112, 112])[0])
num_mult_list_resnet18.append(calc_conv2d_vector_mult(128, [7, 7], [-1, 256, 56, 56])[0])
num_mult_list_resnet18.append(calc_conv2d_vector_mult(256, [7, 7], [-1, 256, 56, 56])[0])
num_mult_list_resnet18.append(calc_conv2d_vector_mult(256, [7, 7], [-1, 256, 56, 56])[0])
num_mult_list_resnet18.append(calc_conv2d_vector_mult(256, [7, 7], [-1, 256, 56, 56])[0])
num_mult_list_resnet18.append(calc_conv2d_vector_mult(256, [7, 7], [-1, 512, 28, 28])[0])
num_mult_list_resnet18.append(calc_conv2d_vector_mult(512, [7, 7], [-1, 512, 28, 28])[0])
num_mult_list_resnet18.append(calc_conv2d_vector_mult(512, [7, 7], [-1, 512, 14, 14])[0])
num_mult_list_resnet18.append(calc_conv2d_vector_mult(512, [7, 7], [-1, 512, 14, 14])[0])

## ResNet18 ##

#ResNet18 = [[min_task_size]*int(num_mult/(min_task_size/9)+1) for num_mult in tqdm(num_mult_list_resnet18, desc='building ResNet18...')]
#ResNet18 = [[resnet_task_size]*int(num_mult/(resnet_task_size/9)+1) for num_mult in num_mult_list_resnet18]

#ResNet18.append([resnet_task_size]*int(512/(resnet_task_size/1000)+1))

ResNet18 = [[9]*num_mult for num_mult in tqdm(num_mult_list_resnet18, desc='building ResNet18...')]

ResNet18.append([1000]*512)

# VGG16 time estimation (Mingran)
num_mult_list_vgg16 = []
num_mult_list_vgg16.append(calc_conv2d_vector_mult(3, [3, 3], [-1, 64, 224, 224])[0]) # layer 1
num_mult_list_vgg16.append(calc_conv2d_vector_mult(64, [3, 3], [-1, 64, 224, 224])[0]) # layer 3
num_mult_list_vgg16.append(calc_conv2d_vector_mult(64, [3, 3], [-1, 128, 112, 112])[0]) # layer 6
num_mult_list_vgg16.append(calc_conv2d_vector_mult(128, [3, 3], [-1, 128, 112, 112])[0]) # layer 8
num_mult_list_vgg16.append(calc_conv2d_vector_mult(128, [3, 3], [-1, 256, 56, 56])[0]) # layer 11
num_mult_list_vgg16.append(calc_conv2d_vector_mult(256, [3, 3], [-1, 256, 56, 56])[0]) # layer 13
num_mult_list_vgg16.append(calc_conv2d_vector_mult(256, [3, 3], [-1, 256, 56, 56])[0]) # layer 15
num_mult_list_vgg16.append(calc_conv2d_vector_mult(256, [3, 3], [-1, 512, 28, 28])[0]) # layer 18
num_mult_list_vgg16.append(calc_conv2d_vector_mult(512, [3, 3], [-1, 512, 28, 28])[0]) # layer 20
num_mult_list_vgg16.append(calc_conv2d_vector_mult(512, [3, 3], [-1, 512, 28, 28])[0]) # layer 22
num_mult_list_vgg16.append(calc_conv2d_vector_mult(512, [3, 3], [-1, 512, 14, 14])[0]) # layer 25
num_mult_list_vgg16.append(calc_conv2d_vector_mult(512, [3, 3], [-1, 512, 14, 14])[0]) # layer 27
num_mult_list_vgg16.append(calc_conv2d_vector_mult(512, [3, 3], [-1, 512, 14, 14])[0]) # layer 29

## VGG16 ##

#VGG16 = [[min_task_size]*int(num_mult/(min_task_size/9)+1) for num_mult in tqdm(num_mult_list_vgg16, desc='building VGG16...')]
#VGG16 = [[min_task_size]*int(num_mult/(min_task_size/9)+1) for num_mult in num_mult_list_vgg16]


#VGG16.append([min_task_size]*int(4096/(min_task_size/25088)+1))
#VGG16.append([min_task_size]*int(4096/(min_task_size/4096)+1))
#VGG16.append([min_task_size]*int(1000/(min_task_size/25088)+1))

VGG16 = [[9]*num_mult for num_mult in tqdm(num_mult_list_vgg16, desc='building VGG16...')]

VGG16.append([25088]*4096) # layer 23
VGG16.append([4096]*4096) # layer 16
VGG16.append([4096]*1000) # layer 29

# VGG19 time estimation
num_mult_list_vgg19 = []
num_mult_list_vgg19.append(calc_conv2d_vector_mult(3, [3, 3], [-1, 64, 224, 224])[0]) # layer 1
num_mult_list_vgg19.append(calc_conv2d_vector_mult(64, [3, 3], [-1, 64, 224, 224])[0]) # layer 3
num_mult_list_vgg19.append(calc_conv2d_vector_mult(64, [3, 3], [-1, 128, 112, 112])[0]) # layer 6
num_mult_list_vgg19.append(calc_conv2d_vector_mult(128, [3, 3], [-1, 128, 112, 112])[0]) # layer 8
num_mult_list_vgg19.append(calc_conv2d_vector_mult(128, [3, 3], [-1, 256, 56, 56])[0]) # layer 11
num_mult_list_vgg19.append(calc_conv2d_vector_mult(256, [3, 3], [-1, 256, 56, 56])[0]) # layer 13
num_mult_list_vgg19.append(calc_conv2d_vector_mult(256, [3, 3], [-1, 256, 56, 56])[0]) # layer 15
num_mult_list_vgg19.append(calc_conv2d_vector_mult(256, [3, 3], [-1, 256, 56, 56])[0]) # layer 17
num_mult_list_vgg19.append(calc_conv2d_vector_mult(256, [3, 3], [-1, 512, 28, 28])[0]) # layer 20
num_mult_list_vgg19.append(calc_conv2d_vector_mult(512, [3, 3], [-1, 512, 28, 28])[0]) # layer 22
num_mult_list_vgg19.append(calc_conv2d_vector_mult(512, [3, 3], [-1, 512, 28, 28])[0]) # layer 24
num_mult_list_vgg19.append(calc_conv2d_vector_mult(512, [3, 3], [-1, 512, 28, 28])[0]) # layer 26
num_mult_list_vgg19.append(calc_conv2d_vector_mult(512, [3, 3], [-1, 512, 14, 14])[0]) # layer 29
num_mult_list_vgg19.append(calc_conv2d_vector_mult(512, [3, 3], [-1, 512, 14, 14])[0]) # layer 31
num_mult_list_vgg19.append(calc_conv2d_vector_mult(512, [3, 3], [-1, 512, 14, 14])[0]) # layer 33
num_mult_list_vgg19.append(calc_conv2d_vector_mult(512, [3, 3], [-1, 512, 14, 14])[0]) # layer 35

## VGG19 ##

#VGG19 = [[min_task_size]*int(num_mult/(min_task_size/9)+1) for num_mult in tqdm(num_mult_list_vgg19, desc='building VGG19...')]
#VGG19 = [[min_task_size]*int(num_mult/(min_task_size/9)+1) for num_mult in num_mult_list_vgg19]

#VGG19.append([min_task_size]*int(4096/(min_task_size/25088)+1))
#VGG19.append([min_task_size]*int(4096/(min_task_size/4096)+1))
#VGG19.append([min_task_size]*int(1000/(min_task_size/25088)+1))

VGG19 = [[9]*num_mult for num_mult in tqdm(num_mult_list_vgg19, desc='building VGG19...')]

VGG19.append([25088]*4096) # layer 23
VGG19.append([4096]*4096) # layer 16
VGG19.append([4096]*1000) # layer 29

## BERT ##

BERT = [[30522]*1024, [512]*1024, [2]*1024, [1024]*2]
for _ in tqdm(range(24), desc='building MegatronBERT...'):
    #BERT.append([min_task_size]*int(1024/(min_task_size/1024)+1))
    BERT.append([1024]*(1024+1))
    for i in range(16):
        #BERT.append([min_task_size]*int((64+1)/(min_task_size/768)+1))
        #BERT.append([min_task_size]*int((64+1)/(min_task_size/768)+1))
        #BERT.append([min_task_size]*int((64+1)/(min_task_size/768)+1))
        BERT.append([1024]*(64+1))
        BERT.append([1024]*(64+1))
        BERT.append([1024]*(64+1))
        #BERT.append([min_task_size]*int((768+1)/(min_task_size/768)+1))
        #BERT.append([min_task_size]*int((1+1)/(min_task_size/768)+1))
        #BERT.append([min_task_size]*int((768+1)/(min_task_size/3072)+1))
        #BERT.append([min_task_size]*int((3072+1)/(min_task_size/768)+1))
        #BERT.append([min_task_size]*int((1+1)/(min_task_size/768)+1))
        BERT.append([1024]*(1024+1))
        BERT.append([1024]*(1+1))
        BERT.append([1024]*(1024+1))
        BERT.append([1024]*(3072+1))
        BERT.append([1024]*(3072+1))
        BERT.append([1024]*(1+1))

BERT.append([1024]*(1024+1))

GPT2 = [[50257]*(1600)+[1024,]] # token embedding + positional encoding
for _ in tqdm(range(48), desc='building GPT-2...'):
    GPT2.append([1600]*(1600+1)) # attention
    GPT2.append([1600]*(1600*3+1)) # attention
    GPT2.append([1600]*(1600*4+1)) #ffnn
    GPT2.append([1600*4]*(1600+1)) #ffnn
    #GPT2.append([min_task_size]*int(1600/(min_task_size/1600)+1))
    #GPT2.append([min_task_size]*int((1600*3)/(min_task_size/1600)+1))
    #GPT2.append([min_task_size]*int((1600*4)/(min_task_size/1600)+1))
    #GPT2.append([min_task_size]*int(1600/(min_task_size/(1600*4))+1))
GPT2.append([1600]*(1600*4)) # decoding
#GPT2.append([min_task_size]*int(1600/(min_task_size/(1600*4))+1))

## ChatGPT ##

ChatGPT = [[50257]*(1600)+[1024,]] # token embedding + positional encoding
for _ in tqdm(range(5750), desc='building ChatGPT*...'):
    ChatGPT.append([1600]*(1600+1)) # attention
    ChatGPT.append([1600]*(1600*3+1)) # attention
    ChatGPT.append([1600]*(1600*4+1)) #ffnn
    ChatGPT.append([1600*4]*(1600+1)) #ffnn
    #ChatGPT.append([min_task_size]*int(1600/(min_task_size/1600)+1))
    #ChatGPT.append([min_task_size]*int((1600*3)/(min_task_size/1600)+1))
    #ChatGPT.append([min_task_size]*int((1600*4)/(min_task_size/1600)+1))
    #ChatGPT.append([min_task_size]*int(1600/(min_task_size/(1600*4))+1))
ChatGPT.append([1600]*(1600*4)) # decoding
#ChatGPT.append([min_task_size]*int(1600/(min_task_size/(1600*4))+1))

## DLRM ##

embedding_sizes_dlrm = [[9980333],[36084],[17217],[7378],[20134],[3],[7112],[1442],[61], \
    [9758201],[1333352],[313829],[10],[2208],[11156],[112],[4],[970],[14],[9994222],[7267859], \
    [9946608],[415421],[12420],[101],[36]]

DLRM = [emb_size*64 for emb_size in tqdm(embedding_sizes_dlrm, desc='building DLRM...')]
#DLRM = [emb_size*64 for emb_size in embedding_sizes_dlrm]

ff_layer_sizes_dlrm = [[13],[512],[256],[415],[512],[512],[256]]
num_mult_ff_dlrm = [512,256,64,512,512,256,1]

for i in range(len(ff_layer_sizes_dlrm)):
    DLRM.append(ff_layer_sizes_dlrm[i]*num_mult_ff_dlrm[i])

MODEL_DICT = {'AlexNet':AlexNet,'ResNet18':ResNet18,'VGG16':VGG16,'VGG19':VGG19,'BERT': BERT,'GPT2':GPT2,'ChatGPT':ChatGPT,'DLRM':DLRM}
TOTAL_MODEL_COUNT = len(MODEL_DICT.values())

#SIMULATION_T = sum(sum(layer for layer in model) for model in MODEL_DICT.values())/TOTAL_MODEL_COUNT

#print('simulation objects (tasks) per model:')
#tasksum = 0
#for model in MODEL_DICT:
#    num_tasks = sum(len(model_layer) for model_layer in MODEL_DICT[model])
#    print(model + ' tasks: ', num_tasks)
#    tasksum += num_tasks
#

#print('model sizes:')
#total_size = 0
#for model in MODEL_DICT:
#    model_size = sum(sum(model_layer) for model_layer in MODEL_DICT[model])
#    total_size += model_size
#    print(model + ' size: ', model_size)

#print('average model size: ', total_size/len(MODEL_DICT.values()))

#SUM_MODEL_SIZES = sum(sum([sum(layer) for layer in model]) for model in MODEL_DICT.values())

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

MODEL_LAYER_NUMS = {}

#for model in MODEL_DICT:
#    MODEL_LAYER_NUMS[model] = len(MODEL_DICT[model])
#    for i in range(len(MODEL_DICT[model])):
#        new_job = Job(None, None, model, i, None, None, None)
#        job_tasks = []
#        for j in range(len(MODEL_DICT[model][i])):
#            is_first_layer = True if i == 0 else False
#            job_tasks.append(Task(None, None, model, MODEL_DICT[model][i][j], None, None, is_first_layer)) 
#        new_job.add_tasks(job_tasks)
#        with open(f"built_models/{model}_layer_{i}.p", 'wb') as f:
#            # Pickle the model layer using the highest protocol available.
#            pickle.dump(new_job, f, pickle.HIGHEST_PROTOCOL)

for i in range(len(LeNet)):
    model = 'LeNet'
    new_job = Job(None, None, model, i, None, None, None)
    job_tasks = []
    for j in range(len(LeNet[i])):
        is_first_layer = True if i == 0 else False
        job_tasks.append(Task(None, None, model, LeNet[i][j], None, None, is_first_layer))
    new_job.add_tasks(job_tasks)
    with open(f"built_models/{model}_layer_{i}.p", 'wb') as f:
        # Pickle the model layer using the highest protocol available.
        pickle.dump(new_job, f, pickle.HIGHEST_PROTOCOL)

with open(f'built_models/model_layer_nums.p', 'rb') as f:
    MODEL_LAYER_NUMS = cPickle.load(f)
    MODEL_LAYER_NUMS[model] = len(LeNet)

with open(f'built_models/model_layer_nums.p', 'wb') as f:
    pickle.dump(MODEL_LAYER_NUMS, f, pickle.HIGHEST_PROTOCOL)


#with open(f'built_models/model_layer_nums.p', 'wb') as f:
#    pickle.dump(MODEL_LAYER_NUMS, f, pickle.HIGHEST_PROTOCOL)
