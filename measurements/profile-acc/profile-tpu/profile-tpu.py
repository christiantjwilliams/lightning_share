import torch
import torch_xla.core.xla_model as xm
import time
import pickle

import numpy as np
import torchvision.models as models

import sys

print('if stalls here, check tpu ip address')
device = xm.xla_device()

model_to_profile = "alexnet"

if model_to_profile == "lenet":
    class LeNet(torch.nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            self.linear1 = torch.nn.Linear(784, 300)
            self.activation = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(300, 100)
            self.softmax = torch.nn.Softmax()
            self.linear3 = torch.nn.Linear(100, 10)

        def forward(self, x):
            x = self.linear1(x)
            x = self.activation(x)
            x = self.linear2(x)
            x = self.softmax(x)
            x = self.linear3(x)
            return x

    model = LeNet()
    model.to(device)
    
    for i in range(1, 4):
        model.state_dict()[f'layer{i}.weight'] = pickle.load(open(f"fc_{i}.p", "rb"))

elif model_to_profile == "GPT":
    # GPT-2
    from transformers import GPT2Tokenizer, GPT2Model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2').to(device)

    import random
    import string

    def get_random_string(length):
        # With combination of lower and upper case
        result_str = ''.join(random.choice(string.ascii_letters) for i in range(length))
        # print random string
        return result_str

elif model_to_profile == "alexnet":
    model = models.alexnet(pretrained=True).to(device)

elif model_to_profile == "VGG-11":
    model = models.vgg11(pretrained=True).to(device)

elif model_to_profile == "VGG-16":
    model = models.vgg16(pretrained=True).to(device)
    
elif model_to_profile == "VGG-19":
    model = models.vgg19(pretrained=True).to(device)
    
def get_dummy_input(model_type):
    '''
    random rgb image for CV input
    '''
    if model_type in ['alexnet', 'VGG-11', 'VGG-16', 'VGG-19']:
        return torch.randn(1, 3,224,224,dtype=torch.float).to(device)
    elif model_type == 'lenet':
        return torch.randn(1, 3,784,784,dtype=torch.float).to(device)
    elif model_type == 'GPT':
        return tokenizer(get_random_string(50), return_tensors='pt').to(device)

def warm_up(model_type):
    '''
    warm up GPU
    '''
    if model_type in ['lenet', 'alexnet', 'VGG-11', 'VGG-16', 'VGG-19']:
        for i in range(100):
            _ = model(dummy_input)
    elif model_type == 'GPT':
        for i in range(100):
            _ = model(**dummy_input)

def infer(model_type):
    '''
    run one inference
    '''
    if model_type in ['lenet', 'alexnet', 'VGG-11', 'VGG-16', 'VGG-19']:
        _ = model(dummy_input)
    elif model_type == 'GPT':
        _ = model(**dummy_input)

# get model size
param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()
size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))

dummy_input = get_dummy_input(model_to_profile)
repetitions = 300
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
warm_up(model_to_profile)
# MEASURE PERFORMANCE
with torch.no_grad():
  for rep in range(repetitions):
     dummy_input = get_dummy_input(model_to_profile)
     start = time.time()
     infer(model_to_profile)
     inference_time = time.time() - start
     # WAIT FOR GPU SYNC
     timings[rep] = inference_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
#print(timings)
print(mean_syn)
