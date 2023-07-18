import time
import torch
import numpy as np
import torchvision.models as models

model_to_profile = "VGG"

if model_to_profile == "lenet":
    class LeNet(torch.nn.Module):
        '''
        simple 3-layer FF LeNet
        '''
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
            self.linear3(x)
            return x

    model = LeNet()
    model = torch.load("Deep-Compression-PyTorch/models/initial_model.ptmodel")

elif model_to_profile == "GPT":
    import random
    import string
    from transformers import GPT2Tokenizer, GPT2Model

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')

    def get_random_string(length):
        '''
        random 50 character string for GPT input
        '''
        # With combination of lower and upper case
        result_str = ''.join(random.choice(string.ascii_letters) for i in range(length))
        # print random string
        return result_str

elif model_to_profile == "VGG":
    model = models.vgg11(pretrained=True)
    dummy_input = torch.randn(1, 3,224,224,dtype=torch.float)

else:
    print("Error, no model chosen. set model_to_profile")

device = torch.device("cuda")
model.to(device)

def get_dummy_input(model_type):
    '''
    random rgb image for CV input
    '''
    if model_type == 'VGG' or model_type == 'lenet':
        return torch.randn(1, 3,224,224,dtype=torch.float).to(device)
    elif model_type == 'GPT':
        return tokenizer(get_random_string(50), return_tensors='pt').to(device)

def warm_up(model_type):
    '''
    warm up GPU
    '''
    if model_type == 'VGG' or model_type == 'lenet':
        _ = model(dummy_input)
    elif model_type == 'GPT':
        _ = model(**dummy_input)

def infer(model_type):
    '''
    run one inference
    '''
    if model_type == 'VGG' or model_type == 'lenet':
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
print(f'model size: {size_all_mb}MB')

# dummy input
dummy_input = get_dummy_input(model_to_profile)
repetitions = 300
timings=np.zeros((repetitions,1))

#GPU WARM-UP
for i in range(100):
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
