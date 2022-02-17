import slayerSNN as snn
import numpy as np
from slayerSNN import slayer
import torch
from torchvision.datasets import CIFAR10
import torchvision
from torch.utils.data import DataLoader
from encoding import Encoding
from learningstats import learningStats
import torchvision.models as models
from datetime import date, datetime
from torch.profiler import profile, record_function, ProfilerActivity

netParams = snn.params("network.yaml")
device = torch.device("cuda")
# device = torch.cuda.set_device(0)

num_epochs = 10
pixel_to_time_index = {}
load = False

transformation = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), lambda x: x*255])

def calculate_spike_times():
    encode = Encoding(netParams['simulation']['tSample'])
    for i in range(256):
        pixel_to_time_index[i] = encode._calculate_spike_times(i)



def image_to_spike_tensor(input:torch.Tensor, empty_array:torch.Tensor, Ts:int):
    # xEvent = np.arange(input.shape[2])
    # yEvent = np.arange(input.shape[1])
    # cEvent = np.arange(input.shape[0])
    # tEvent = np.arange(calculate_spike_times())
    
    # # empty_array[cEvent, xEvent, yEvent, tEvent] = 1/Ts
    for B in range(input.shape[0]):
        for C in range(input.shape[1]):
            for H in range(input.shape[2]):
                for W in range(input.shape[3]):
                    pixel = np.array(pixel_to_time_index[int(input[B][C][H][W])])
                    # empty_array[C,H,W, pixel] = 1/Ts
                    empty_array[B][C][H][W][pixel] = 1/Ts

    return empty_array.to(device)

    
    

class Network(torch.nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.fc1 = self.slayer.dense((32,32,3), 240)
        self.fc2 = self.slayer.dense(240,10)
        self.nTimeBins = int(netParams['simulation']['tSample'] / netParams['simulation']['Ts'])
        self.timeStep  = int(netParams['simulation']['Ts'])

        self.pspLayer = self.slayer.pspLayer()

    def forward(self, input):

        spikes = image_to_spike_tensor(input, torch.zeros((1,3,32,32,self.nTimeBins)), self.timeStep)
        if int(torch.sum(spikes)) != int(torch.sum(input)):
            raise Exception("Error in conversion")
            
        layer1 = self.slayer.spike(self.slayer.psp(self.fc1(spikes)))
        layer2 = self.slayer.spike(self.slayer.psp(self.fc2(layer1)))


        return layer2

def overfit_single_batch():
    dataset_train = CIFAR10(root="", download=False, transform=transformation, train=True)
    dataset_test = CIFAR10(root="", download=False, transform=transformation, train=False)

    loaded_train = DataLoader(dataset_train, batch_size=1, num_workers=0, shuffle=False)
    loaded_test = DataLoader(dataset_test, batch_size=1, num_workers=0, shuffle=False)

    print("Finish loading data")

    print("Computing pixel time indexes")
    calculate_spike_times()
    print("Finish calculating pixel spike times")

    
    network = Network().to(device)
    criterion = snn.loss(netParams).to(device)

    if load == True:
        network.load_state_dict(torch.load("network1"))

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001, amsgrad=True)

    if load == True:
        optimizer.load_state_dict(torch.load("optimizer1"))

    stats = learningStats()



    (sample, label) = next(iter(loaded_train))
    sample = sample.to(device)
    label = label.to(device)

    desired = torch.zeros((10,1,1,1))
    desired[label,...] = 1

    for i in range(1000):
        output = network.forward(sample)
        loss = criterion.numSpikes(output, desired)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'The loss is {loss.item()}')



# with profile(activities=[
#         ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#     with record_function("model_inference"):
#         overfit_single_batch()

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        

if __name__ == "__main__":
    dataset_train = CIFAR10(root="", download=False, transform=transformation, train=True)
    dataset_test = CIFAR10(root="", download=False, transform=transformation, train=False)

    loaded_train = DataLoader(dataset_train, batch_size=1, num_workers=0, shuffle=False)
    loaded_test = DataLoader(dataset_test, batch_size=1, num_workers=0, shuffle=False)

    print("Finish loading data")

    print("Computing pixel time indexes")
    calculate_spike_times()
    print("Finish calculating pixel spike times")

    network = Network().to(device)
    criterion = snn.loss(netParams).to(device)

    if load == True:
        network.load_state_dict(torch.load("network1"))

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001, amsgrad=True)

    if load == True:
        optimizer.load_state_dict(torch.load("optimizer1"))

    stats = learningStats()

    for epoch in range(num_epochs):
        time_start = datetime.now()

        for i,(sample,label) in enumerate(loaded_train):
            sample.to(device)
            label.to(device)
            desired = torch.zeros((10,1,1,1))
            desired[int(label),...] = 1
            desired = desired.to(device)

            output = network(sample)

            loss = criterion.numSpikes(output, desired)

            stats.training.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
            stats.training.numSamples     += len(label)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            stats.training.lossSum += loss.cpu().data.item()

            if i%100 ==0:
                stats.print(epoch, i, (datetime.now() - time_start).total_seconds())
        
        torch.save(network.state_dict(), "network" + epoch)
        torch.save(optimizer.state_dict(), "optimizer" + epoch)
        print("Starting the testing")

        for i, (input, label) in enumerate(loaded_test, 0):
            input  = input.to(device)
            # target = label.to(device) 
            
            output = network.forward(input)

            stats.testing.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
            stats.testing.numSamples     += len(label)

            loss = criterion.numSpikes(output, label)
            stats.testing.lossSum += loss.cpu().data.item()
            if i%100 == 0:   stats.print(epoch, i)

        



    