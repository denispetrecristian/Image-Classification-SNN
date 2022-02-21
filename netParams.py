import numpy as np
import matplotlib.pyplot as plt

class NetParams():

    def __init__(self, threhshold, tauSR, scaleRef, tauRho, scaleRho ) -> None:
        self.threshold = threhshold
        self.tauSR = tauSR
        self.scaleRef = scaleRef
        self.tauRho = tauRho
        self.scaleRho = scaleRho
        self.fc1 = 10
        self.fc2 = 4
        self.samples = 2
    
    def init_weights(self):
        self.weights1 = np.random.rand(self.fc1, self.fc2)
        self.weights2 = np.random.rand(self.fc2, self.samples)


class SimParams():

    def __init__(self,marginRef, marginKer, marginRho, timeline):
        self.marginRef = marginRef
        self.marginKer = marginKer
        self.marginRho = marginRho
        self.timeline = timeline