import numpy as np
import matplotlib.pyplot as plt

class NetParams():

    def __init__(self, threhshold, tauSR, scaleRef, tauRho, scaleRho ) -> None:
        self.threshold = threhshold
        self.tauSR = tauSR
        self.scaleRef = scaleRef
        self.tauRho = tauRho
        self.scaleRho = scaleRho


class SimParams():

    def __init__(self,marginRef, marginKer):
        self.marginRef = marginRef
        self.marginKer = marginKer