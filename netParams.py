from cv2 import threshold
import numpy as np
import matplotlib.pyplot as plt

class NetParams():

    def __init__(self, threhshold) -> None:
        self.threshold = threshold