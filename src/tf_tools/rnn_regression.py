import pandas as pd
import numpy as np
import random
import os

import tensorflow as tf
from keras import layers, Input, Model, models

dropout = 0.15

def coolRNN():
    
    model = models.Sequential([
        layers.SimpleRNN(20,return_sequences=True),
        layers.SimpleRNN(20,return_sequences=True),
        layers.Dense(1)
    ])
    
    return model