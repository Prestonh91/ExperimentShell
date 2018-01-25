import pandas as pd
import numpy as np

class file_reader:
    def __init__(self):
        pass

    def read_file(self, file_name):
        data = pd.read_csv(file_name, delimiter=",", skipinitialspace=True,
                           header=None)
        return data