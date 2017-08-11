import pandas as pd
import os

directory = os.path.dirname(__file__)
data_path = os.path.join(directory, 'inline-supplementary-material-1.xls')
data = pd.read_excel(data_path)