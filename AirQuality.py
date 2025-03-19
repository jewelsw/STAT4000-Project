import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',', usecols=range(15))

# Print first 5 rows
print(data.head())
