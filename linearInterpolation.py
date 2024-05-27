import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def readFile():
    Water = pd.read_csv('waterquality.csv')
    Water = Water.interpolate("linear")

    Water['lat'] = np.random.uniform(-90, 90, Water.shape[0])
    Water['lng'] = np.random.uniform(-180, 180, Water.shape[0])

    return Water

def createScatterMatrix(df):
    scatter_matrix(df, alpha=0.8, figsize=(12, 12), diagonal='kde')
    plt.show()

def main():
    df = readFile()

    print(df)

    #createScatterMatrix(df)
    

if __name__ == "__main__":
    main()