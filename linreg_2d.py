import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# waterDepth, waterTemperature

def readFile():
    Water = pd.read_csv('waterquality.csv').dropna().tail(200)

    salinity = Water['Salinity (ppt)']
    dissOx = Water['DissolvedOxygen (mg/L)']
    pH = Water['pH']
    secchiDepth = Water['SecchiDepth (m)']
    waterDepth = Water['WaterDepth (m)']
    waterTemperature = Water['WaterTemp (C)']
    airTemperature = Water['AirTemp (C)']

    print(Water.loc[:,['AirTemp (C)', 'WaterTemp (C)']])

    return salinity, dissOx, pH, secchiDepth, waterDepth, waterTemperature, airTemperature

def readNumpy(x, y):
    coefficient = np.polyfit(x, y, 1)
    slope = coefficient[0]
    intercept = coefficient[1]
    model = slope * x + intercept

    calcMSE(model, y)

    
    plt.scatter(x, y, color='blue')
    plt.xlabel("X[AirTemp]")
    plt.ylabel("Y[WaterTemp]")
    plt.grid(True)

    PRED_X = 21
    plt.scatter(PRED_X, pred(slope, intercept, PRED_X), color="green")

    plt.plot(x, model, color="red")

    plt.show()

def calcMSE(y, waterTemperature):
    MSE = np.mean((y-waterTemperature)**2)
    print(MSE)

def pred(slope, intercept, x):
    print('pred:', slope * x + intercept)
    return slope * x + intercept
    

def main():
    salinity, dissOx, pH, secchiDepth, waterDepth, waterTemperature, airTemperature = readFile()
    readNumpy(airTemperature, waterTemperature)

if __name__ == "__main__":
    main()