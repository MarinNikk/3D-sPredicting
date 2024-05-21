import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def readFile():
    Water = pd.read_csv('waterquality.csv').dropna().tail(200)
    #Water = Water.set_index('Date')
    #print(Water.loc[:, ['pH']])


    salinity = Water['Salinity (ppt)']
    dissOx = Water['DissolvedOxygen (mg/L)']
    pH = Water['pH']
    secchiDepth = Water['SecchiDepth (m)']
    waterDepth = Water['WaterDepth (m)']
    waterTemperature = Water['WaterTemp (C)']
    airTemperature = Water['AirTemp (C)']

    return salinity, dissOx, pH, secchiDepth, waterDepth, waterTemperature, airTemperature

def readNumpy(salinity, dissOx, pH, secchiDepth, waterDepth, waterTemperature, airTemperature):
    coefficient = np.polyfit(pH, waterTemperature, 1)
    slope = coefficient[0]
    intercept = coefficient[1]
    y = slope * pH + intercept

    calcMSE(y, waterTemperature)

    plt.plot(pH, y, color="red")
    plt.scatter(pH, waterTemperature, color='blue')
    plt.xlabel("PH")
    plt.ylabel("Water Temp (C)")
    plt.grid(True)
    plt.show()

def readScipy(salinity, dissOx, pH, secchiDepth, waterDepth, waterTemperature, airTemperature):
    slope, intercept, r, p, std_err = stats.linregress(pH, waterTemperature)

    x = pH.tolist()

    def myfun(x):
        return slope * x + intercept
    
    y = list(map(myfun, x))

    calcMSE(y, waterTemperature)

    plt.scatter(pH, waterTemperature, color='blue')
    plt.plot(pH, y, color="red")
    plt.xlabel("PH")
    plt.ylabel("Water Temp (C)")
    plt.grid(True)
    plt.show()

def readNikola(salinity, dissOx, pH, secchiDepth, waterDepth, waterTemperature, airTemperature):

    ph = pH.tolist()
    temp = waterTemperature.tolist()

    def linregress(b0, b1, ph, temp, l):
        n = 200
        b0_final = 0
        b1_final = 0

        for i in range(n):
            x = ph[i]
            y = temp[i]

            b0_final += (-2 / n) * (y - b0 - b1 * x)
            b1_final += (-2 / n) * ((y - b0 - b1 * x) * x)

        b0_final = b0 - l * b0_final
        b1_final = b1 - l * b1_final

        return b0_final, b1_final
    
    b0 = 0
    b1 = 0

    l = 0.0001

    counter = 1000

    for i in range(counter):
        b0,b1 = linregress(b0, b1, ph, temp, l)

    y = b1 * pH + b0

    calcMSE(y, waterTemperature)
    
    plt.scatter(pH, waterTemperature, color='blue')
    plt.plot(pH, y, color="red")
    plt.xlabel("PH")
    plt.ylabel("Water Temp (C)")
    plt.grid(True)
    plt.show()

def calcMSE(y, waterTemperature):
    MSE = np.mean((y-waterTemperature)**2)

    print(MSE)



def main():
    salinity, dissOx, pH, secchiDepth, waterDepth, waterTemperature, airTemperature = readFile()

    #readNumpy(salinity, dissOx, pH, secchiDepth, waterDepth, waterTemperature, airTemperature)

    #readScipy(salinity, dissOx, pH, secchiDepth, waterDepth, waterTemperature, airTemperature)

    readNikola(salinity, dissOx, pH, secchiDepth, waterDepth, waterTemperature, airTemperature)

if __name__ == "__main__":
    main()