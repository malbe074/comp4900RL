import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import glob

def plot_experiment(experimentParameter, experimentVariable):
    path = "./*.csv"

    plt.title('Result')
    plt.xlabel('Every 10th Episode')
    plt.ylabel('Average Duration')

    plots=[]
    plotColor=[]
    for fname in glob.glob(path):
        df = pd.read_csv(fname[2:]) #ignoring ./

        p = plt.plot(df)

        plots.append(p[0])
        #instead of LR, CHANGE the variable name to the parameter you are assigned to experiement (epsilon, batch, reward, discount factor, Q-network weight, hidden layers, space)
        plotColor.append("$"+experimentParameter+"="+str(experimentVariable)+"$")

    plt.legend(plots,plotColor)
    plt.show()