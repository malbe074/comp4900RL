import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import glob

def plot_experiment(experimentParameter):
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
        #CHANGE the range of the file name to only specify the variable number in your file name
        # E.g. if fileName is Alpha5e-05.csv, then str(fname[7:12]) will return 5e-05
        plotColor.append("$"+experimentParameter+"="+str(fname[7:12])+"$")

    plt.legend(plots,plotColor)
    plt.show()