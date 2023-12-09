import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import glob


def plot_experiment(experimentParameter):
    path = "./*.csv"

    # plt.title('Result')
    plt.xlabel('Number of Episodes')
    plt.ylabel('$κ_{100}$')

    plots = []
    plotColor = []
    fnames = glob.glob(path)
    fnames.sort()
    for fname in fnames:
        df = pd.read_csv(fname[2:])  # ignoring ./

        p = plt.plot(df)

        plots.append(p[0])
        # CHANGE the range of the file name to only specify the variable number in your file name
        # E.g. if fileName is Alpha5e-05.csv, then str(fname[7:12]) will return 5e-05
        plotColor.append("$"+experimentParameter+"=" +
                         str(fname[3:]).rstrip('.csv')+"$")

    plt.legend(plots, plotColor)
    plt.show()


plot_experiment('Model')
