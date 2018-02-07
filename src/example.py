import numpy as np
import matplotlib.pyplot as plt

from Kmeancluster import kmeanclusterer
from GaussianMixture import GMclusterer
DATAFILE='sample_input/train.txt'
X=np.loadtxt(DATAFILE)
part=2
if part==1:
    Kmean=kmeanclusterer(X,5)

    Kmean.plot_state(0)
    for i in range(1,11):
        Kmean.runiteration()
        Kmean.plot_state(i)
    plt.show()
if part==2:
    G=GMclusterer(X,5)
    for i in range(1, 11):
        G.runiteration()
print("finished")


