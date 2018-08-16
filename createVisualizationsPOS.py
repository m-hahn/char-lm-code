import os
basePath = "/checkpoint/mhahn/pos/"

import sys

language = sys.argv[1]



import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np



#for language in ["Japanese", "Sesotho", "Indonesian"]:
if True:
   prefix = "char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-plurals-2.py_wiki-german-nospaces-bugfix-checkpoints_CHECKPOINT"
   files = [x for x in os.listdir(basePath) if language in x]
   print(basePath+"/"+prefix)
   data = {"LM" : [], "Baseline" : []}
   for filename in files:
       with open(basePath+"/"+filename, "r") as inFile:
           dataNew = inFile.read().strip().split("\n")
           trainSize = int(dataNew[1])
           model = "LM" if "bptt" in dataNew[2] else "Baseline"
           result = float(dataNew[3])
           data[model].append((trainSize, result))
   print(data)
   for group, datapoints in data.items():
          datapoints = sorted(datapoints, key=lambda x:x[0])
          plt.plot([x[0] for x in datapoints], [x[1] for x in datapoints], label=group)
   plt.legend()
   plt.show()
   plt.savefig(f"/checkpoint/mhahn/figures/{language}_pos_nouns_verbs.png")
   plt.close()

