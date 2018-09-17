import MLRBC
import random

NumberOfExperiments = 1
seedNum = 11
random.seed(seedNum)

for exp in range(NumberOfExperiments):
    print('Number of experiment: ', exp + 1)
    MLRBC.MLRBC(exp)


