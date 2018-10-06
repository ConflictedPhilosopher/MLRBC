import MLRBC
import random

NumberOfExperiments = 1
seedNum = 11
random.seed(seedNum)

MAIN_RESULTS_PATH = "MLRBC_Run_Results"
pathlib.Path(os.path.join(MAIN_RESULTS_PATH)).mkdir(parents=True, exist_ok=True)

for exp in range(NumberOfExperiments):
    print('Number of experiment: ', exp + 1)
    MLRBC.MLRBC(exp)


