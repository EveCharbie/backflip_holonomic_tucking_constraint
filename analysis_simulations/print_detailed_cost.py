from contextlib import redirect_stdout
import os

import numpy as np
import pickle

from examples.constants import PHASE_TIME, N_SHOOTING
from examples.somersault_taudot import prepare_ocp as prepare_ocp_free
from examples.somersault_htc_taudot import prepare_ocp as prepare_ocp_HTC
from examples.somersault_ktc_taudot import prepare_ocp as prepare_ocp_KTC
from src.constants import PATH_MODEL, PATH_MODEL_1_CONTACT

biorbd_model_path = (PATH_MODEL_1_CONTACT, PATH_MODEL, PATH_MODEL, PATH_MODEL, PATH_MODEL_1_CONTACT)
phase_time = PHASE_TIME
n_shooting = N_SHOOTING

common_path = "../results/backflip_Vpost_submission_collision_feb25/"
folder_HTC = common_path + "htc/"
folder_KTC = common_path + "ktc/"
folder_FREE = common_path + "ntc/"

file_idx = []
for config, (common_path, str_suffix) in enumerate(zip([folder_KTC, folder_FREE, folder_HTC], ["KTC", "NTC", "HTC"])):
    n_files = len([name for name in os.listdir(common_path) if name.endswith("_CVG.pkl")])
    smallest_idx, smallest_value = 0, np.inf
    for i in range(0, n_files):
        data = pickle.load(open(common_path + f"/sol_{i}_CVG.pkl", "rb"))
        if data["cost"] < smallest_value:
            smallest_value = data["cost"]
            smallest_idx = i
            print(f"New smallest value of {str_suffix} : {smallest_value} at index {smallest_idx}")
    file_idx.append(smallest_idx)

zipped = zip(
    [folder_KTC, folder_FREE, folder_HTC],
    ["KTC", "NTC", "HTC"],
    [prepare_ocp_KTC, prepare_ocp_free, prepare_ocp_HTC],
    file_idx,
)

data = pickle.load(open(folder_KTC + f"/sol_{file_idx[0]}_CVG.pkl", "rb"))
sol = pickle.load(open(folder_KTC + f"/sol_{file_idx[0]}_CVG_sol.pkl", "rb"))
sol.ocp = prepare_ocp_KTC(biorbd_model_path, phase_time, n_shooting, WITH_MULTI_START=False)

sol.print_cost()
with open(f"best_objectives_and_constraints_KTC.txt", "w") as f:
    with redirect_stdout(f):
        sol.print_cost()


data = pickle.load(open(folder_FREE + f"/sol_{file_idx[1]}_CVG.pkl", "rb"))
sol = pickle.load(open(folder_FREE + f"/sol_{file_idx[1]}_CVG_sol.pkl", "rb"))
sol.ocp = prepare_ocp_free(biorbd_model_path, data["phase_time"], n_shooting, WITH_MULTI_START=False)

with open(f"best_objectives_and_constraints_NTC.txt", "w") as f:
    with redirect_stdout(f):
        sol.print_cost()

data = pickle.load(open(folder_HTC + f"/sol_{file_idx[2]}_CVG.pkl", "rb"))
sol = pickle.load(open(folder_HTC + f"/sol_{file_idx[2]}_CVG_sol.pkl", "rb"))
sol.ocp = prepare_ocp_HTC(biorbd_model_path, data["phase_time"], n_shooting, WITH_MULTI_START=False)

with open(f"best_objectives_and_constraints_HTC.txt", "w") as f:
    with redirect_stdout(f):
        sol.print_cost()
