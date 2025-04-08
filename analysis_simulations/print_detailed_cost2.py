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

# folder = "with_noise_same_computer/"
common_path = "../results/backflip_Vpost_submission_collision_feb25/"
folder_HTC = common_path + "htc/"
folder_KTC = common_path + "ktc/"
folder_NTC = common_path + "htc/"

file_name = []
for config, (common_path, str_suffix) in enumerate(zip([folder_KTC, folder_NTC, folder_HTC], ["KTC", "NTC", "HTC"])):
    files = [name for name in os.listdir(common_path) if name.endswith("_CVG.pkl")]
    smallest_name, smallest_value = "0", np.inf
    for file in files:

        data = pickle.load(open(common_path + file, "rb"))

        if data["cost"] < smallest_value:
            smallest_value = data["cost"]
            smallest_name= file
            print(f"New smallest value of {str_suffix} : {smallest_value} in file {smallest_name}")
    file_name.append(smallest_name)

zipped = zip(
    [folder_KTC, folder_NTC, folder_HTC],
    ["KTC", "NTC", "HTC"],
    [prepare_ocp_KTC, prepare_ocp_free, prepare_ocp_HTC],
    file_name,
)

data = pickle.load(open(folder_KTC + file_name[0], "rb"))
sol = pickle.load(open(folder_KTC + file_name[0][0:-4] + "_sol.pkl", "rb"))
sol.ocp = prepare_ocp_KTC(biorbd_model_path, data["phase_time"], n_shooting, WITH_MULTI_START=False)

sol.print_cost()
with open(f"best_objectives_and_constraints_KTC.txt", "w") as f:
    with redirect_stdout(f):
        sol.print_cost()


data = pickle.load(open(folder_NTC + file_name[1], "rb"))
sol = pickle.load(open(folder_NTC + file_name[1][0:-4] + "_sol.pkl", "rb"))
sol.ocp = prepare_ocp_free(biorbd_model_path, data["phase_time"], n_shooting, WITH_MULTI_START=False)

with open(f"best_objectives_and_constraints_NTC.txt", "w") as f:
    with redirect_stdout(f):
        sol.print_cost()

data = pickle.load(open(folder_HTC + file_name[2], "rb"))
sol = pickle.load(open(folder_HTC + file_name[2][0:-4] + "_sol.pkl", "rb"))
sol.ocp = prepare_ocp_HTC(biorbd_model_path, data["phase_time"], n_shooting, WITH_MULTI_START=False)

with open(f"best_objectives_and_constraints_HTC.txt", "w") as f:
    with redirect_stdout(f):
        sol.print_cost()
