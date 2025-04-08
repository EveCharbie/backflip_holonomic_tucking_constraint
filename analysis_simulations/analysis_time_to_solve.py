import os

import numpy as np
import pickle

common_path = "../results/backflip_Vpost_submission_collision_feb25/"
folder_HTC = common_path + "htc/"
folder_KTC = common_path + "ktc/"
folder_FREE = common_path + "ntc/"

time_to_solve = np.zeros((20, 3))
time_to_solve[:, :] = np.nan
for config, folder in enumerate([folder_KTC, folder_HTC, folder_FREE]):
    i_file = 0
    for name_file in os.listdir(folder):
        if name_file.endswith("_CVG.pkl"):
            data = pickle.load(open(folder + name_file, "rb"))
            time_to_solve[i_file, config] = data["real_time_to_optimize"]
            i_file += 1

print(time_to_solve)
print(np.nanmean(time_to_solve, axis=0))
print(np.nanstd(time_to_solve, axis=0))

# print nicely
print("Mean time to solve")
print(f"KTC: {np.nanmean(time_to_solve[:, 0])} ± {np.nanstd(time_to_solve[:, 0])} s")
print(f"HTC: {np.nanmean(time_to_solve[:, 1])} ± {np.nanstd(time_to_solve[:, 1])} s")
print(f"NTC: {np.nanmean(time_to_solve[:, 2])} ± {np.nanstd(time_to_solve[:, 2])} s")

# in minutes, round to 2 decimals
print("Mean time to solve")
print(f"KTC: {np.nanmean(time_to_solve[:, 0]) / 60:.2f} ± {np.nanstd(time_to_solve[:, 0]) / 60:.2f} min")
print(f"HTC: {np.nanmean(time_to_solve[:, 1]) / 60:.2f} ± {np.nanstd(time_to_solve[:, 1]) / 60:.2f} min")
print(f"NTC: {np.nanmean(time_to_solve[:, 2]) / 60:.2f} ± {np.nanstd(time_to_solve[:, 2]) / 60:.2f} min")
