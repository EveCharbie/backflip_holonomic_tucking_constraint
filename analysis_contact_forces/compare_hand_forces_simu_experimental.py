"""
Main code calling scipt python create_cylinder_insole functions to analyze pressure inserts data.
WARNING: the right foot insole was placed on the left tibia and inversely.

forces_insoles is expressed in the local frame of the insole (side, front), so we keep only the front component, which is the force perpendicular to the tibia in the plane.
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd

from matplotlib import rcParams

rcParams["font.family"] = "DeJavu Serif"  # Use serif font
rcParams["font.serif"] = ["Times New Roman"]  # Specify Times New Roman or Times


def force_treatment(forces_insoles, time, first_peak_time):

    idx_1s = np.where(time_insoles > 4.5)[0][0]
    idx_6s = np.where(time_insoles > 7)[0][0]

    # Remove the base force value
    forces_insoles_zero = np.nanmean(forces_insoles[idx_1s:idx_6s, 1])
    force = forces_insoles[:, 1] - forces_insoles_zero

    first_peak_time_idx = np.argmin(np.abs(time - first_peak_time))
    begining = first_peak_time_idx - 4
    end = first_peak_time_idx + 96

    return force[begining:end], time[begining:end] - time[begining] - 0.008


# Get simulation forces
sol_CL = "../results/backflip_Vpost_submission_collision_feb25/htc/sol_3_CVG.pkl"
data_CL = pd.read_pickle(sol_CL)
lambdas = data_CL["lambda"]
format_graph = "svg"

fig, ax = plt.subplots(1, 1)

force_results_path = "EmCo_insoles_forces/"
# Time of the first steep peak identified by hand
file_names = {
    "salto_control_pre_1": 16.3979823,
    "salto_control_pre_2": 12.9099990,
    "salto_control_pre_3": 11.634994,
    "salto_control_post_1": 12.482998,
    "salto_control_post_2": 13.61491,
    "salto_control_post_3": 13.2349997,
}

times = np.zeros((100, 6))
forces = np.zeros((100, 6))
for file in file_names.keys():
    with open(force_results_path + file + "_L.pkl", "rb") as f:
        data_L = pickle.load(f)

    if not isinstance(data_L, dict):
        raise FileExistsError(
            "The loaded file is not a dictionary. " "It needs to be update @charbie, to put the version with the time"
        )

    forces_insoles = data_L["force_data"] * 2
    time_insoles = data_L["time"]

    force, time = force_treatment(-forces_insoles, time_insoles, file_names[file])
    ax.plot(time, force, label=file, color="k", alpha=0.1)

    times[:, list(file_names.keys()).index(file)] = time
    forces[:, list(file_names.keys()).index(file)] = force

# Plot the mean
ax.plot(time, np.mean(forces, axis=1), label="Mean", color="k", linewidth=2)
# ax.fill_between(time, np.min(forces, axis=1), np.max(forces, axis=1), color='k', alpha=0.1)
ax.plot([time[0], time[-1]], [0, 0], "-k", linewidth=0.5)

# Plot the simulation results
time_tuck = data_CL["time"][2] - data_CL["time"][2][0]
ax.plot(time_tuck, lambdas[0], color="r", label=["Normal force"])
ax.plot(time_tuck, lambdas[1], color="g", label=["Shear force"])

ax.legend(bbox_to_anchor=(1.0, 1), loc="upper left")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Force on the tibia [N]")
plt.subplots_adjust(right=0.7)
plt.savefig("hand_leg_forces_experimental_vs_simulations." + format_graph, format=format_graph)
plt.show()

print("Max lambda norm : ", np.max(np.linalg.norm(lambdas, axis=0)))
