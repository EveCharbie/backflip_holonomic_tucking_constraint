import os
import numpy as np
import pandas as pd
import biorbd
import matplotlib.pyplot as plt
import pickle

from matplotlib import rcParams

from examples.somersault_taudot import prepare_ocp as prepare_ocp_without
from examples.constants import PHASE_TIME, N_SHOOTING
from src.actuators import Joint, actuator_function
from src.actuator_constants import ACTUATORS
from src.constants import (
    PATH_MODEL_1_CONTACT,
    PATH_MODEL,
)

rcParams["font.family"] = "DeJavu Serif"  # Use serif font
rcParams["font.serif"] = ["Times New Roman"]  # Specify Times New Roman or Times


def get_created_data_from_pickle(file: str):
    """
    This code is used to open a pickle document and exploit its data_CL.

    Parameters
    ----------
    file: path of the pickle document

    Returns
    -------
    data_CL: All the data_CL of the pickle document
    """
    with open(file, "rb") as f:
        while True:
            try:
                data_tmp = pickle.load(f)
            except:
                break

    return data_tmp


def plot_vertical_time_lines(
    time_end_phase_CL, time_end_phase_without, time_end_phase_free, ax, color, linestyle, linewidth
):
    color_CL = "tab:orange" if color is None else color
    color_without = "tab:blue" if color is None else color
    color_free = "tab:green" if color is None else color
    linewidth = 0.7 if linewidth is None else linewidth
    ax.axvline(time_end_phase_free, color=color_free, linestyle=linestyle, linewidth=linewidth)
    ax.axvline(time_end_phase_without, color=color_without, linestyle=linestyle, linewidth=linewidth)
    ax.axvline(time_end_phase_CL, color=color_CL, linestyle=linestyle, linewidth=linewidth)
    return


def plot_all_lines(time_end_phase_CL, time_end_phase_without, time_end_phase_free, ax):
    plot_vertical_time_lines(
        time_end_phase_CL[0],
        time_end_phase_without[0],
        time_end_phase_free[0],
        ax,
        color="k",
        linestyle="-",
        linewidth=0.5,
    )
    plot_vertical_time_lines(
        time_end_phase_CL[1],
        time_end_phase_without[1],
        time_end_phase_free[1],
        ax,
        color=None,
        linestyle=phase_delimiter[1],
        linewidth=None,
    )
    plot_vertical_time_lines(
        time_end_phase_CL[2],
        time_end_phase_without[2],
        time_end_phase_free[2],
        ax,
        color=None,
        linestyle=phase_delimiter[2],
        linewidth=None,
    )
    plot_vertical_time_lines(
        time_end_phase_CL[3],
        time_end_phase_without[3],
        time_end_phase_free[3],
        ax,
        color=None,
        linestyle=phase_delimiter[3],
        linewidth=None,
    )
    return


def adjust_q_with_full_floating_base(q: np.ndarray) -> np.ndarray:
    """
    Adjust the q vector to take into account the full floating base

    Parameters
    ----------
    q: np.ndarray
        The q vector to adjust

    Returns
    -------
    The adjusted q vector
    """
    q_adjusted = np.zeros((q.shape[0] + 3, q.shape[1]))
    q_adjusted[1:4, :] = q[0:3, :]
    q_adjusted[6:, :] = q[3:, :]
    return q_adjusted


# Solution with and without holonomic constraints
common_path = "../results/backflip_Vpost_submission_collision_feb25/"
path_without = common_path + "ktc/"
path_CL = common_path + "htc/"
path_free = common_path + "ntc/"

path_model = "../models/Model2D_7Dof_2C_5M_CL_V3.bioMod"
model = biorbd.Model(path_model)
path_model_adjusted = "../models/Model2D_7Dof_3C_5M_CL_V3_V3D.bioMod"
model_adjusted = biorbd.Model(path_model_adjusted)

CONSIDER_ONLY_CONVERGED = True
if CONSIDER_ONLY_CONVERGED:
    end_file = "CVG.pkl"
else:
    end_file = ".pkl"

biorbd_model_path = (PATH_MODEL_1_CONTACT, PATH_MODEL, PATH_MODEL, PATH_MODEL, PATH_MODEL_1_CONTACT)
n_shooting = N_SHOOTING
n_nodes = tuple(value + 1 for value in n_shooting)

min_cost_without = np.inf
for file in os.listdir(path_without):
    if file.endswith(end_file):
        data = pd.read_pickle(path_without + file)
        if data["cost"] < min_cost_without:
            min_cost_without = data["cost"]
            sol_without = path_without + file
print("Min cost without: ", min_cost_without)

min_cost_CL = np.inf
for file in os.listdir(path_CL):
    if file.endswith(end_file):
        data = pd.read_pickle(path_CL + file)
        if data["cost"] < min_cost_CL:
            min_cost_CL = data["cost"]
            sol_CL = path_CL + file
print("Min cost CL: ", min_cost_CL)

min_cost_free = np.inf
for file in os.listdir(path_free):
    if file.endswith(end_file):
        data = pd.read_pickle(path_free + file)
        if data["cost"] < min_cost_free:
            min_cost_free = data["cost"]
            sol_free = path_free + file
print("Min cost free: ", min_cost_free)

data_CL = pd.read_pickle(sol_CL)
data_without = pd.read_pickle(sol_without)
data_free = pd.read_pickle(sol_free)

print("Computational time without: ", data_without["real_time_to_optimize"] / 60, "min")
print("Computational time CL: ", data_CL["real_time_to_optimize"] / 60, "min")
print("Computational time free: ", data_free["real_time_to_optimize"] / 60, "min")

print("Residual forces at take-off without: ", np.linalg.norm(data_without["contact_forces"][0][:, -1]), "N")
print("Residual forces at take-off CL: ", np.linalg.norm(data_CL["contact_forces"][0][:, -1]), "N")
print("Residual forces at take-off free: ", np.linalg.norm(data_free["contact_forces"][0][:, -1]), "N")

aerial_node_start = n_nodes[0]
tuck_node_start = n_nodes[0] + n_nodes[1]
tuck_node_end = tuck_node_start + n_nodes[2]
aerial_node_end = n_nodes[0] + n_nodes[1] + n_nodes[2] + n_nodes[3]

PLOT_TAU_FLAG = True
PLOT_INERTIA_FLAG = True
PLOT_ENERGY_FLAG = True
# format_graph = "png"
# format_graph = "pdf"
format_graph = "svg"


phase_delimiter = ["-", "--", ":", "-.", "-"]

plt.figure()
plt.plot(0, 1, "-k", label="end phase 1 - start phase 2")
plt.plot(0, 1, "--k", label="end phase 2 - start phase 3")
plt.plot(0, 1, ":k", label="end phase 3 - start phase 4")
plt.plot(0, 1, "-.k", label="end phase 4 - start phase 5")
plt.legend()
plt.savefig("delimiters" + "." + format_graph, format=format_graph, dpi=300)
plt.close()

dof_names = [
    "Pelvis \n(Translation Y)",
    "Pelvis \n(Translation Z)",
    "Pelvis (Rotation X)",
    "Shoulder",
    "Elbow",
    "Hip",
    "Knee",
    "Ankle",
]
dof_names_tau = ["Shoulder", "Elbow", "Hip", "Knee", "Ankle"]

# Data to plot ----------------------------------------------

# Solution with closed-loop constraints
q_CL_rad = data_CL["q_all"][:, :]
q_CL_rad_original = data_CL["q_all"][:, :]
q_CL_rad[6, :] = q_CL_rad[6, :] * -1
q_CL_deg = np.vstack([q_CL_rad[0:2, :], q_CL_rad[2:, :] * 180 / np.pi])
qdot_CL_rad = data_CL["qdot_all"]
qdot_CL_rad[6, :] = qdot_CL_rad[6, :] * -1
qdot_CL_deg = np.vstack([qdot_CL_rad[0:2, :], qdot_CL_rad[2:, :] * 180 / np.pi])
tau_CL = data_CL["tau_all"]
taudot_CL = data_CL["taudot_all"] if "taudot_all" in data_CL.keys() else np.hstack(data_CL["taudot"])

time_CL = data_CL["time"]
time_end_phase_CL = []
time_end_phase_tau_CL = []
for i in range(len(time_CL)):
    time_end_phase_CL.append(time_CL[i][-1])
    time_end_phase_tau_CL.append(time_CL[i][-2])
time_vector_CL = (
    np.vstack(data_CL["time"]).reshape(
        -1,
    )
    - time_CL[0][-1]
)
time_end_phase_CL = np.array(time_end_phase_CL) - time_CL[0][-1]
time_control_CL = (
    np.vstack([arr[:-1, :] for arr in time_CL]).reshape(
        -1,
    )
    - time_CL[0][-1]
)
time_tuck_CL = time_vector_CL[tuck_node_start:tuck_node_end]

# Solution without closed-loop constraints
q_without_rad = data_without["q_all"][:, :]
q_without_rad_original = data_without["q_all"][:, :]
q_without_rad[6, :] = q_without_rad[6, :] * -1
q_without_deg = np.vstack([q_without_rad[0:2, :], q_without_rad[2:, :] * 180 / np.pi])
qdot_without_rad = data_without["qdot_all"]
qdot_without_rad[6, :] = qdot_without_rad[6, :] * -1
qdot_without_deg = np.vstack([qdot_without_rad[0:2, :], qdot_without_rad[2:, :] * 180 / np.pi])
tau_without = data_without["tau_all"]
taudot_without = (
    data_without["taudot_all"] if "taudot_all" in data_without.keys() else np.hstack(data_without["taudot"])
)

time_without = data_without["time"]
time_end_phase_without = []
time_end_phase_tau_without = []
for i in range(len(time_without)):
    time_end_phase_without.append(time_without[i][-1])
    time_end_phase_tau_without.append(time_without[i][-2])
time_vector_without = (
    np.vstack(data_without["time"]).reshape(
        -1,
    )
    - time_without[0][-1]
)
time_end_phase_without = np.array(time_end_phase_without) - time_without[0][-1]
time_control_without = (
    np.vstack([arr[:-1, :] for arr in time_without]).reshape(
        -1,
    )
    - time_without[0][-1]
)
time_tuck_without = time_vector_without[tuck_node_start:tuck_node_end]

# Solution without any tucking constraints
q_free_rad = data_free["q_all"][:, :]
q_free_rad_original = data_free["q_all"][:, :]
q_free_rad[6, :] = q_free_rad[6, :] * -1
q_free_deg = np.vstack([q_free_rad[0:2, :], q_free_rad[2:, :] * 180 / np.pi])
qdot_free_rad = data_free["qdot_all"]
qdot_free_rad[6, :] = qdot_free_rad[6, :] * -1
qdot_free_deg = np.vstack([qdot_free_rad[0:2, :], qdot_free_rad[2:, :] * 180 / np.pi])
tau_free = data_free["tau_all"]
taudot_free = data_free["taudot_all"] if "taudot_all" in data_free.keys() else np.hstack(data_free["taudot"])

time_free = data_free["time"]
time_end_phase_free = []
time_end_phase_tau_free = []
for i in range(len(time_free)):
    time_end_phase_free.append(time_free[i][-1])
    time_end_phase_tau_free.append(time_free[i][-2])
time_vector_free = (
    np.vstack(data_free["time"]).reshape(
        -1,
    )
    - time_free[0][-1]
)
time_end_phase_free = np.array(time_end_phase_free) - time_free[0][-1]
time_control_free = (
    np.vstack([arr[:-1, :] for arr in time_free]).reshape(
        -1,
    )
    - time_free[0][-1]
)
time_tuck_free = time_vector_free[tuck_node_start:tuck_node_end]

time_max_graph = max(time_vector_without[-1], time_vector_CL[-1], time_vector_free[-1])
time_min_graph = min(time_vector_without[0], time_vector_CL[0], time_vector_free[0])

# Duree phase solutions with and without holonomic constraints
time_phase_CL = []
time_phase_without = []
time_phase_free = []

for i in range(len(data_CL["time"])):
    time_phase_CL.append(data_CL["time"][i][-1] - data_CL["time"][i][0])
    time_phase_without.append(data_without["time"][i][-1] - data_without["time"][i][0])
    time_phase_free.append(data_free["time"][i][-1] - data_free["time"][i][0])

print(
    "*** Phase_time *** \nCL :",
    time_phase_CL,
    np.sum(time_phase_CL[1:4]),
    "\nwithout : ",
    time_phase_without,
    np.sum(time_phase_without[1:4]),
    "\nfree : ",
    time_phase_free,
    np.sum(time_phase_free[1:4]),
)
print(
    "Total CL :",
    np.sum(time_phase_CL),
    "\nTotal without : ",
    np.sum(time_phase_without),
    "\nTotal free : ",
    np.sum(time_phase_free),
)


plt.figure()
plt.plot(data_CL["lambda"][0, :], "b")
plt.plot(data_CL["lambda"][1, :], "r")
plt.savefig("lambda_tempo.png")
# plt.show()
print("Maximal hand-on-shank contact force (norm): ", np.max(np.linalg.norm(data_CL["lambda"], axis=0)), "N")


# Graphique
if PLOT_TAU_FLAG:

    # Min and max bounds on q
    min_bound_q = np.ones((q_without_rad.shape[0], time_vector_without.shape[0])) * -1000
    max_bound_q = np.ones((q_without_rad.shape[0], time_vector_without.shape[0])) * 1000
    for i_phase in range(5):
        idx_beginning = sum(n_nodes[:i_phase])
        idx_end = sum(n_nodes[: i_phase + 1]) - 1
        for i_dof in range(8):
            if data_without["min_bounds_q"][i_phase][i_dof, 0] > min_bound_q[i_dof, idx_beginning]:
                min_bound_q[i_dof, idx_beginning] = data_without["min_bounds_q"][i_phase][i_dof, 0]
            if data_without["max_bounds_q"][i_phase][i_dof, 0] < max_bound_q[i_dof, idx_beginning]:
                max_bound_q[i_dof, idx_beginning] = data_without["max_bounds_q"][i_phase][i_dof, 0]
        for i_node in range(idx_beginning + 1, idx_end):
            min_bound_q[:, i_node] = data_without["min_bounds_q"][i_phase][:, 1]
            max_bound_q[:, i_node] = data_without["max_bounds_q"][i_phase][:, 1]
        min_bound_q[:, idx_end] = data_without["min_bounds_q"][i_phase][:, 2]
        max_bound_q[:, idx_end] = data_without["max_bounds_q"][i_phase][:, 2]
    min_bound_q *= 180 / np.pi
    max_bound_q *= 180 / np.pi
    tempo_min_6 = min_bound_q[6, :].copy()
    min_bound_q[6, :] = -max_bound_q[6, :]
    max_bound_q[6, :] = -tempo_min_6

    # Figure q
    fig, axs = plt.subplots(3, 3, figsize=(10, 6))
    num_col = 0
    num_line = 0
    y_max_1 = np.max([abs(q_CL_deg[0:2, :]), abs(q_without_deg[0:2, :]), abs(q_free_deg[0:2, :])])
    y_max_2 = np.max([abs(q_CL_deg[2:5, :]), abs(q_without_deg[2:5, :]), abs(q_free_deg[2:5, :])])
    y_max_3 = np.max([abs(q_CL_deg[5:, :]), abs(q_without_deg[5:, :]), abs(q_free_deg[5:, :])])
    for i_dof in range(q_CL_deg.shape[0]):
        axs[num_line, num_col].plot(np.array([time_min_graph, time_max_graph]), np.array([0, 0]), "-k", linewidth=0.5)

        plot_all_lines(time_end_phase_CL, time_end_phase_without, time_end_phase_free, axs[num_line, num_col])

        axs[num_line, num_col].fill_between(
            time_vector_free,
            max_bound_q[i_dof, :],
            np.ones(max_bound_q[i_dof, :].shape) * 1000,
            color="tab:green",
            alpha=0.1,
            step="pre",
            linewidth=1,
        )
        axs[num_line, num_col].fill_between(
            time_vector_free,
            np.ones(min_bound_q[i_dof, :].shape) * -1000,
            min_bound_q[i_dof, :],
            color="tab:green",
            alpha=0.1,
            step="pre",
            linewidth=1,
        )
        axs[num_line, num_col].fill_between(
            time_vector_without,
            max_bound_q[i_dof, :],
            np.ones(max_bound_q[i_dof, :].shape) * 1000,
            color="tab:blue",
            alpha=0.1,
            step="pre",
            linewidth=1,
        )
        axs[num_line, num_col].fill_between(
            time_vector_without,
            np.ones(min_bound_q[i_dof, :].shape) * -1000,
            min_bound_q[i_dof, :],
            color="tab:blue",
            alpha=0.1,
            step="pre",
            linewidth=1,
        )
        axs[num_line, num_col].fill_between(
            time_vector_CL,
            max_bound_q[i_dof, :],
            np.ones(max_bound_q[i_dof, :].shape) * 1000,
            color="tab:orange",
            alpha=0.1,
            step="pre",
            linewidth=1,
        )
        axs[num_line, num_col].fill_between(
            time_vector_CL,
            np.ones(min_bound_q[i_dof, :].shape) * -1000,
            min_bound_q[i_dof, :],
            color="tab:orange",
            alpha=0.1,
            step="pre",
            linewidth=1,
        )

        axs[num_line, num_col].plot(
            time_vector_free,
            q_free_deg[i_dof, :],
            color="tab:green",
            label="No tucking constraint",
            alpha=0.75,
            linewidth=1,
        )
        axs[num_line, num_col].plot(
            time_tuck_free,
            q_free_deg[i_dof, tuck_node_start:tuck_node_end],
            color="tab:green",
            alpha=0.75,
            linewidth=3,
        )
        axs[num_line, num_col].plot(
            time_vector_without,
            q_without_deg[i_dof, :],
            color="tab:blue",
            label="Kinematic tucking constraints",
            alpha=0.75,
            linewidth=1,
        )
        axs[num_line, num_col].plot(
            time_tuck_without,
            q_without_deg[i_dof, tuck_node_start:tuck_node_end],
            color="tab:blue",
            alpha=0.75,
            linewidth=3,
        )
        axs[num_line, num_col].plot(
            time_vector_CL,
            q_CL_deg[i_dof, :],
            color="tab:orange",
            label="Holonomic tucking constraints",
            alpha=0.75,
            linewidth=1,
        )
        axs[num_line, num_col].plot(
            time_tuck_CL,
            q_CL_deg[i_dof, tuck_node_start:tuck_node_end],
            color="tab:orange",
            alpha=0.75,
            linewidth=3,
        )

        axs[num_line, num_col].set_title(dof_names[i_dof], fontsize=8)
        axs[num_line, num_col].set_xlim(time_min_graph, time_max_graph)
        axs[num_line, num_col].grid(True, linewidth=0.4)
        # Réduire la taille des labels des xticks et yticks
        axs[num_line, num_col].tick_params(axis="both", which="major", labelsize=6)
        if num_line == 0:
            axs[num_line, num_col].set_ylim(-y_max_1 + (-y_max_1 * 0.1), y_max_1 + (y_max_1 * 0.1))
        elif num_line == 1:
            axs[num_line, num_col].set_ylim(-y_max_2 + (-y_max_2 * 0.1), y_max_2 + (y_max_2 * 0.1))
        elif num_line == 2:
            axs[num_line, num_col].set_ylim(-y_max_3 + (-y_max_3 * 0.1), y_max_3 + (y_max_3 * 0.1))

        num_col = num_col + 1
        if i_dof == 1:
            num_col = 0
            num_line += 1
        if num_col == 3:
            num_col = 0
            num_line += 1
        if num_line == 2:
            axs[num_line, num_col].set_xlabel("Time [s]", fontsize=7)

    # Y_label
    axs[0, 0].set_ylabel("Position [m]", fontsize=7)  # Pelvis Translation
    axs[1, 0].set_ylabel(r"Joint angle [$^\circ$]", fontsize=7)  # Pelvis Rotation
    axs[2, 0].set_ylabel(r"Joint angle [$^\circ$]", fontsize=7)  # Thight Rotation

    axs[0, 0].fill_between(
        np.array([0, 0]),
        np.array([0, 0]),
        np.array([0, 0]),
        color="tab:green",
        alpha=0.1,
        step="pre",
        linewidth=1,
        label="Joint coordinates bounds ($q_{bounds}$) - NTC",
    )
    axs[0, 0].fill_between(
        np.array([0, 0]),
        np.array([0, 0]),
        np.array([0, 0]),
        color="tab:blue",
        alpha=0.1,
        step="pre",
        linewidth=1,
        label="Joint coordinates bounds ($q_{bounds}$) - KTC",
    )
    axs[0, 0].fill_between(
        np.array([0, 0]),
        np.array([0, 0]),
        np.array([0, 0]),
        color="tab:orange",
        alpha=0.1,
        step="pre",
        linewidth=1,
        label="Joint coordinates bounds ($q_{bounds}$) - HTC",
    )
    handles, labels = axs[0, 0].get_legend_handles_labels()

    # Ajouter la légende à la figure de la première ligne, troisième colonne
    axs[0, 2].legend(handles, labels, loc="center", fontsize=8)
    axs[0, 2].axis("off")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.4)
    fig.savefig("q" + "." + format_graph, format=format_graph, dpi=300)

    # Figure qdot
    fig, axs = plt.subplots(3, 3, figsize=(10, 6))
    num_col = 0
    num_line = 0
    y_max_1 = np.max([abs(qdot_without_deg[0:2, :]), abs(qdot_without_deg[0:2, :])])
    y_max_2 = np.max([abs(qdot_without_deg[2:5, :]), abs(qdot_without_deg[2:5, :])])
    y_max_3 = np.max([abs(qdot_without_deg[5:, :]), abs(qdot_without_deg[5:, :])])
    for i_dof in range(qdot_CL_deg.shape[0]):
        axs[num_line, num_col].plot(np.array([time_min_graph, time_max_graph]), np.array([0, 0]), "-k", linewidth=0.5)

        plot_all_lines(time_end_phase_CL, time_end_phase_without, time_end_phase_free, axs[num_line, num_col])

        axs[num_line, num_col].plot(
            time_vector_free,
            qdot_free_deg[i_dof],
            color="tab:green",
            label="No tucking constraint",
            alpha=0.75,
            linewidth=1,
        )
        axs[num_line, num_col].plot(
            time_tuck_free,
            qdot_free_deg[i_dof, tuck_node_start:tuck_node_end],
            color="tab:green",
            alpha=0.75,
            linewidth=3,
        )
        axs[num_line, num_col].plot(
            time_vector_without,
            qdot_without_deg[i_dof],
            color="tab:blue",
            label="Kinematic tucking constraints",
            alpha=0.75,
            linewidth=1,
        )
        axs[num_line, num_col].plot(
            time_tuck_without,
            qdot_without_deg[i_dof, tuck_node_start:tuck_node_end],
            color="tab:blue",
            alpha=0.75,
            linewidth=3,
        )
        axs[num_line, num_col].plot(
            time_vector_CL,
            qdot_CL_deg[i_dof],
            color="tab:orange",
            label="Holonomic tucking constraints",
            alpha=0.75,
            linewidth=1,
        )
        axs[num_line, num_col].plot(
            time_tuck_CL,
            qdot_CL_deg[i_dof, tuck_node_start:tuck_node_end],
            color="tab:orange",
            alpha=0.75,
            linewidth=3,
        )

        axs[num_line, num_col].set_title(dof_names[i_dof])

        axs[num_line, num_col].set_title(dof_names[i_dof], fontsize=8)
        axs[num_line, num_col].set_xlim(time_min_graph, time_max_graph)
        axs[num_line, num_col].grid(True, linewidth=0.4)
        # Réduire la taille des labels des xticks et yticks
        axs[num_line, num_col].tick_params(axis="both", which="major", labelsize=6)
        if num_line == 0:
            axs[num_line, num_col].set_ylim(-y_max_1 + (-y_max_1 * 0.1), y_max_1 + (y_max_1 * 0.1))
        elif num_line == 1:
            axs[num_line, num_col].set_ylim(-y_max_2 + (-y_max_2 * 0.1), y_max_2 + (y_max_2 * 0.1))
        elif num_line == 2:
            axs[num_line, num_col].set_ylim(-y_max_3 + (-y_max_3 * 0.1), y_max_3 + (y_max_3 * 0.1))

        num_col = num_col + 1
        if i_dof == 1:
            num_col = 0
            num_line += 1
        if num_col == 3:
            num_col = 0
            num_line += 1
        if num_line == 2:
            axs[num_line, num_col].set_xlabel("Time [s]", fontsize=7)

    # Y_label
    axs[0, 0].set_ylabel("Velocity [m/s]", fontsize=7)  # Pelvis Translation
    axs[1, 0].set_ylabel(r"Joint angular velocity [$^\circ/s$]", fontsize=7)  # Pelvis Rotation
    axs[2, 0].set_ylabel(r"Joint angular velocity [$^\circ/s$]", fontsize=7)  # Thight Rotation

    handles, labels = axs[0, 0].get_legend_handles_labels()

    # Ajouter la légende à la figure de la première ligne, troisième colonne
    axs[0, 2].legend(handles, labels, loc="center", fontsize=7)
    axs[0, 2].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.4)
    fig.savefig("qdot" + "." + format_graph, format=format_graph, dpi=300)

    # Figure tau
    tau_CL_min_bound = np.zeros((5, tau_CL.shape[1]))
    tau_CL_max_bound = np.zeros((5, tau_CL.shape[1]))
    tau_without_min_bound = np.zeros((5, tau_without.shape[1]))
    tau_without_max_bound = np.zeros((5, tau_without.shape[1]))
    tau_free_min_bound = np.zeros((5, tau_free.shape[1]))
    tau_free_max_bound = np.zeros((5, tau_free.shape[1]))
    for i_dof, key in enumerate(ACTUATORS.keys()):
        # Change the sign of the theta_opt for the knee
        if i_dof == 3:
            true_min_CL = -actuator_function(
                ACTUATORS[key].tau_max_minus,
                ACTUATORS[key].theta_opt_minus,
                ACTUATORS[key].r_minus,
                -q_CL_rad[i_dof + 3],
            )
            true_max_CL = actuator_function(
                ACTUATORS[key].tau_max_plus,
                ACTUATORS[key].theta_opt_plus,
                ACTUATORS[key].r_plus,
                -q_CL_rad[i_dof + 3],
            )
            true_min_without = -actuator_function(
                ACTUATORS[key].tau_max_minus,
                ACTUATORS[key].theta_opt_minus,
                ACTUATORS[key].r_minus,
                -q_without_rad[i_dof + 3],
            )
            true_max_without = actuator_function(
                ACTUATORS[key].tau_max_plus,
                ACTUATORS[key].theta_opt_plus,
                ACTUATORS[key].r_plus,
                -q_without_rad[i_dof + 3],
            )
            true_min_free = -actuator_function(
                ACTUATORS[key].tau_max_minus,
                ACTUATORS[key].theta_opt_minus,
                ACTUATORS[key].r_minus,
                -q_free_rad[i_dof + 3],
            )
            true_max_free = actuator_function(
                ACTUATORS[key].tau_max_plus,
                ACTUATORS[key].theta_opt_plus,
                ACTUATORS[key].r_plus,
                -q_free_rad[i_dof + 3],
            )
            tau_CL_min_bound[i_dof, :] = -true_max_CL
            tau_CL_max_bound[i_dof, :] = -true_min_CL
            tau_without_min_bound[i_dof, :] = -true_max_without
            tau_without_max_bound[i_dof, :] = -true_min_without
            tau_free_min_bound[i_dof, :] = -true_max_free
            tau_free_max_bound[i_dof, :] = -true_min_free
        else:
            tau_CL_min_bound[i_dof, :] = -actuator_function(
                ACTUATORS[key].tau_max_minus,
                ACTUATORS[key].theta_opt_minus,
                ACTUATORS[key].r_minus,
                q_CL_rad[i_dof + 3],
            )
            tau_CL_max_bound[i_dof, :] = actuator_function(
                ACTUATORS[key].tau_max_plus,
                ACTUATORS[key].theta_opt_plus,
                ACTUATORS[key].r_plus,
                q_CL_rad[i_dof + 3],
            )
            tau_without_min_bound[i_dof, :] = -actuator_function(
                ACTUATORS[key].tau_max_minus,
                ACTUATORS[key].theta_opt_minus,
                ACTUATORS[key].r_minus,
                q_without_rad[i_dof + 3],
            )
            tau_without_max_bound[i_dof, :] = actuator_function(
                ACTUATORS[key].tau_max_plus,
                ACTUATORS[key].theta_opt_plus,
                ACTUATORS[key].r_plus,
                q_without_rad[i_dof + 3],
            )
            tau_free_min_bound[i_dof, :] = -actuator_function(
                ACTUATORS[key].tau_max_minus,
                ACTUATORS[key].theta_opt_minus,
                ACTUATORS[key].r_minus,
                q_free_rad[i_dof + 3],
            )
            tau_free_max_bound[i_dof, :] = actuator_function(
                ACTUATORS[key].tau_max_plus,
                ACTUATORS[key].theta_opt_plus,
                ACTUATORS[key].r_plus,
                q_free_rad[i_dof + 3],
            )

    # Figure tau
    fig, axs = plt.subplots(2, 3, figsize=(10, 4))
    num_col = 1
    num_line = 0

    y_max_1 = np.max([abs(tau_without[0:2, :]), abs(tau_CL[0:2, :]), abs(tau_free[0:2, :])])
    y_max_2 = np.max([abs(tau_without[2:, :]), abs(tau_CL[2:, :]), abs(tau_free[2:, :])])

    axs[0, 0].plot([], [], color="tab:green", label="No tucking constraint (NTC)")
    axs[0, 0].plot([], [], color="tab:blue", label="Kinematic tucking constraints (KTC)")
    axs[0, 0].plot([], [], color="tab:orange", label="Holonomic tucking contraints (HTC)")
    axs[0, 0].fill_between(
        [],
        [],
        [],
        color="tab:green",
        alpha=0.1,
        label=r"Non-physiological torque ($\tilde{\tau}^{max}(q, \tau)$) - NTC",
        linewidth=1,
    )
    axs[0, 0].fill_between(
        [],
        [],
        [],
        color="tab:blue",
        alpha=0.1,
        label=r"Non-physiological torque ($\tilde{\tau}^{max}(q, \tau)$) - KTC",
        linewidth=1,
    )
    axs[0, 0].fill_between(
        [],
        [],
        [],
        color="tab:orange",
        alpha=0.1,
        label=r"Non-physiological torque ($\tilde{\tau}^{max}(q, \tau)$) - (HTC)",
        linewidth=1,
    )
    axs[0, 0].legend(loc="center right", bbox_to_anchor=(0.9, 0.5), fontsize=8)
    axs[0, 0].axis("off")

    for i_dof in range(tau_CL.shape[0]):
        axs[num_line, num_col].plot(np.array([time_min_graph, time_max_graph]), np.array([0, 0]), "-k", linewidth=0.5)
        axs[num_line, num_col].fill_between(
            time_vector_free,
            tau_free_max_bound[i_dof],
            np.ones(tau_free_max_bound[i_dof].shape) * 1000,
            color="tab:green",
            alpha=0.1,
            step="pre",
            linewidth=1,
        )
        axs[num_line, num_col].fill_between(
            time_vector_free,
            np.ones(tau_free_max_bound[i_dof].shape) * -1000,
            tau_free_min_bound[i_dof],
            color="tab:green",
            alpha=0.1,
            step="pre",
            linewidth=1,
        )
        axs[num_line, num_col].fill_between(
            time_vector_without,
            tau_without_max_bound[i_dof],
            np.ones(tau_without_max_bound[i_dof].shape) * 1000,
            color="tab:blue",
            alpha=0.1,
            step="pre",
            linewidth=1,
        )
        axs[num_line, num_col].fill_between(
            time_vector_without,
            np.ones(tau_without_max_bound[i_dof].shape) * -1000,
            tau_without_min_bound[i_dof],
            color="tab:blue",
            alpha=0.1,
            step="pre",
            linewidth=1,
        )
        axs[num_line, num_col].fill_between(
            time_vector_CL,
            tau_CL_max_bound[i_dof],
            np.ones(tau_CL_max_bound[i_dof].shape) * 1000,
            color="tab:orange",
            alpha=0.1,
            step="pre",
            linewidth=1,
        )
        axs[num_line, num_col].fill_between(
            time_vector_CL,
            np.ones(tau_CL_max_bound[i_dof].shape) * -1000,
            tau_CL_min_bound[i_dof],
            color="tab:orange",
            alpha=0.1,
            step="pre",
            linewidth=1,
        )

        axs[num_line, num_col].plot(
            time_vector_free,
            tau_free[i_dof],
            color="tab:green",
            alpha=0.75,
            linewidth=1,
            label="No tucking constraint",
        )
        axs[num_line, num_col].plot(
            time_tuck_free,
            tau_free[i_dof, tuck_node_start:tuck_node_end],
            color="tab:green",
            alpha=0.75,
            linewidth=3,
        )
        axs[num_line, num_col].plot(
            time_vector_without,
            tau_without[i_dof],
            color="tab:blue",
            alpha=0.75,
            linewidth=1,
            label="Kinematic tucking constraints",
        )
        axs[num_line, num_col].plot(
            time_tuck_without,
            tau_without[i_dof, tuck_node_start:tuck_node_end],
            color="tab:blue",
            alpha=0.75,
            linewidth=3,
        )
        axs[num_line, num_col].plot(
            time_vector_CL,
            tau_CL[i_dof],
            color="tab:orange",
            alpha=0.75,
            linewidth=1,
            label="Holonomic tucking constraints",
        )
        axs[num_line, num_col].plot(
            time_tuck_CL,
            tau_CL[i_dof, tuck_node_start:tuck_node_end],
            color="tab:orange",
            alpha=0.75,
            linewidth=3,
        )
        plot_all_lines(time_end_phase_CL, time_end_phase_without, time_end_phase_free, axs[num_line, num_col])

        axs[num_line, num_col].set_title(dof_names_tau[i_dof], fontsize=8)
        axs[num_line, num_col].set_xlim(time_min_graph, time_max_graph)
        axs[num_line, num_col].grid(True, linewidth=0.4)
        # Réduire la taille des labels des xticks et yticks
        axs[num_line, num_col].tick_params(axis="both", which="major", labelsize=6)
        if num_line == 0:
            axs[num_line, num_col].set_ylim(-y_max_1 + (-y_max_1 * 0.1), y_max_1 + (y_max_1 * 0.1))
        else:
            axs[num_line, num_col].set_ylim(-y_max_2 + (-y_max_2 * 0.1), y_max_2 + (y_max_2 * 0.1))

        num_col += 1
        if num_col == 3:
            num_col = 0
            num_line += 1
        if num_line == 1:
            axs[num_line, num_col].set_xlabel("Time [s]", fontsize=7)

    # Y_label
    axs[0, 1].set_ylabel("Joint torque [Nm]", fontsize=7)  # Arm Rotation
    axs[1, 0].set_ylabel("Joint torque [Nm]", fontsize=7)  # Leg Rotation
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.4)
    fig.savefig("tau" + "." + format_graph, format=format_graph, dpi=300)

    # Tau ratio all phases
    tau_CL_ratio_all = np.zeros(tau_CL.shape)
    tau_without_ratio_all = np.zeros(tau_without.shape)
    tau_free_ratio_all = np.zeros(tau_free.shape)

    fig, axs = plt.subplots(2, 2, figsize=(10, 4))
    axs[0, 0].plot(
        time_vector_free,
        np.sum(np.abs(tau_free), axis=0),
        color="tab:green",
        label="No tucking constraint (NTC)",
        alpha=0.75,
        linewidth=1,
    )
    axs[0, 0].plot(
        time_tuck_free,
        np.sum(np.abs(tau_free[:, tuck_node_start:tuck_node_end]), axis=0),
        color="tab:green",
        alpha=0.75,
        linewidth=3,
    )
    axs[0, 0].plot(
        time_vector_without,
        np.sum(np.abs(tau_without), axis=0),
        color="tab:blue",
        label="Kinematic tucking constraints (KTC)",
        alpha=0.75,
        linewidth=1,
    )
    axs[0, 0].plot(
        time_tuck_without,
        np.sum(np.abs(tau_without[:, tuck_node_start:tuck_node_end]), axis=0),
        color="tab:blue",
        alpha=0.75,
        linewidth=3,
    )
    axs[0, 0].plot(
        time_vector_CL,
        np.sum(np.abs(tau_CL), axis=0),
        color="tab:orange",
        label="Holonomic tucking constraints (HTC)",
        alpha=0.75,
        linewidth=1,
    )
    axs[0, 0].plot(
        time_tuck_CL,
        np.sum(np.abs(tau_CL[:, tuck_node_start:tuck_node_end]), axis=0),
        color="tab:orange",
        alpha=0.75,
        linewidth=3,
    )
    plot_all_lines(time_end_phase_CL, time_end_phase_without, time_end_phase_free, axs[0, 0])

    for i_dof in range(5):
        for i_node in range(tau_CL.shape[1]):
            if tau_CL[i_dof, i_node] > 0:
                tau_CL_ratio_all[i_dof, i_node] = tau_CL[i_dof, i_node] / tau_CL_max_bound[i_dof, i_node]
            else:
                tau_CL_ratio_all[i_dof, i_node] = np.abs(tau_CL[i_dof, i_node] / tau_CL_min_bound[i_dof, i_node])
            if tau_without[i_dof, i_node] > 0:
                tau_without_ratio_all[i_dof, i_node] = tau_without[i_dof, i_node] / tau_without_max_bound[i_dof, i_node]
            else:
                tau_without_ratio_all[i_dof, i_node] = np.abs(
                    tau_without[i_dof, i_node] / tau_without_min_bound[i_dof, i_node]
                )
            if tau_free[i_dof, i_node] > 0:
                tau_free_ratio_all[i_dof, i_node] = tau_free[i_dof, i_node] / tau_free_max_bound[i_dof, i_node]
            else:
                tau_free_ratio_all[i_dof, i_node] = np.abs(tau_free[i_dof, i_node] / tau_free_min_bound[i_dof, i_node])

    axs[1, 0].plot(
        time_vector_free,
        np.sum(np.abs(tau_free_ratio_all), axis=0),
        color="tab:green",
        label="No tucking constraint",
        alpha=0.75,
        linewidth=1,
    )
    axs[1, 0].plot(
        time_tuck_free,
        np.sum(np.abs(tau_free_ratio_all[:, tuck_node_start:tuck_node_end]), axis=0),
        color="tab:green",
        alpha=0.75,
        linewidth=3,
    )
    axs[1, 0].plot(
        time_vector_without,
        np.sum(np.abs(tau_without_ratio_all), axis=0),
        color="tab:blue",
        label="Kinematic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )
    axs[1, 0].plot(
        time_tuck_without,
        np.sum(np.abs(tau_without_ratio_all[:, tuck_node_start:tuck_node_end]), axis=0),
        color="tab:blue",
        alpha=0.75,
        linewidth=3,
    )
    axs[1, 0].plot(
        time_vector_CL,
        np.sum(np.abs(tau_CL_ratio_all), axis=0),
        color="tab:orange",
        label="Holonomic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )
    axs[1, 0].plot(
        time_tuck_CL,
        np.sum(np.abs(tau_CL_ratio_all[:, tuck_node_start:tuck_node_end]), axis=0),
        color="tab:orange",
        alpha=0.75,
        linewidth=3,
    )

    plot_all_lines(time_end_phase_CL, time_end_phase_without, time_end_phase_free, axs[1, 0])

    # Tau ratio all phases
    # tau_CL_ratio_all = np.zeros(tau_CL.shape)
    # tau_without_ratio_all = np.zeros(tau_without.shape)
    # tau_free_ratio_all = np.zeros(tau_free.shape)

    integral_tau_all_CL = np.trapz(np.sum(np.abs(tau_CL), axis=0), x=time_vector_CL)
    integral_tau_all_without = np.trapz(np.sum(np.abs(tau_without), axis=0), x=time_vector_without)
    integral_tau_all_free = np.trapz(np.sum(np.abs(tau_free), axis=0), x=time_vector_free)

    axs[0, 1].bar(
        [0, 1, 2],
        [integral_tau_all_free, integral_tau_all_without, integral_tau_all_CL],
        color=["tab:green", "tab:blue", "tab:orange"],
    )
    axs[0, 1].text(0, integral_tau_all_free + 0.1, f"{integral_tau_all_free:.2f} Nms", ha="center", va="bottom")
    axs[0, 1].text(1, integral_tau_all_without + 0.1, f"{integral_tau_all_without:.2f} Nms", ha="center", va="bottom")
    axs[0, 1].text(2, integral_tau_all_CL + 0.1, f"{integral_tau_all_CL:.2f} Nms", ha="center", va="bottom")

    integral_tau_all_ratio_CL = np.trapz(np.sum(np.abs(tau_CL_ratio_all), axis=0), x=time_vector_CL)
    integral_tau_all_ratio_without = np.trapz(np.sum(np.abs(tau_without_ratio_all), axis=0), x=time_vector_without)
    integral_tau_all_ratio_free = np.trapz(np.sum(np.abs(tau_free_ratio_all), axis=0), x=time_vector_free)
    axs[1, 1].bar(
        [0, 1, 2],
        [integral_tau_all_ratio_free, integral_tau_all_ratio_without, integral_tau_all_ratio_CL],
        color=["tab:green", "tab:blue", "tab:orange"],
    )
    axs[1, 1].text(
        0, integral_tau_all_ratio_free + 0.1, f"{integral_tau_all_ratio_free:.2f} s", ha="center", va="bottom"
    )
    axs[1, 1].text(
        1, integral_tau_all_ratio_without + 0.1, f"{integral_tau_all_ratio_without:.2f} s", ha="center", va="bottom"
    )
    axs[1, 1].text(2, integral_tau_all_ratio_CL + 0.1, f"{integral_tau_all_ratio_CL:.2f} s", ha="center", va="bottom")

    axs[0, 0].set_ylabel("Joint torque \n[Nm]")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Physiological \njoint torque ratio")
    axs[0, 1].set_xticks([0, 1, 2], ["NTC", "KTC", "HTC"])
    axs[1, 1].set_xticks([0, 1, 2], ["NTC", "KTC", "HTC"])
    axs[0, 1].set_ylabel(r"$\int{ | \tau | dt}$" + "\n[Nm.s]")
    axs[0, 1].set_ylim(0, 1200)
    axs[1, 1].set_ylim(0, 8)
    axs[0, 0].set_xticks([0.0, 0.5, 1.0, 1.5], ["0.0", "0.5", "1.0", "1.5"])
    axs[1, 0].set_xticks([0.0, 0.5, 1.0, 1.5], ["0.0", "0.5", "1.0", "1.5"])

    axs[1, 1].plot([], [], color="tab:green", label="No tucking constraint")
    axs[1, 1].plot([], [], color="tab:blue", label="Kinematic tucking constraints")
    axs[1, 1].plot([], [], color="tab:orange", label="Holonomic tucking constraints")
    axs[1, 1].legend(bbox_to_anchor=(1.0, 2.8), ncol=3)

    axs[1, 1].set_ylabel(r"$\int{ | \tau/{\tilde{\tau}^{max}} | dt}$" + "\n[s]")
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.85)
    plt.savefig("tau_ratio_all" + "." + format_graph, format=format_graph, dpi=300)
    # plt.show()

    fig, axs = plt.subplots(1, 1, figsize=(10, 4))

    for i_dof in range(5):
        for i_node in range(tau_CL.shape[1]):
            if tau_CL[i_dof, i_node] > 0:
                tau_CL_ratio_all[i_dof, i_node] = tau_CL[i_dof, i_node] / tau_CL_max_bound[i_dof, i_node]
            else:
                tau_CL_ratio_all[i_dof, i_node] = np.abs(tau_CL[i_dof, i_node] / tau_CL_min_bound[i_dof, i_node])
            if tau_without[i_dof, i_node] > 0:
                tau_without_ratio_all[i_dof, i_node] = tau_without[i_dof, i_node] / tau_without_max_bound[i_dof, i_node]
            else:
                tau_without_ratio_all[i_dof, i_node] = np.abs(
                    tau_without[i_dof, i_node] / tau_without_min_bound[i_dof, i_node]
                )
            if tau_free[i_dof, i_node] > 0:
                tau_free_ratio_all[i_dof, i_node] = tau_free[i_dof, i_node] / tau_free_max_bound[i_dof, i_node]
            else:
                tau_free_ratio_all[i_dof, i_node] = np.abs(tau_free[i_dof, i_node] / tau_free_min_bound[i_dof, i_node])

    axs.plot(
        time_vector_free,
        np.abs(tau_free_ratio_all[2, :]),
        color="tab:green",
        label="No tucking constraint",
        alpha=0.73,
        linewidth=1,
    )
    axs.plot(
        time_tuck_free,
        np.abs(tau_free_ratio_all[2, tuck_node_start:tuck_node_end]),
        color="tab:green",
        alpha=0.73,
        linewidth=3,
    )
    axs.plot(
        time_vector_without,
        np.abs(tau_without_ratio_all)[2, :],
        color="tab:blue",
        label="Kinematic tucking constraints",
        alpha=0.73,
        linewidth=1,
    )
    axs.plot(
        time_tuck_without,
        np.abs(tau_without_ratio_all[2, tuck_node_start:tuck_node_end]),
        color="tab:blue",
        alpha=0.73,
        linewidth=3,
    )
    axs.plot(
        time_vector_CL,
        np.abs(tau_CL_ratio_all[2, :]),
        color="tab:orange",
        label="Holonomic tucking constraints",
        alpha=0.73,
        linewidth=1,
    )
    axs.plot(
        time_tuck_CL,
        np.abs(tau_CL_ratio_all[2, tuck_node_start:tuck_node_end]),
        color="tab:orange",
        alpha=0.73,
        linewidth=3,
    )

    plot_all_lines(time_end_phase_CL, time_end_phase_without, time_end_phase_free, axs)

    axs.set_xlabel("Time [s]")
    axs.set_ylabel("Physiological Hip\njoint torque ratio")
    # axs.set_ylim(0, 1200)
    # axs.set_ylim(0, 8)
    axs.set_xticks([0.0, 0.5, 1.0, 1.5], ["0.0", "0.5", "1.0", "1.5"])
    axs.set_xticks([0.0, 0.5, 1.0, 1.5], ["0.0", "0.5", "1.0", "1.5"])

    axs.plot([], [], color="tab:green", label="No tucking constraint")
    axs.plot([], [], color="tab:blue", label="Kinematic tucking constraints")
    axs.plot([], [], color="tab:orange", label="Holonomic tucking constraints")
    axs.legend(bbox_to_anchor=(1.0, 2.8), ncol=3)
    axs.set_xlim(time_min_graph, time_max_graph)
    axs.grid(True, linewidth=0.4)

    plt.savefig("tau_ratio_hip" + "." + format_graph, format=format_graph, dpi=300)
    plt.show()

    hip_tau_CL = np.trapz(np.abs(tau_CL[2, aerial_node_start:aerial_node_end]), x=time_vector_CL[aerial_node_start:aerial_node_end])
    hip_tau_without = np.trapz(np.abs(tau_without[2, aerial_node_start:aerial_node_end]), x=time_vector_without[aerial_node_start:aerial_node_end])
    hip_tau_free = np.trapz(np.abs(tau_free[2, aerial_node_start:aerial_node_end]), x=time_vector_free[aerial_node_start:aerial_node_end])
    print("Hip tau CL: ", hip_tau_CL)
    print("Hip tau without: ", hip_tau_without)
    print("Hip tau free: ", hip_tau_free)
    print(f"A reduction of {(hip_tau_without - hip_tau_CL) / hip_tau_without * 100:.2f}% with the HTC")
    print(f"A reduction of {(hip_tau_free - hip_tau_CL) / hip_tau_free * 100:.2f}% with the NTC")

    hip_tau_CL_ratio_all = np.trapz(np.abs(tau_CL_ratio_all[2, aerial_node_start:aerial_node_end]), x=time_vector_CL[aerial_node_start:aerial_node_end])
    hip_tau_without_ratio_all = np.trapz(np.abs(tau_without_ratio_all[2, aerial_node_start:aerial_node_end]), x=time_vector_without[aerial_node_start:aerial_node_end])
    hip_tau_free_ratio_all = np.trapz(np.abs(tau_free_ratio_all[2, aerial_node_start:aerial_node_end]), x=time_vector_free[aerial_node_start:aerial_node_end])
    print("Hip physiological tau CL: ", hip_tau_CL_ratio_all)
    print("Hip physiological tau without: ", hip_tau_without_ratio_all)
    print("Hip physiological tau free: ", hip_tau_free_ratio_all)
    print(f"A reduction of {(hip_tau_without_ratio_all - hip_tau_CL_ratio_all) / hip_tau_without_ratio_all * 100:.2f}% with the HTC")
    print(f"A reduction of {(hip_tau_free_ratio_all - hip_tau_CL_ratio_all) / hip_tau_free_ratio_all * 100:.2f}% with the NTC")


    knee_tau_CL = np.trapz(np.abs(tau_CL[3, aerial_node_start:aerial_node_end]), x=time_vector_CL[aerial_node_start:aerial_node_end])
    knee_tau_without = np.trapz(np.abs(tau_without[3, aerial_node_start:aerial_node_end]), x=time_vector_without[aerial_node_start:aerial_node_end])
    knee_tau_free = np.trapz(np.abs(tau_free[3, aerial_node_start:aerial_node_end]), x=time_vector_free[aerial_node_start:aerial_node_end])
    print("knee tau CL: ", knee_tau_CL)
    print("knee tau without: ", knee_tau_without)
    print("knee tau free: ", knee_tau_free)
    print(f"A reduction of {(knee_tau_without - knee_tau_CL) / knee_tau_without * 100:.2f}% with the HTC")
    print(f"A reduction of {(knee_tau_free - knee_tau_CL) / knee_tau_free * 100:.2f}% with the NTC")

    knee_tau_CL_ratio_all = np.trapz(np.abs(tau_CL_ratio_all[3, aerial_node_start:aerial_node_end]), x=time_vector_CL[aerial_node_start:aerial_node_end])
    knee_tau_without_ratio_all = np.trapz(np.abs(tau_without_ratio_all[3, aerial_node_start:aerial_node_end]), x=time_vector_without[aerial_node_start:aerial_node_end])
    knee_tau_free_ratio_all = np.trapz(np.abs(tau_free_ratio_all[3, aerial_node_start:aerial_node_end]), x=time_vector_free[aerial_node_start:aerial_node_end])
    print("knee physiological tau CL: ", knee_tau_CL_ratio_all)
    print("knee physiological tau without: ", knee_tau_without_ratio_all)
    print("knee physiological tau free: ", knee_tau_free_ratio_all)
    print(f"A reduction of {(knee_tau_without_ratio_all - knee_tau_CL_ratio_all) / knee_tau_without_ratio_all * 100:.2f}% with the HTC")
    print(f"A reduction of {(knee_tau_free_ratio_all - knee_tau_CL_ratio_all) / knee_tau_free_ratio_all * 100:.2f}% with the NTC")


    arm_tau_CL = np.trapz(np.abs(tau_CL[0, aerial_node_start:aerial_node_end]) + np.abs(tau_CL[1, aerial_node_start:aerial_node_end]), x=time_vector_CL[aerial_node_start:aerial_node_end])
    arm_tau_without = np.trapz(
        np.abs(tau_without[0, aerial_node_start:aerial_node_end]) + np.abs(tau_without[1, aerial_node_start:aerial_node_end]), x=time_vector_without[aerial_node_start:aerial_node_end]
    )
    arm_tau_free = np.trapz(np.abs(tau_free[0, aerial_node_start:aerial_node_end]) + np.abs(tau_free[1, aerial_node_start:aerial_node_end]), x=time_vector_free[aerial_node_start:aerial_node_end])
    print("Arm tau CL: ", arm_tau_CL)
    print("Arm tau without: ", arm_tau_without)
    print("Arm tau free: ", arm_tau_free)
    print(f"A reduction of {(arm_tau_without - arm_tau_CL) / arm_tau_without * 100:.2f}% with the HTC")
    print(f"A reduction of {(arm_tau_free - hip_tau_CL) / arm_tau_free * 100:.2f}% with the NTC")

    arm_tau_CL_ratio_all = np.trapz(np.abs(tau_CL_ratio_all[0, aerial_node_start:aerial_node_end]) + np.abs(tau_CL_ratio_all[1, aerial_node_start:aerial_node_end]), x=time_vector_CL[aerial_node_start:aerial_node_end])
    arm_tau_without_ratio_all = np.trapz(
        np.abs(tau_without_ratio_all[0, aerial_node_start:aerial_node_end]) + np.abs(tau_without_ratio_all[1, aerial_node_start:aerial_node_end]), x=time_vector_without[aerial_node_start:aerial_node_end]
    )
    arm_tau_free_ratio_all = np.trapz(np.abs(tau_free_ratio_all[0, aerial_node_start:aerial_node_end]) + np.abs(tau_free_ratio_all[1, aerial_node_start:aerial_node_end]), x=time_vector_free[aerial_node_start:aerial_node_end])
    print("Arm physiological tau CL: ", arm_tau_CL_ratio_all)
    print("Arm physiological tau without: ", arm_tau_without_ratio_all)
    print("Arm physiological tau free: ", arm_tau_free_ratio_all)
    print(f"A reduction of {(arm_tau_without_ratio_all - arm_tau_CL_ratio_all) / arm_tau_without_ratio_all * 100:.2f}% with the HTC")
    print(f"A reduction of {(arm_tau_free_ratio_all - arm_tau_CL_ratio_all) / arm_tau_free_ratio_all * 100:.2f}% with the NTC")

    # Figure taudot
    fig, axs = plt.subplots(2, 3, figsize=(10, 4))
    num_col = 1
    num_line = 0

    y_min = np.min([taudot_without, taudot_CL, taudot_free])
    y_max = np.max([taudot_without, taudot_CL, taudot_free])

    axs[0, 0].plot([], [], color="tab:green", label="No tucking constraint")
    axs[0, 0].plot([], [], color="tab:blue", label="Kinematic tucking constraints")
    axs[0, 0].plot([], [], color="tab:orange", label="Holonomic tucking contraints")
    axs[0, 0].legend(loc="center right", bbox_to_anchor=(0.9, 0.5), fontsize=8)
    axs[0, 0].axis("off")

    for i_dof in range(tau_CL.shape[0]):
        axs[num_line, num_col].plot(np.array([time_min_graph, time_max_graph]), np.array([0, 0]), "-k", linewidth=0.5)

        axs[num_line, num_col].step(
            time_control_free,
            taudot_free[i_dof],
            color="tab:green",
            alpha=0.75,
            linewidth=1,
            label="No tucking constraint",
        )
        axs[num_line, num_col].step(
            time_tuck_free[:-1],
            taudot_free[i_dof, 40:70],
            color="tab:green",
            alpha=0.75,
            linewidth=3,
        )
        axs[num_line, num_col].step(
            time_control_without,
            taudot_without[i_dof],
            color="tab:blue",
            alpha=0.75,
            linewidth=1,
            label="Kinematic tucking constraints",
        )
        axs[num_line, num_col].step(
            time_tuck_without[:-1],
            taudot_without[i_dof, 40:70],
            color="tab:blue",
            alpha=0.75,
            linewidth=3,
        )
        axs[num_line, num_col].step(
            time_control_CL,
            taudot_CL[i_dof],
            color="tab:orange",
            alpha=0.75,
            linewidth=1,
            label="Holonomic tucking constraints",
        )
        axs[num_line, num_col].step(
            time_tuck_CL[:-1],
            taudot_CL[i_dof, 40:70],
            color="tab:orange",
            alpha=0.75,
            linewidth=3,
        )
        plot_all_lines(time_end_phase_CL, time_end_phase_without, time_end_phase_free, axs[num_line, num_col])

        axs[num_line, num_col].set_title(dof_names_tau[i_dof], fontsize=8)
        axs[num_line, num_col].set_xlim(time_min_graph, time_max_graph)
        axs[num_line, num_col].grid(True, linewidth=0.4)

        # Réduire la taille des labels des xticks et yticks
        axs[num_line, num_col].tick_params(axis="both", which="major", labelsize=6)
        axs[num_line, num_col].set_ylim(y_min*1.05, y_max*1.05)

        num_col += 1
        if num_col == 3:
            num_col = 0
            num_line += 1
        if num_line == 1:
            axs[num_line, num_col].set_xlabel("Time [s]", fontsize=7)

    # Y_label
    axs[0, 1].set_ylabel("Joint torque derivative [Nm/s]", fontsize=7)
    axs[1, 0].set_ylabel("Joint torque derivative [Nm/s]", fontsize=7)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.4)
    fig.savefig("taudot" + "." + format_graph, format=format_graph, dpi=300)


if PLOT_INERTIA_FLAG:
    data_CL = pd.read_pickle(sol_CL)
    data_without = pd.read_pickle(sol_without)
    data_free = pd.read_pickle(sol_free)

    # Inertia
    inertia_CL = np.zeros((data_CL["q_all"].shape[1], 3))
    inertia_without = np.zeros((data_without["q_all"].shape[1], 3))
    inertia_free = np.zeros((data_free["q_all"].shape[1], 3))
    for i in range(data_CL["q_all"].shape[1]):
        inertia_CL[i, :] = np.diagonal(model.bodyInertia(data_CL["q_all"][:, i]).to_array()).T
        inertia_without[i, :] = np.diagonal(model.bodyInertia(data_without["q_all"][:, i]).to_array()).T
        inertia_free[i, :] = np.diagonal(model.bodyInertia(data_free["q_all"][:, i]).to_array()).T

    fig, ax = plt.subplots(4, 1, figsize=(8, 9))
    ax[0].plot(
        time_vector_free,
        inertia_free[:, 0],
        color="tab:green",
        label="No tucking constraint",
        alpha=0.75,
        linewidth=1,
    )
    ax[0].plot(
        time_tuck_free,
        inertia_free[tuck_node_start:tuck_node_end, 0],
        color="tab:green",
        alpha=0.75,
        linewidth=3,
    )
    ax[0].plot(
        time_vector_without,
        inertia_without[:, 0],
        color="tab:blue",
        label="Kinematic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )
    ax[0].plot(
        time_tuck_without,
        inertia_without[tuck_node_start:tuck_node_end, 0],
        color="tab:blue",
        alpha=0.75,
        linewidth=1,
    )
    ax[0].plot(
        time_vector_CL,
        inertia_CL[:, 0],
        color="tab:orange",
        label="Holonomic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )
    ax[0].plot(
        time_tuck_CL,
        inertia_CL[tuck_node_start:tuck_node_end, 0],
        color="tab:orange",
        alpha=0.75,
        linewidth=3,
    )
    plot_all_lines(time_end_phase_CL, time_end_phase_without, time_end_phase_free, ax[0])

    ax[0].set_ylabel("Moment of inertia\n" + r"[$kg.m^2$]")
    ax[0].set_xlim(time_min_graph, time_max_graph)
    ax[0].grid(True, linewidth=0.4)
    ax[0].legend(bbox_to_anchor=(1.1, 1.45), ncol=3)

    print("Min somersault inertia CL: ", np.min(inertia_CL[:, 0]))
    print("Min somersault inertia without: ", np.min(inertia_without[:, 0]))
    print("Min somersault inertia free: ", np.min(inertia_free[:, 0]))

    # Angular momentum
    # Adjust the DoFs to add a free floating base to work around a bug in biornd
    ang_mom_CL = np.zeros((data_CL["q_all"].shape[1], 3))
    ang_mom_without = np.zeros((data_without["q_all"].shape[1], 3))
    ang_mom_free = np.zeros((data_free["q_all"].shape[1], 3))

    adjusted_q_without = adjust_q_with_full_floating_base(data_without["q_all"])
    adjusted_qdot_without = adjust_q_with_full_floating_base(data_without["qdot_all"])
    adjusted_q_CL = adjust_q_with_full_floating_base(data_CL["q_all"])
    adjusted_qdot_CL = adjust_q_with_full_floating_base(data_CL["qdot_all"])
    adjusted_q_free = adjust_q_with_full_floating_base(data_free["q_all"])
    adjusted_qdot_free = adjust_q_with_full_floating_base(data_free["qdot_all"])

    for i in range(data_CL["q_all"].shape[1]):
        ang_mom_CL[i, :] = model_adjusted.angularMomentum(adjusted_q_CL[:, i], adjusted_qdot_CL[:, i], True).to_array()
        ang_mom_without[i, :] = model_adjusted.angularMomentum(
            adjusted_q_without[:, i], adjusted_qdot_without[:, i], True
        ).to_array()
        ang_mom_free[i, :] = model_adjusted.angularMomentum(
            adjusted_q_free[:, i], adjusted_qdot_free[:, i], True
        ).to_array()

    ax[1].plot(
        time_vector_free,
        np.linalg.norm(ang_mom_free, axis=1),
        color="tab:green",
        label="No tucking constraint",
        alpha=0.75,
        linewidth=1,
    )
    ax[1].plot(
        time_tuck_free,
        np.linalg.norm(ang_mom_free[tuck_node_start:tuck_node_end, :], axis=1),
        color="tab:green",
        alpha=0.75,
        linewidth=3,
    )
    ax[1].plot(
        time_vector_without,
        np.linalg.norm(ang_mom_without, axis=1),
        color="tab:blue",
        label="Kinematic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )
    ax[1].plot(
        time_tuck_without,
        np.linalg.norm(ang_mom_without[tuck_node_start:tuck_node_end, :], axis=1),
        color="tab:blue",
        alpha=0.75,
        linewidth=3,
    )
    ax[1].plot(
        time_vector_CL,
        np.linalg.norm(ang_mom_CL, axis=1),
        color="tab:orange",
        label="Holonomic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )
    ax[1].plot(
        time_tuck_CL,
        np.linalg.norm(ang_mom_CL[tuck_node_start:tuck_node_end, :], axis=1),
        color="tab:orange",
        alpha=0.75,
        linewidth=3,
    )
    plot_all_lines(time_end_phase_CL, time_end_phase_without, time_end_phase_free, ax[1])

    ax[1].set_ylabel("Angular momentum\n" + r"[$kg.m^2/s$]")
    ax[1].set_xlim(time_min_graph, time_max_graph)
    ax[1].grid(True, linewidth=0.4)

    print("Max angular momentum CL: ", np.max(np.linalg.norm(ang_mom_CL, axis=1)))
    print("Max angular momentum without: ", np.max(np.linalg.norm(ang_mom_without, axis=1)))
    print("Max angular momentum free: ", np.max(np.linalg.norm(ang_mom_free, axis=1)))

    # Body velocity
    body_velo_CL = np.zeros((data_CL["q_all"].shape[1], 3))
    body_velo_without = np.zeros((data_without["q_all"].shape[1], 3))
    body_velo_free = np.zeros((data_free["q_all"].shape[1], 3))
    for i in range(data_CL["q_all"].shape[1]):
        body_velo_CL[i, :] = (
            model.bodyAngularVelocity(data_CL["q_all"][:, i], data_CL["qdot_all"][:, i]).to_array() * 180 / np.pi
        )
        body_velo_without[i, :] = (
            model.bodyAngularVelocity(data_without["q_all"][:, i], data_without["qdot_all"][:, i]).to_array()
            * 180
            / np.pi
        )
        body_velo_free[i, :] = (
            model.bodyAngularVelocity(data_free["q_all"][:, i], data_free["qdot_all"][:, i]).to_array() * 180 / np.pi
        )

    ax[2].plot(
        time_vector_free,
        body_velo_free[:, 0],
        color="tab:green",
        label="No tucking constraint",
        alpha=0.75,
        linewidth=1,
    )
    ax[2].plot(
        time_tuck_free,
        body_velo_free[tuck_node_start:tuck_node_end, 0],
        color="tab:green",
        alpha=0.75,
        linewidth=3,
    )
    ax[2].plot(
        time_vector_without,
        body_velo_without[:, 0],
        color="tab:blue",
        label="Kinematic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )
    ax[2].plot(
        time_tuck_without,
        body_velo_without[tuck_node_start:tuck_node_end, 0],
        color="tab:blue",
        alpha=0.75,
        linewidth=3,
    )
    ax[2].plot(
        time_vector_CL,
        body_velo_CL[:, 0],
        color="tab:orange",
        label="Holonomic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )
    ax[2].plot(
        time_tuck_CL,
        body_velo_CL[tuck_node_start:tuck_node_end, 0],
        color="tab:orange",
        alpha=0.75,
        linewidth=3,
    )
    plot_all_lines(time_end_phase_CL, time_end_phase_without, time_end_phase_free, ax[2])

    ax[2].set_ylabel("Somersault velocity\n" + r"[$rad/s$]")
    ax[2].set_xlim(time_min_graph, time_max_graph)
    ax[2].grid(True, linewidth=0.4)

    print("Max somersault velocity CL: ", np.max(body_velo_CL[:, 0]))
    print("Max somersault velocity without: ", np.max(body_velo_without[:, 0]))
    print("Max somersault velocity free: ", np.max(body_velo_free[:, 0]))

    # Centrifugal effect
    centricugal_CL = model.mass() * (body_velo_CL[:, 0] * np.pi / 180) ** 2 * np.sqrt(inertia_CL[:, 0] / model.mass())
    centricugal_without = model.mass() * (body_velo_without[:, 0] * np.pi / 180) ** 2 * np.sqrt(inertia_without[:, 0] / model.mass())
    centricugal_free = model.mass() * (body_velo_free[:, 0] * np.pi / 180) ** 2 * np.sqrt(inertia_free[:, 0] / model.mass())

    ax[3].plot(
        time_vector_free,
        centricugal_free,
        color="tab:green",
        label="No tucking constraint",
        alpha=0.75,
        linewidth=1,
    )
    ax[3].plot(
        time_tuck_free,
        centricugal_free[tuck_node_start:tuck_node_end],
        color="tab:green",
        alpha=0.75,
        linewidth=3,
    )
    ax[3].plot(
        time_vector_without,
        centricugal_without,
        color="tab:blue",
        label="Kinematic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )
    ax[3].plot(
        time_tuck_without,
        centricugal_without[tuck_node_start:tuck_node_end],
        color="tab:blue",
        alpha=0.75,
        linewidth=3,
    )
    ax[3].plot(
        time_vector_CL,
        centricugal_CL,
        color="tab:orange",
        label="Holonomic tucking constraints",
        alpha=0.75,
        linewidth=1,
    )
    ax[3].plot(
        time_tuck_CL,
        centricugal_CL[tuck_node_start:tuck_node_end],
        color="tab:orange",
        alpha=0.75,
        linewidth=3,
    )
    plot_all_lines(time_end_phase_CL, time_end_phase_without, time_end_phase_free, ax[3])

    ax[3].set_ylabel("Centrifugal \npseudo-force\n" + r"[$N$]")
    ax[3].set_xlim(time_min_graph, time_max_graph)
    ax[3].grid(True, linewidth=0.4)
    ax[3].set_xlabel("Time [s]")

    fig.subplots_adjust()
    plt.savefig("Inertia" + "." + format_graph, format=format_graph)
    # plt.show()

    print("Max centrifugal pseudo-force CL:", np.max(centricugal_CL))
    print("Max centrifugal pseudo-force without: ", np.max(centricugal_without))
    print("Max centrifugal pseudo-force free: ", np.max(centricugal_free))

if PLOT_ENERGY_FLAG:
    power_without = np.abs(tau_without * qdot_without_rad[3:, :])
    power_free = np.abs(tau_free * qdot_free_rad[3:, :])
    power_CL = np.abs(tau_CL * qdot_CL_rad[3:, :])
    power_total_without = np.sum(power_without, axis=0)
    power_total_free = np.sum(power_free, axis=0)
    power_total_CL = np.sum(power_CL, axis=0)
    energy_without = np.trapz(power_total_without, time_vector_without)
    energy_free = np.trapz(power_total_free, time_vector_free)
    energy_CL = np.trapz(power_total_CL, time_vector_CL)

    print("Energy CL : ", energy_CL, "J")
    print("Energy without : ", energy_without, "J")
    print("Energy free : ", energy_free, "J")
    print("A reduction of ", (energy_without - energy_CL) / energy_without * 100, "% with the HTC")
    print("A reduction of ", (energy_free - energy_CL) / energy_free * 100, "% with the NTC")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(time_vector_free, power_total_free, color="tab:green", label="No tucking constraint")
    axs[0].plot(time_vector_without, power_total_without, color="tab:blue", label="Kinematic tucking constraints")
    axs[0].plot(time_vector_CL, power_total_CL, color="tab:orange", label="Holonomic tucking constraints")
    plot_all_lines(time_end_phase_CL, time_end_phase_without, time_end_phase_free, axs[0])

    axs[1].bar([0, 1, 2], [energy_free, energy_without, energy_CL], color=["tab:green", "tab:blue", "tab:orange"])
    axs[1].text(0, energy_free + 0.1, f"{energy_free:.2f} J", ha="center", va="bottom")
    axs[1].text(1, energy_without + 0.1, f"{energy_without:.2f} J", ha="center", va="bottom")
    axs[1].text(2, energy_CL + 0.1, f"{energy_CL:.2f} J", ha="center", va="bottom")

    axs[0].legend(bbox_to_anchor=(2.4, 1.3), ncol=3)
    axs[0].grid(True, linewidth=0.4)
    axs[0].set_ylabel("Joint power \n[J/s]")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_xticks([0.0, 0.5, 1.0, 1.5], ["0.0", "0.5", "1.0", "1.5"])

    axs[1].set_ylabel("Energy expenditure \n[J]")
    axs[1].set_ylim(0, 5500)
    axs[1].set_xticks([0, 1, 2], ["NTC", "KTC", "HTC"])

    plt.subplots_adjust(wspace=0.4, bottom=0.2, top=0.8)
    plt.savefig("Energy" + "." + format_graph, format=format_graph, dpi=300)
    plt.show()



print("Residual forces at take-off without: ", np.linalg.norm(data_without["contact_forces"][0][:, -1]), "N")
print("Residual forces at take-off CL: ", np.linalg.norm(data_CL["contact_forces"][0][:, -1]), "N")
print("Residual forces at take-off free: ", np.linalg.norm(data_free["contact_forces"][0][:, -1]), "N")

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(np.array([time_vector_free[0], time_vector_free[-1]]), np.array([0, 0]), "-k", linewidth=0.5)
axs[1].plot(np.array([time_vector_free[0], time_vector_free[-1]]), np.array([0, 0]), "-k", linewidth=0.5)
# Propulsion phase
axs[0].plot(time_vector_free[:n_shooting[0]+1], data_free["contact_forces"][0][0, :], color="tab:green", label="No tucking constraint")
axs[1].plot(time_vector_free[:n_shooting[0]+1], data_free["contact_forces"][0][1, :], color="tab:green", label="No tucking constraint")
axs[0].plot(time_vector_without[:n_shooting[0]+1], data_without["contact_forces"][0][0, :], color="tab:blue", label="Kinematic tucking constraints")
axs[1].plot(time_vector_without[:n_shooting[0]+1], data_without["contact_forces"][0][1, :], color="tab:blue", label="Kinematic tucking constraints")
axs[0].plot(time_vector_CL[:n_shooting[0]+1], data_CL["contact_forces"][0][0, :], color="tab:orange", label="Holonomic tucking constraints")
axs[1].plot(time_vector_CL[:n_shooting[0]+1], data_CL["contact_forces"][0][1, :], color="tab:orange", label="Holonomic tucking constraints")
# Landing phase
axs[0].plot(time_vector_free[-n_shooting[-1]-1:], data_free["contact_forces"][-1][0, :], color="tab:green")
axs[1].plot(time_vector_free[-n_shooting[-1]-1::], data_free["contact_forces"][-1][1, :], color="tab:green")
axs[0].plot(time_vector_without[-n_shooting[-1]-1::], data_without["contact_forces"][-1][0, :], color="tab:blue")
axs[1].plot(time_vector_without[-n_shooting[-1]-1::], data_without["contact_forces"][-1][1, :], color="tab:blue")
axs[0].plot(time_vector_CL[-n_shooting[-1]-1::], data_CL["contact_forces"][-1][0, :], color="tab:orange")
axs[1].plot(time_vector_CL[-n_shooting[-1]-1::], data_CL["contact_forces"][-1][1, :], color="tab:orange")
plot_all_lines(time_end_phase_CL, time_end_phase_without, time_end_phase_free, axs[0])
plot_all_lines(time_end_phase_CL, time_end_phase_without, time_end_phase_free, axs[1])
axs[0].legend()
axs[0].set_title("Horizontal contact force [N]")
axs[1].set_title("Vertical contact force [N]")
plt.savefig("Contact_forces" + "." + format_graph, format=format_graph, dpi=300)
plt.show()