import os
import numpy as np
import pickle
import rerun as rr

from pyorerun import PhaseRerun, BiorbdModel
from pyorerun.multi_frame_rate_phase_rerun import MultiFrameRatePhaseRerun

model_path = "../models/Model2D_7Dof_3C_5M_CL_V3_less_markers.bioMod"
common_path = "backflip_Vpost_submission_collision_feb25/"
folder_KTC = common_path + "ktc/"
folder_HTC = common_path + "htc/"
folder_NTC = common_path + "ntc/"

#  get the data with the smallest cost
file_name = []
for config, (folder, str_suffix) in enumerate(zip([folder_KTC, folder_HTC, folder_NTC], ["KTC", "HTC", "NTC"])):
    files = [name for name in os.listdir(folder) if name.endswith("_CVG.pkl")]
    smallest_name, smallest_value = "0", np.inf
    for file in files:

        data = pickle.load(open(folder + file, "rb"))

        if data["cost"] < smallest_value:
            smallest_value = data["cost"]
            smallest_name= file
            print(f"New smallest value of {str_suffix} : {smallest_value} in file {smallest_name}")
    file_name.append(smallest_name)

# get matplotlib colors instead
colors = [
    # tab:blue
    (31, 119, 180),
    # tab:orange
    (255, 127, 14),
    # tab:green
    (44, 160, 44),
    # tab:red
    (214, 39, 40),
    # tab:purple
    (148, 103, 189),
    # tab:brown
    (140, 86, 75),
    # tab:pink
    (227, 119, 194),
    # tab:gray
    (127, 127, 127),
    # tab:olive
    (188, 189, 34),
    # tab:cyan
    (23, 190, 207),
]
colors = colors + colors + colors  # duplicate the colors to have enough for all the phases

# Charger les données
phase_reruns = []

for config, (file, folder, str_suffix) in enumerate(zip(file_name, [folder_KTC, folder_HTC, folder_NTC], ["KTC", "HTC", "NTC"])):

    data = pickle.load(open(folder + f"/{file}", "rb"))
    time = np.concatenate([np.array(time) for time in data["time"]], axis=0).squeeze()
    q = np.concatenate([np.array(q) for q in data["q"]], axis=1)
    # # offset
    q[0, :] = q[0, :] + (config - 1)

    phase_reruns.append(PhaseRerun(t_span=time, window=f"simulation_{str_suffix}_{config}"))
    m = BiorbdModel(model_path)
    m.options.mesh_color = colors[config - 1]
    phase_reruns[-1].add_animated_model(m, q)

mrr2 = MultiFrameRatePhaseRerun(phase_reruns=phase_reruns)
mrr2.rerun_by_frame("NTC_KTC_HTC")


rr.log(
    "world/cam",
    rr.Pinhole(
        fov_y=0.7853982,
        aspect_ratio=1.7777778,
        camera_xyz=rr.ViewCoordinates.FLU,
        #image_plane_distance=3,
    ),
)
rr.log(
    "world/cam",
    rr.Transform3D(
        translation=[-3, 0, 1.05],
    ),
)
