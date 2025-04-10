import os

import numpy as np
import pickle
from plotly.colors import DEFAULT_PLOTLY_COLORS
from pyorerun import PhaseRerun, BiorbdModel
from pyorerun.multi_frame_rate_phase_rerun import MultiFrameRatePhaseRerun

folder = "backflip_Vpost_submission_collision_feb25/htc/"
model_path = "../models/Model2D_7Dof_2C_5M_CL_V3.bioMod"

# remove "rgb(" and ")" and split by ","
colors = [color[4:-1].split(",") for color in DEFAULT_PLOTLY_COLORS]
colors = [(int(color[0]), int(color[1]), int(color[2])) for color in colors]
colors = colors + colors + colors  # duplicate the colors to have enough for all the phases

only_converged = True

# Charger les données
phase_reruns = []
for i in range(0, 20):
    suffix = "CVG"
    if os.path.isfile(folder + f"/sol_{i}_CVG.pkl") is False:
        suffix = "DVG"
        if only_converged:
            continue
    data = pickle.load(open(folder + f"/sol_{i}_{suffix}.pkl", "rb"))
    time = np.concatenate([np.array(time) for time in data["time"]], axis=0).squeeze()
    q = np.concatenate([np.array(q) for q in data["q"]], axis=1)

    phase_reruns.append(PhaseRerun(t_span=time, window=f"simulation_{i}"))
    m = BiorbdModel(model_path)
    if suffix == "DVG":
        m.options.mesh_color = (0, 0, 0)
    else:
        m.options.mesh_color = colors[i - 1]
    phase_reruns[-1].add_animated_model(m, q)

mrr2 = MultiFrameRatePhaseRerun(phase_reruns=phase_reruns)
mrr2.rerun("all_multistart")
