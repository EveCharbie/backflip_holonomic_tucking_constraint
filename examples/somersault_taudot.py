"""
This script is used to solve the somersault problem with 5 phases and a pelvis landing.
- Taudot control
- no closed loop for the tucking phase
"""

# --- Import package --- #
import numpy as np
from bioptim import (
    BiorbdModel,
    InterpolationType,
    OptimalControlProgram,
    ConstraintList,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    PhaseTransitionList,
    DynamicsFcn,
    BiMappingList,
    BoundsList,
    InitialGuessList,
    Solver,
    PhaseTransitionFcn,
    MagnitudeType,
)
import matplotlib.pyplot as plt
from src.save_load_helpers import get_created_data_from_pickle
from src.bounds_x import add_x_bounds
from src.save_results import save_results_taudot, save_sol_no_ocp
from src.objectives import add_objectives, minimize_actuator_torques, add_taudot_objectives
from src.constants import (
    JUMP_INIT_PATH,
    POSE_TUCKING_START,
    POSE_TUCKING_END,
    POSE_LANDING_START,
)

from src.constraints import add_constraints
from src.actuator_constants import ACTUATORS, initialize_tau
from src.multistart import prepare_multi_start
from src.phase_transitions import custom_takeoff
from src.objectives import WEIGHTS
from src.initial_guess_utils import interpolate_array
from .constants import MODEL_PATHS, PHASE_TIME, N_SHOOTING

# --- Prepare ocp --- #
def prepare_ocp(biorbd_model_path, phase_time, n_shooting, WITH_MULTI_START, seed=0):
    bio_model = (
        BiorbdModel(biorbd_model_path[0]),
        BiorbdModel(biorbd_model_path[1]),
        BiorbdModel(biorbd_model_path[2]),
        BiorbdModel(biorbd_model_path[3]),
        BiorbdModel(biorbd_model_path[4]),
    )

    n_q = bio_model[0].nb_q
    n_qdot = n_q

    # Actuators parameters
    actuators = ACTUATORS

    tau_min, tau_max, tau_init = initialize_tau()
    dof_mapping = BiMappingList()
    dof_mapping.add("tau", to_second=[None, None, None, 0, 1, 2, 3, 4], to_first=[3, 4, 5, 6, 7])
    dof_mapping.add("taudot", to_second=[None, None, None, 0, 1, 2, 3, 4], to_first=[3, 4, 5, 6, 7])

    # --- Objectives functions ---#
    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions = add_objectives(objective_functions, actuators)
    objective_functions = add_taudot_objectives(objective_functions)
    objective_functions.add(
        minimize_actuator_torques,
        custom_type=ObjectiveFcn.Lagrange,
        actuators=actuators,
        quadratic=True,
        weight=WEIGHTS["2_TORQUE_RATIO"],
        phase=2,
    )

    # --- Dynamics ---#
    dynamics = DynamicsList()
    dynamics.add(
        DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN, expand_dynamics=True, expand_continuity=False, with_contact=True, phase=0
    )
    dynamics.add(DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN, expand_dynamics=True, expand_continuity=False, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN, expand_dynamics=True, expand_continuity=False, phase=2)
    dynamics.add(DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN, expand_dynamics=True, expand_continuity=False, phase=3)
    dynamics.add(
        DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN, expand_dynamics=True, expand_continuity=False, with_contact=True, phase=4
    )

    # Transition de phase
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(custom_takeoff, phase_pre_idx=0)
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=3)

    # --- Constraints ---#
    constraints = ConstraintList()
    constraints = add_constraints(constraints)

    # --- Bounds ---#
    x_bounds = BoundsList()
    q_bounds, qdot_bounds = add_x_bounds(bio_model)
    for i_phase in range(len(bio_model)):
        x_bounds.add("q", bounds=q_bounds[i_phase], phase=i_phase)
        x_bounds.add("qdot", bounds=qdot_bounds[i_phase], phase=i_phase)
        x_bounds.add(
            "tau",
            min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
            max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
            phase=i_phase,
        )

    # Initial guess
    sol_salto = get_created_data_from_pickle(JUMP_INIT_PATH)
    x_init = InitialGuessList()
    # Initial guess from Jump

    x_init.add("q", interpolate_array(sol_salto["q"][0],n_shooting[0]+1), interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("qdot", interpolate_array(sol_salto["qdot"][0],n_shooting[0]+1), interpolation=InterpolationType.EACH_FRAME, phase=0)

    x_init.add("q", sol_salto["q"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    x_init.add("qdot", sol_salto["qdot"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    x_init.add("q", np.array([POSE_TUCKING_START, POSE_TUCKING_END]).T, interpolation=InterpolationType.LINEAR, phase=2)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=2)
    x_init.add("q", np.array([POSE_TUCKING_END, POSE_LANDING_START]).T, interpolation=InterpolationType.LINEAR, phase=3)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=3)
    x_init.add("q", POSE_LANDING_START, interpolation=InterpolationType.CONSTANT, phase=4)
    x_init.add("qdot", np.zeros(n_q), interpolation=InterpolationType.CONSTANT, phase=4)

    # Tau initial guess
    x_init.add(
        "tau",
        interpolate_array(np.hstack((sol_salto["tau"][0], np.zeros((5, 1)))), n_shooting[0] + 1),
        interpolation=InterpolationType.EACH_FRAME,
        phase=0,
    )
    x_init.add(
        "tau",
        interpolate_array(np.hstack((sol_salto["tau"][1], np.zeros((5, 1)))), n_shooting[1] + 1),
        interpolation=InterpolationType.EACH_FRAME,
        phase=1,
    )
    x_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=2)
    x_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=3)
    x_init.add(
        "tau",
        interpolate_array(np.hstack((sol_salto["tau"][3], np.zeros((5, 1)))), n_shooting[4] + 1),
        interpolation=InterpolationType.EACH_FRAME,
        phase=4
    )

    # Define control path constraint
    u_bounds = BoundsList()
    u_init = InitialGuessList()
    for i_phase in range(len(bio_model)):
        u_bounds.add("taudot", min_bound=[-10000] * 5, max_bound=[10000] * 5, phase=i_phase)
        u_init.add("taudot", [0] * 5, phase=i_phase)

    if WITH_MULTI_START:
        x_init.add_noise(
            bounds=x_bounds,
            magnitude=0.2,
            magnitude_type=MagnitudeType.RELATIVE,
            n_shooting=[n_shooting[i] + 1 for i in range(len(n_shooting))],
            seed=seed,
        )
        u_init.add_noise(
            bounds=u_bounds,
            magnitude=0.1,
            magnitude_type=MagnitudeType.RELATIVE,
            n_shooting=[n_shooting[i] for i in range(len(n_shooting))],
            seed=seed,
        )

    return OptimalControlProgram(
        bio_model=bio_model,
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=phase_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        n_threads=32,
        phase_transitions=phase_transitions,
        variable_mappings=dof_mapping,
    )


# --- Load model --- #
def main():
    # --- Parameters --- #
    movement = "Salto"
    version = "after_submission_ntc"

    WITH_MULTI_START = True
    save_folder = f"../results/{str(movement)}_V{version}"

    # Solver options
    solver = Solver.IPOPT(show_options=dict(show_bounds=True), _linear_solver="MA57", show_online_optim=False)
    solver.set_maximum_iterations(10000)
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)
    solver.set_tol(1e-6)

    if WITH_MULTI_START:

        combinatorial_parameters = {
            "bio_model_path": [MODEL_PATHS],
            "phase_time": [PHASE_TIME],
            "n_shooting": [N_SHOOTING],
            "WITH_MULTI_START": [True],
            "seed": list(range(0, 20)),
        }

        multi_start = prepare_multi_start(
            prepare_ocp,
            save_results_taudot,
            combinatorial_parameters=combinatorial_parameters,
            save_folder=save_folder,
            solver=solver,
            n_pools=1,
        )

        multi_start.solve()
    else:
        ocp = prepare_ocp(MODEL_PATHS, PHASE_TIME, N_SHOOTING, WITH_MULTI_START=False)
        # ocp.add_plot_penalty()

        solver.show_online_optim = False
        sol = ocp.solve(solver)
        sol.print_cost()

        # --- Save results --- #
        sol.animate(viewer="pyorerun")

        combinatorial_parameters = [MODEL_PATHS, PHASE_TIME, N_SHOOTING, False, "no_seed"]
        save_results_taudot(sol, *combinatorial_parameters, save_folder=save_folder)
        sol.graphs(show_bounds=True, save_name=str(movement) + "_V" + version, show_now=False)
        # NOTE: This will save the solution without the ocp, so the graphs cannot be generated after this line
        save_sol_no_ocp(sol, *combinatorial_parameters, save_folder=save_folder)
        # Showing the graphs
        plt.show()


if __name__ == "__main__":
    main()
