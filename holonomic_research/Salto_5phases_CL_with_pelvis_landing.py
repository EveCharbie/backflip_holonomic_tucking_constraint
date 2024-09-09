"""
The aim of this code is to test the holonomic constraint of the flight phase
with the pelvis and with the landing phase.
The simulation have 5 phases: propulsion, flight phase, tucked phase, preparation landing, landing.
We also want to see how well the transition between phases with and without holonomic constraints works.

Phase 0: Propulsion
- Dynamic(s): TORQUE_DRIVEN with contact
- Constraint(s): 1 contact, no holonomic constraints
- Objective(s) function(s): minimize torque and time

Phase 1: Flight
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): zero contact, no holonomic constraints
- Objective(s) function(s): minimize torque and time

Phase 2: Tucked phase
- Dynamic(s): TORQUE_DRIVEN with holonomic constraints
- Constraint(s): zero contact, 1 holonomic constraints body-body
- Objective(s) function(s): minimize torque and time

Phase 3: Preparation landing
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): zero contact, no holonomic constraints
- Objective(s) function(s): minimize torque and time

Phase 4: Landing
- Dynamic(s): TORQUE_DRIVEN with contact
- Constraint(s): 2 contact, no holonomic constraints
- Objective(s) function(s): minimize torque and time
"""

# --- Import package --- #
import numpy as np
import casadi as cas
import pickle
from bioptim import (
    BiorbdModel,
    Node,
    InterpolationType,
    OptimalControlProgram,
    ConstraintList,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    PhaseTransitionList,
    DynamicsFcn,
    BiMappingList,
    ConstraintFcn,
    BoundsList,
    InitialGuessList,
    Solver,
    Axis,
    SolutionMerge,
    PenaltyController,
    PhaseTransitionFcn,
    DynamicsFunctions,
    HolonomicConstraintsList,
    HolonomicConstraintsFcn,
)
from casadi import MX, vertcat
from holonomic_research.biorbd_model_holonomic_updated import BiorbdModelCustomHolonomic
#from visualisation import visualisation_closed_loop_5phases_reception, visualisation_movement, graph_all
from Save import get_created_data_from_pickle
from Salto_5phases_with_pelvis_landing import CoM_over_toes, add_objectives, actuator_function
from plot_actuators import Joint

# --- Save results --- #
def save_results_holonomic(sol, c3d_file_path, biomodel, index_holo):
    """
    Solving the ocp
    Parameters
     ----------
     sol: Solution
        The solution to the ocp at the current pool
    c3d_file_path: str
        The path to the c3d file of the task
    """

    data = {}
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    list_time = sol.decision_time(to_merge=SolutionMerge.NODES)

    q = []
    qdot = []
    qddot =[]
    tau = []
    time = []
    min_bounds_q = []
    max_bounds_q = []
    min_bounds_qdot = []
    max_bounds_qdot = []
    min_bounds_tau = []
    max_bounds_tau = []

    if len(sol.ocp.n_shooting) == 1:
        q = states["q_u"]
        qdot = states["q_udot"]
        tau = controls["tau"]
    else:
        for i in range(len(states)):
            if i == index_holo:
                q_holo, qdot_holo, qddot_holo, lambdas = BiorbdModelCustomHolonomic.compute_all_states(
                    biomodel[index_holo], sol, index_holo)
                q.append(q_holo)
                qdot.append(qdot_holo)
                qddot.append(qddot_holo)
                tau.append(controls[i]["tau"])
                data["lambda"] = lambdas
                time.append(list_time[i])
                min_bounds_q.append(sol.ocp.nlp[i].x_bounds['q_u'].min)
                max_bounds_q.append(sol.ocp.nlp[i].x_bounds['q_u'].max)
                min_bounds_qdot.append(sol.ocp.nlp[i].x_bounds['qdot_u'].min)
                max_bounds_qdot.append(sol.ocp.nlp[i].x_bounds['qdot_u'].max)
                min_bounds_tau.append(sol.ocp.nlp[i].u_bounds["tau"].min)
                max_bounds_tau.append(sol.ocp.nlp[i].u_bounds["tau"].max)
            else:
                q.append(states[i]["q"])
                qdot.append(states[i]["qdot"])
                tau.append(controls[i]["tau"])
                time.append(list_time[i])
                min_bounds_q.append(sol.ocp.nlp[i].x_bounds['q'].min)
                max_bounds_q.append(sol.ocp.nlp[i].x_bounds['q'].max)
                min_bounds_qdot.append(sol.ocp.nlp[i].x_bounds['qdot'].min)
                max_bounds_qdot.append(sol.ocp.nlp[i].x_bounds['qdot'].max)
                min_bounds_tau.append(sol.ocp.nlp[i].u_bounds["tau"].min)
                max_bounds_tau.append(sol.ocp.nlp[i].u_bounds["tau"].max)

    data["q"] = q
    data["qdot"] = qdot
    data["qddot"] = qddot
    data["tau"] = tau
    data["time"] = time
    data["min_bounds_q"] = min_bounds_q
    data["max_bounds_q"] = max_bounds_q
    data["min_bounds_qdot"] = min_bounds_qdot
    data["max_bounds_qdot"] = max_bounds_qdot
    data["min_bounds_tau"] = min_bounds_q
    data["max_bounds_tau"] = max_bounds_q
    data["cost"] = sol.cost
    data["iterations"] = sol.iterations
    # data["detailed_cost"] = sol.add_detailed_cost
    data["status"] = sol.status
    data["real_time_to_optimize"] = sol.real_time_to_optimize
    data["phase_time"] = sol.phases_dt
    data["constraints"] = sol.constraints
    data["n_shooting"] = sol.ocp.n_shooting
    data["lam_g"] = sol.lam_g
    data["lam_p"] = sol.lam_p
    data["lam_x"] = sol.lam_x
    data["phase_time"] = sol.ocp.phase_time
    data["dof_names"] = sol.ocp.nlp[0].dof_names
    data["q_all"] = np.hstack(data["q"])
    data["qdot_all"] = np.hstack(data["qdot"])
    data["qddot_all"] = np.hstack(data["qddot"])
    data["tau_all"] = np.hstack(data["tau"])
    time_end_phase = []
    time_total = 0
    time_all = []
    for i in range(len(data["time"])):
        time_all.append(data["time"][i] + time_total)
        time_total = time_total + data["time"][i][-1]
        time_end_phase.append(time_total)
    data["time_all"] = np.vstack(time_all)
    data["time_total"] = time_total
    data["time_end_phase"] = time_end_phase

    if sol.status == 1:
        data["status"] = "Optimal Control Solution Found"
    else:
        data["status"] = "Restoration Failed !"

    with open(f"{c3d_file_path}", "wb") as file:
        pickle.dump(data, file)


def custom_phase_transition_pre(
        controllers: list[PenaltyController, PenaltyController]) -> MX:
    """
    The constraint of the transition. The values from the end of the phase to the next are multiplied by coef to
    determine the transition.

    Parameters
    ----------
    controllers: list[PenaltyController, PenaltyController]
        The controller for all the nodes in the penalty

    Returns
    -------
    The constraint such that: c(x) = 0
    """

    # Take the values of q of the BioMod without holonomics constraints
    states_pre = controllers[0].states.cx

    nb_independent = controllers[1].model.nb_independent_joints
    u_post = controllers[1].states.cx[:nb_independent]
    udot_post = controllers[1].states.cx[nb_independent:]

    # Take the q of the indepente joint and calculate the q of dependent joint
    v_post = controllers[1].model.compute_v_from_u_explicit_symbolic(u_post)
    q_post = controllers[1].model.state_from_partition(u_post, v_post)

    Bvu = controllers[1].model.coupling_matrix(q_post)
    vdot_post = Bvu @ udot_post
    qdot_post = controllers[1].model.state_from_partition(udot_post, vdot_post)

    states_post = vertcat(q_post, qdot_post)

    return states_pre - states_post


def custom_phase_transition_post(
        controllers: list[PenaltyController, PenaltyController]) -> MX:
    """
    The constraint of the transition. The values from the end of the phase to the next are multiplied by coef to
    determine the transition.

    Parameters
    ----------
    controllers: list[PenaltyController, PenaltyController]
        The controller for all the nodes in the penalty

    Returns
    -------
    The constraint such that: c(x) = 0
    """

    # Take the values of q of the BioMod without holonomics constraints
    nb_independent = controllers[0].model.nb_independent_joints
    u_pre = controllers[0].states.cx[:nb_independent]
    udot_pre = controllers[0].states.cx[nb_independent:]

    # Take the q of the indepente joint and calculate the q of dependent joint
    v_pre = controllers[0].model.compute_v_from_u_explicit_symbolic(u_pre)
    q_pre = controllers[0].model.state_from_partition(u_pre, v_pre)
    Bvu = controllers[0].model.coupling_matrix(q_pre)
    vdot_pre = Bvu @ udot_pre
    qdot_pre = controllers[0].model.state_from_partition(udot_pre, vdot_pre)

    states_pre = vertcat(q_pre, qdot_pre)
    states_post = controllers[1].states.cx

    return states_pre - states_post


def custom_contraint_lambdas_normal(
        controller: PenaltyController, bio_model: BiorbdModelCustomHolonomic) -> MX:

    # Recuperer les q
    q_u = controller.states["q_u"].cx
    qdot_u = controller.states["qdot_u"].cx
    tau = controller.controls["tau"].cx
    pelvis_mx = MX.zeros(3)
    new_tau = vertcat(pelvis_mx, tau)

    # Calculer lambdas
    lambdas = bio_model.compute_the_lagrangian_multipliers(q_u, qdot_u, new_tau)

    # Contrainte lagrange_0 (min_bound = -1, max_bound = 1)
    lagrange_0 = lambdas[0]

    return lagrange_0


def custom_contraint_lambdas_cisaillement_1(
        controller: PenaltyController, bio_model: BiorbdModelCustomHolonomic) -> MX:
    # Recuperer les q
    q_u = controller.states["q_u"].cx
    qdot_u = controller.states["qdot_u"].cx
    tau = controller.controls["tau"].cx
    pelvis_mx = MX.zeros(3)
    new_tau = vertcat(pelvis_mx, tau)

    # Calculer lambdas
    lambdas = bio_model.compute_the_lagrangian_multipliers(q_u, qdot_u, new_tau)

    # Contrainte lagrange_0 (min_bound = -1, max_bound = 1)
    lagrange_0 = lambdas[0]

    # Contrainte lagrange_1 (min_bound = L_1/L_0 = -0.2, max_bound = L_1/L_0 = 0.2)
    lagrange_1 = lambdas[1]

    return lagrange_1 - lagrange_0


def custom_contraint_lambdas_cisaillement_2(
        controller: PenaltyController, bio_model: BiorbdModelCustomHolonomic) -> MX:
    # Recuperer les q
    q_u = controller.states["q_u"].cx
    qdot_u = controller.states["qdot_u"].cx
    tau = controller.controls["tau"].cx
    pelvis_mx = MX.zeros(3)
    new_tau = vertcat(pelvis_mx, tau)

    # Calculer lambdas
    lambdas = bio_model.compute_the_lagrangian_multipliers(q_u, qdot_u, new_tau)

    # Contrainte lagrange_0 (min_bound = -1, max_bound = 1)
    lagrange_0 = lambdas[0]

    # Contrainte lagrange_1 (min_bound = L_1/L_0 = -0.2, max_bound = L_1/L_0 = 0.2)
    lagrange_1 = lambdas[1]

    return lagrange_1 - 0.01 * lagrange_0

def minimize_actuator_torques_CL(controller: PenaltyController, actuators) -> cas.MX:

    nb_independent = controller.model.nb_independent_joints
    u = controller.states.cx[:nb_independent]
    v = controller.model.compute_v_from_u_explicit_symbolic(u)
    q = controller.model.state_from_partition(u, v)

    tau = controller.controls["tau"].cx_start
    out = 0
    for i, key in enumerate(actuators.keys()):
        current_max_tau = cas.if_else(
            tau[i] > 0,
            actuator_function(actuators[key].tau_max_plus, actuators[key].theta_opt_plus, actuators[key].r_plus, q[i+3]),
            actuator_function(actuators[key].tau_max_minus, actuators[key].theta_opt_minus, actuators[key].r_minus, q[i+3]))
        out += (tau[i] / current_max_tau)**2
    return cas.sum1(out)


# --- Parameters --- #
movement = "Salto_close_loop_landing"
version = "Eve2"
nb_phase = 5
name_folder_model = "../Model"
# pickle_sol_init = "/home/mickaelbegon/Documents/Anais/Results_simu/Salto_close_loop_landing_5phases_V76.pkl"
# pickle_sol_init = "/home/mickaelbegon/Documents/Anais/Results_simu/Salto_5phases_V13.pkl"
pickle_sol_init = "init/Salto_5phases_V13.pkl"
sol_salto = get_created_data_from_pickle(pickle_sol_init)

# --- Prepare ocp --- #
def prepare_ocp(biorbd_model_path, phase_time, n_shooting, min_bound, max_bound):
    bio_model = (BiorbdModel(biorbd_model_path[0]),
                 BiorbdModel(biorbd_model_path[1]),
                 BiorbdModelCustomHolonomic(biorbd_model_path[2]),
                 BiorbdModel(biorbd_model_path[3]),
                 BiorbdModel(biorbd_model_path[4]),
                 )
    # Actuators parameters

    actuators = {"Shoulders": Joint(tau_max_plus=112.8107 * 2,
                                    theta_opt_plus=-41.0307 * np.pi / 180,
                                    r_plus=109.6679 * np.pi / 180,
                                    tau_max_minus=162.7655 * 2,
                                    theta_opt_minus=-101.6627 * np.pi / 180,
                                    r_minus=103.9095 * np.pi / 180,
                                    min_q=-0.7,
                                    max_q=3.1),
                 "Elbows": Joint(tau_max_plus=100 * 2,
                                 theta_opt_plus=np.pi / 2 - 0.1,
                                 r_plus=40 * np.pi / 180,
                                 tau_max_minus=50 * 2,
                                 theta_opt_minus=np.pi / 2 - 0.1,
                                 r_minus=70 * np.pi / 180,
                                 min_q=0,
                                 max_q=2.09),
                 # this one was not measured, I just tried to fit https://www.researchgate.net/figure/Maximal-isometric-torque-angle-relationship-for-elbow-extensors-fitted-by-polynomial_fig3_286214602
                 "Hips": Joint(tau_max_plus=220.3831 * 2,
                               theta_opt_plus=25.6939 * np.pi / 180,
                               r_plus=56.4021 * np.pi / 180,
                               tau_max_minus=490.5938 * 2,
                               theta_opt_minus=72.5836 * np.pi / 180,
                               r_minus=48.6999 * np.pi / 180,
                               min_q=-0.4,
                               max_q=2.6),
                 "Knees": Joint(tau_max_plus=367.6643 * 2,
                                theta_opt_plus=-61.7303 * np.pi / 180,
                                r_plus=31.7218 * np.pi / 180,
                                tau_max_minus=177.9694 * 2,
                                theta_opt_minus=-33.2908 * np.pi / 180,
                                r_minus=57.0370 * np.pi / 180,
                                min_q=-2.3,
                                max_q=0.02),
                 "Ankles": Joint(tau_max_plus=153.8230 * 2,
                                 theta_opt_plus=0.7442 * np.pi / 180,
                                 r_plus=58.9832 * np.pi / 180,
                                 tau_max_minus=171.9903 * 2,
                                 theta_opt_minus=12.6824 * np.pi / 180,
                                 r_minus=21.8717 * np.pi / 180,
                                 min_q=-0.7,
                                 max_q=0.7)
                 }


    tau_min_total = [0, 0, 0, -325.531, -138, -981.1876, -735.3286, -343.9806]
    tau_max_total = [0, 0, 0, 325.531, 138, 981.1876, 735.3286, 343.9806]
    #tau_min_total = [0, 0, 0, -162.7655, -69, -490.5938, -367.6643, -171.9903]
    #tau_max_total = [0, 0, 0, 162.7655, 69, 490.5938, 367.6643, 171.9903]
    tau_min = [i * 0.7 for i in tau_min_total]
    tau_max = [i * 0.7 for i in tau_max_total]
    tau_init = 0
    variable_bimapping = BiMappingList()
    dof_mapping = BiMappingList()
    variable_bimapping.add("q", to_second=[0, 1, 2, None, None, 3, 4, 5], to_first=[0, 1, 2, 5, 6, 7])
    variable_bimapping.add("qdot", to_second=[0, 1, 2, None, None, 3, 4, 5], to_first=[0, 1, 2, 5, 6, 7])
    dof_mapping.add("tau", to_second=[None, None, None, 0, 1, 2, 3, 4], to_first=[3, 4, 5, 6, 7])

    # --- Objectives functions ---#
    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions = add_objectives(objective_functions, actuators)
    objective_functions.add(
        minimize_actuator_torques_CL,
        custom_type=ObjectiveFcn.Lagrange,
        actuators=actuators,
        quadratic=False,
        weight=0.1,
        phase=2,
    )

    # --- Dynamics ---#
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, with_contact=True, phase=0)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, phase=1)
    dynamics.add(
        DynamicsFcn.HOLONOMIC_TORQUE_DRIVEN,
        phase=2
    )

    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, phase=3)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, with_contact=True, phase=4)

    # Transition de phase
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(custom_phase_transition_pre, phase_pre_idx=1)
    phase_transitions.add(custom_phase_transition_post, phase_pre_idx=2)
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=3)

    # --- Constraints ---#
    # Constraints
    constraints = ConstraintList()
    holonomic_constraints = HolonomicConstraintsList()

    # Phase 0 (Propulsion):
    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        marker_index="Foot_Toe_marker",
        axes=Axis.Z,
        max_bound=0,
        min_bound=0,
        node=Node.START,
        phase=0,
    )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        marker_index="Foot_Toe_marker",
        axes=Axis.Y,
        max_bound=0.1,
        min_bound=-0.1,
        node=Node.START,
        phase=0,
    )

    #constraints.add(
    #    ConstraintFcn.TRACK_MARKERS,
    #    marker_index="Shoulder_marker",
    #    axes=[Axis.X, Axis.Y, Axis.Z],
    #    max_bound=[0 + 0.10, 0.019 + 0.10, 1.149 + 0.10],
    #    min_bound=[0 - 0.10, 0.019 - 0.10, 1.149 - 0.10],
    #    node=Node.START,
    #    phase=0,
    #)

    #constraints.add(
    #    ConstraintFcn.TRACK_MARKERS,
    #    marker_index="KNEE_marker",
    #    axes=[Axis.X, Axis.Y, Axis.Z],
    #    max_bound=[0 + 0.10, 0.254 + 0.10, 0.413 + 0.10],
    #    min_bound=[0 - 0.10, 0.254 - 0.10, 0.413 - 0.10],
    #    node=Node.START,
    #    phase=0,
    #)

    constraints.add(
        CoM_over_toes,
        node=Node.START,
        phase=0,
    )

    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL_SHOOTING,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.5,
        phase=0,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=0.01,
        max_bound=np.inf,
        node=Node.ALL_SHOOTING,
        contact_index=1,
        phase=0,
    )

    # Phase 2 (Tucked phase):
    holonomic_constraints.add(
        "holonomic_constraints",
        HolonomicConstraintsFcn.superimpose_markers,
        biorbd_model=bio_model[2],
        marker_1="BELOW_KNEE",
        marker_2="CENTER_HAND",
        index=slice(1, 3),
        local_frame_index=11,
        #phase=2
    )
    # Made up constraints

    bio_model[2].set_holonomic_configuration(
        constraints_list=holonomic_constraints, independent_joint_index=[0, 1, 2, 5, 6, 7],
        dependent_joint_index=[3, 4],
    )

    constraints.add(
        custom_contraint_lambdas_cisaillement_1,
        node=Node.ALL_SHOOTING,
        bio_model=bio_model[2],
        max_bound=0,
        min_bound=-np.inf,
        phase=2,
    )

    constraints.add(
        custom_contraint_lambdas_cisaillement_2,
        node=Node.ALL_SHOOTING,
        bio_model=bio_model[2],
        max_bound=np.inf,
        min_bound=0,
        phase=2,
    )

    constraints.add(
        custom_contraint_lambdas_normal,
        node=Node.ALL_SHOOTING,
        bio_model=bio_model[2],
        max_bound=np.inf,
        min_bound=1,
        phase=2,
    )

    # Phase 4 (Landing):

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=0.01,
        max_bound=np.inf,
        node=Node.ALL_SHOOTING,
        contact_index=1,
        phase=4,
    )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        marker_index="Foot_Toe_marker",
        axes=Axis.Z,
        max_bound=0,
        min_bound=0,
        node=Node.END,
        phase=4,
    )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        marker_index="Foot_Toe_marker",
        axes=Axis.Y,
        max_bound=0.1,
        min_bound=-0.1,
        node=Node.END,
        phase=4,
    )

    constraints.add(
        CoM_over_toes,
        node=Node.END,
        phase=4,
    )

    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL_SHOOTING,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.5,
        phase=4,
    )

    # Path constraint
    pose_propulsion_start = [-0.2343, -0.2177, -0.3274, 0.2999, 0.4935, 1.7082, -1.9999, 0.1692]
    pose_takeout_start = [-0.1233, 0.22, 0.3173, 1.5707, 0.1343, -0.2553, -0.1913, -0.342]
    pose_salto_start = [0.135, 0.455, 1.285, 0.481, 1.818, 2.6, -1.658, 0.692]
    pose_salto_end = [0.107, 0.797, 2.892, 0.216, 1.954, 2.599, -2.058, 0.224]
    pose_salto_start_CL = [0.135, 0.455, 1.285, 2.6, -1.658, 0.692]
    pose_salto_end_CL = [0.107, 0.797, 2.892, 2.599, -2.058, 0.224]
    pose_landing_start = [0.013, 0.088, 5.804, -0.305, 8.258444956622276e-06, 1.014, -0.97, 0.006]
    pose_landing_end = [0.053, 0.091, 6.08, 2.9, -0.17, 0.092, 0.17, 0.20]


    # --- Bounds ---#
    # Initialize x_bounds
    n_q = bio_model[0].nb_q
    n_qdot = n_q
    n_independent = bio_model[2].nb_independent_joints

    # Phase 0: Propulsion
    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot"), phase=0)
    x_bounds[0]["q"].min[2:7, 0] = np.array(pose_propulsion_start[2:7]) - 0.3
    x_bounds[0]["q"].max[2:7, 0] = np.array(pose_propulsion_start[2:7]) + 0.3
    x_bounds[0]["q"].max[2, 0] = 0.5
    x_bounds[0]["q"].max[5, 0] = 2
    x_bounds[0]["q"].min[6, 0] = -2
    x_bounds[0]["q"].max[6, 0] = -0.7
    x_bounds[0]["qdot"][:, 0] = [0] * n_qdot
    x_bounds[0]["qdot"].max[5, :] = 0
    x_bounds[0]["q"].min[2, 1:] = -np.pi
    x_bounds[0]["q"].max[2, 1:] = np.pi
    x_bounds[0]["q"].min[0, :] = -1
    x_bounds[0]["q"].max[0, :] = 1
    x_bounds[0]["q"].min[1, :] = -1
    x_bounds[0]["q"].max[1, :] = 2
    x_bounds[0]["qdot"].min[3, :] = 0   # A commenter si marche pas
    x_bounds[0]["q"].min[3, 2] = np.pi/2


    # Phase 1: Flight
    x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot"), phase=1)
    #x_bounds[1]["q"].min[3:, 0] = np.array(pose_takeout_start[3:]) - 0.3 # 0.03
    #x_bounds[1]["q"].max[3:, 0] = np.array(pose_takeout_start[3:]) + 0.3
    x_bounds[1]["q"].min[0, :] = -1
    x_bounds[1]["q"].max[0, :] = 1
    x_bounds[1]["q"].min[1, :] = -1
    x_bounds[1]["q"].max[1, :] = 2
    x_bounds[1]["q"].min[2, 0] = -np.pi
    x_bounds[1]["q"].max[2, 0] = np.pi * 1.5
    x_bounds[1]["q"].min[2, 1] = -np.pi
    x_bounds[1]["q"].max[2, 1] = np.pi * 1.5
    x_bounds[1]["q"].min[2, -1] = -np.pi
    x_bounds[1]["q"].max[2, -1] = np.pi * 1.5
    #x_bounds[1]["q"].min[4, -1] = 1


    # Phase 2: Tucked phase
    x_bounds.add("q_u", bounds=bio_model[2].bounds_from_ranges("q", mapping=variable_bimapping), phase=2)
    x_bounds.add("qdot_u", bounds=bio_model[2].bounds_from_ranges("qdot", mapping=variable_bimapping), phase=2)
    x_bounds[2]["q_u"].min[0, :] = - 1
    x_bounds[2]["q_u"].max[0, :] = 1
    x_bounds[2]["q_u"].min[1, :] = -1
    x_bounds[2]["q_u"].max[1, :] = 2
    x_bounds[2]["q_u"].min[2, 0] = -np.pi
    x_bounds[2]["q_u"].max[2, 0] = np.pi * 1.5
    x_bounds[2]["q_u"].min[2, 1] = -np.pi
    x_bounds[2]["q_u"].max[2, 1] = 2 * np.pi
    x_bounds[2]["q_u"].min[2, 2] = 3/4 * np.pi
    x_bounds[2]["q_u"].max[2, 2] = 3/2 * np.pi
    x_bounds[2]["q_u"].max[3, :-1] = 2.6
    x_bounds[2]["q_u"].min[3, :-1] = 1.96
    x_bounds[2]["q_u"].max[4, :-1] = -1.5
    x_bounds[2]["q_u"].min[4, :-1] = -2.3

    # Phase 3: Preparation landing
    x_bounds.add("q", bounds=bio_model[3].bounds_from_ranges("q"), phase=3)
    x_bounds.add("qdot", bounds=bio_model[3].bounds_from_ranges("qdot"), phase=3)
    x_bounds[3]["q"].min[0, :] = -1
    x_bounds[3]["q"].max[0, :] = 1
    x_bounds[3]["q"].min[1, 1:] = -1
    x_bounds[3]["q"].max[1, 1:] = 2
    x_bounds[3]["q"].min[2, :] = 3/4 * np.pi
    x_bounds[3]["q"].max[2, :] = 2 * np.pi + 0.5
    x_bounds[3]["q"].min[5, -1] = pose_landing_start[5] - 1 #0.06
    x_bounds[3]["q"].max[5, -1] = pose_landing_start[5] + 0.5
    x_bounds[3]["q"].min[6, -1] = pose_landing_start[6] - 1
    x_bounds[3]["q"].max[6, -1] = pose_landing_start[6] + 0.1

    # Phase 3: Landing
    x_bounds.add("q", bounds=bio_model[4].bounds_from_ranges("q"), phase=4)
    x_bounds.add("qdot", bounds=bio_model[4].bounds_from_ranges("qdot"), phase=4)
    x_bounds[4]["q"].min[5, 0] = pose_landing_start[5] - 1 #0.06
    x_bounds[4]["q"].max[5, 0] = pose_landing_start[5] + 0.5
    x_bounds[4]["q"].min[6, 0] = pose_landing_start[6] - 1
    x_bounds[4]["q"].max[6, 0] = pose_landing_start[6] + 0.1
    x_bounds[4]["q"].min[0, :] = -1
    x_bounds[4]["q"].max[0, :] = 1
    x_bounds[4]["q"].min[1, :] = -1
    x_bounds[4]["q"].max[1, :] = 2
    x_bounds[4]["q"].min[2, 0] = 2/4 * np.pi
    x_bounds[4]["q"].max[2, 0] = 2 * np.pi + 1.66
    x_bounds[4]["q"].min[2, 1] = 2/4 * np.pi
    x_bounds[4]["q"].max[2, 1] = 2 * np.pi + 1.66
    x_bounds[4]["q"].max[:, -1] = np.array(pose_landing_end) + 0.2 #0.5
    x_bounds[4]["q"].min[:, -1] = np.array(pose_landing_end) - 0.2
    #x_bounds[4]["qdot"][:, -1] = [0] * n_qdot
    #x_bounds[4]["q"][0, -1] = np.array(pose_propulsion_start[0])
    #x_bounds[4]["q"][7, -1] = np.array(pose_landing_end[7])

    # Initial guess
    x_init = InitialGuessList()
    #x_init.add("q", np.array([pose_propulsion_start, pose_takeout_start]).T, interpolation=InterpolationType.LINEAR,
    #           phase=0)
    #x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=0)
    #x_init.add("q", np.array([pose_takeout_start, pose_salto_start]).T, interpolation=InterpolationType.LINEAR,
    #           phase=1)
    #x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=1)
    #x_init.add("q_u", np.array([pose_salto_start_CL, pose_salto_end_CL]).T,
    #           interpolation=InterpolationType.LINEAR, phase=2)
    #x_init.add("qdot_u", np.array([[0] * n_independent, [0] * n_independent]).T,
    #           interpolation=InterpolationType.LINEAR, phase=2)
    #x_init.add("q", np.array([pose_salto_end, pose_landing_start]).T, interpolation=InterpolationType.LINEAR, phase=3)
    #x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=3)
    #x_init.add("q", np.array([pose_landing_start, pose_landing_end]).T, interpolation=InterpolationType.LINEAR, phase=4)
    #x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=4)

    x_init.add("q", sol_salto["q"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("qdot", sol_salto["qdot"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("q", sol_salto["q"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    x_init.add("qdot", sol_salto["qdot"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    x_init.add("q_u", sol_salto["q"][2][[0,1,2,5,6,7], :], interpolation=InterpolationType.EACH_FRAME, phase=2)
    x_init.add("qdot_u", sol_salto["qdot"][2][[0,1,2,5,6,7], :], interpolation=InterpolationType.EACH_FRAME, phase=2)
    x_init.add("q", sol_salto["q"][3], interpolation=InterpolationType.EACH_FRAME, phase=3)
    x_init.add("qdot", sol_salto["qdot"][3], interpolation=InterpolationType.EACH_FRAME, phase=3)
    x_init.add("q", sol_salto["q"][3], interpolation=InterpolationType.EACH_FRAME, phase=3)
    x_init.add("qdot", sol_salto["qdot"][3], interpolation=InterpolationType.EACH_FRAME, phase=3)
    x_init.add("q", sol_salto["q"][4], interpolation=InterpolationType.EACH_FRAME, phase=4)
    x_init.add("qdot", sol_salto["qdot"][4], interpolation=InterpolationType.EACH_FRAME, phase=4)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=0)
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=1)
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=2)
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=3)
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=4)

    u_init = InitialGuessList()
    #u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=0)
    #u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=1)
    #u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=2)
    #u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=3)
    #u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=4)
    
    u_init.add("tau", sol_salto["tau"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    u_init.add("tau", sol_salto["tau"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    u_init.add("tau", sol_salto["tau"][2], interpolation=InterpolationType.EACH_FRAME, phase=2)
    u_init.add("tau", sol_salto["tau"][3], interpolation=InterpolationType.EACH_FRAME, phase=3)
    u_init.add("tau", sol_salto["tau"][3], interpolation=InterpolationType.EACH_FRAME, phase=4)

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
        #assume_phase_dynamics=True,
        phase_transitions=phase_transitions,
        variable_mappings=dof_mapping,
    ), bio_model


# --- Load model --- #
def main():
    model_path = str(name_folder_model) + "/" + "Model2D_7Dof_0C_5M_CL_V3.bioMod"
    #model_path_2contact = str(name_folder_model) + "/" + "Model2D_7Dof_3C_5M_CL_V3.bioMod"
    model_path_1contact = str(name_folder_model) + "/" + "Model2D_7Dof_2C_5M_CL_V3.bioMod"

    ocp, bio_model = prepare_ocp(
        biorbd_model_path=(model_path_1contact,
                           model_path,
                           model_path,
                           model_path,
                           model_path_1contact),
        phase_time=(0.2, 0.2, 0.3, 0.3, 0.3),
        n_shooting=(20, 20, 30, 30, 30),
        min_bound=0.01,
        max_bound=np.inf,
    )

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_options=dict(show_bounds=True), _linear_solver="MA57")#show_online_optim=True,
    solver.set_maximum_iterations(10000)
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)
    #ocp.add_plot_penalty()
    sol = ocp.solve(solver)
    sol.print_cost()


# --- Save results --- #
    save_results_holonomic(sol, str(movement) + "_" + str(nb_phase) + "phases_V" + version + ".pkl", bio_model, 2)
    sol.graphs(show_bounds=True, save_name=str(movement) + "_" + str(nb_phase) + "phases_V" + str(version))
    sol.animate()

if __name__ == "__main__":
    main()
