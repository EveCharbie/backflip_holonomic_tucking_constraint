from bioptim import (
    ObjectiveFcn,
    Node,
    Axis,
    PenaltyController,
)
import casadi as cas

from .actuators import actuator_function


def minimize_actuator_torques(controller: PenaltyController, actuators) -> cas.MX:
    q = controller.states["q"].cx_start

    if "tau" in controller.states:
        tau = controller.states["tau"].cx_start
    else:
        tau = controller.controls["tau"].cx_start

    out = 0
    for i, key in enumerate(actuators.keys()):
        current_max_tau = cas.if_else(
            tau[i] > 0,
            actuator_function(
                actuators[key].tau_max_plus, actuators[key].theta_opt_plus, actuators[key].r_plus, q[i + 3]
            ),
            actuator_function(
                actuators[key].tau_max_minus, actuators[key].theta_opt_minus, actuators[key].r_minus, q[i + 3]
            ),
        )
        out += (tau[i] / current_max_tau) ** 2
    return cas.sum1(out)


def minimize_actuator_torques_CL(controller: PenaltyController, actuators) -> cas.MX:

    nb_independent = controller.model.nb_independent_joints
    u = controller.states.cx[:nb_independent]
    v = controller.model.compute_v_from_u_explicit_symbolic(u)
    q = controller.model.state_from_partition(u, v)

    if "tau" in controller.states:
        tau = controller.states["tau"].cx_start
    else:
        tau = controller.controls["tau"].cx_start

    out = 0
    for i, key in enumerate(actuators.keys()):
        current_max_tau = cas.if_else(
            tau[i] > 0,
            actuator_function(
                actuators[key].tau_max_plus, actuators[key].theta_opt_plus, actuators[key].r_plus, q[i + 3]
            ),
            actuator_function(
                actuators[key].tau_max_minus, actuators[key].theta_opt_minus, actuators[key].r_minus, q[i + 3]
            ),
        )
        out += (tau[i] / current_max_tau) ** 2
    return cas.sum1(out)


WEIGHTS = {
    "0_COM_VELOCITY": -1,
    "0_CONTACT_FORCES": 5,
    "4_COM_VELOCITY": 25,
    "4_COM_POSITION": 100,
    "4_STATE": 100,
    "0_TIME": 1,
    "1_TIME": 1,
    "2_TIME": -10,
    "3_TIME": 1,
    "4_TIME": 1,
    "0_TORQUE_RATIO": 1,
    "1_TORQUE_RATIO": 1,
    "2_TORQUE_RATIO": 1,
    "3_TORQUE_RATIO": 1,
    "4_TORQUE_RATIO": 1,
    "TAUDOT": 1e-6,
    "TAU_DERIVATIVE": 1,
}


def add_objectives(objective_functions, actuators, weights: dict = None):

    if weights is None:
        weights = WEIGHTS

    # Phase 0 (Propulsion):
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=weights["0_COM_VELOCITY"], axes=Axis.Z, phase=0
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME,
        weight=weights["0_TIME"],
        min_bound=0.3,
        max_bound=0.5,
        phase=0,
    )
    objective_functions.add(
        minimize_actuator_torques,
        custom_type=ObjectiveFcn.Lagrange,
        actuators=actuators,
        quadratic=True,
        weight=weights["0_TORQUE_RATIO"],
        phase=0,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_CONTACT_FORCES,
        node=Node.END,
        weight=weights["0_CONTACT_FORCES"],
        contact_index=1,
        quadratic=True,
        phase=0,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_CONTACT_FORCES,
        node=Node.END,
        weight=weights["0_CONTACT_FORCES"],
        contact_index=0,
        quadratic=True,
        phase=0,
    )

    # Phase 1 (Flight):
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=weights["1_TIME"], min_bound=0.1, max_bound=0.3, phase=1
    )
    objective_functions.add(
        minimize_actuator_torques,
        custom_type=ObjectiveFcn.Lagrange,
        actuators=actuators,
        quadratic=True,
        weight=weights["1_TORQUE_RATIO"],
        phase=1,
    )

    # Phase 2 (Tucked phase):
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=weights["2_TIME"], min_bound=0.1, max_bound=0.4, phase=2
    )

    # Phase 3 (Preparation landing):
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=weights["3_TIME"], min_bound=0.1, max_bound=0.3, phase=3
    )
    objective_functions.add(
        minimize_actuator_torques,
        custom_type=ObjectiveFcn.Lagrange,
        actuators=actuators,
        quadratic=True,
        weight=weights["3_TORQUE_RATIO"],
        phase=3,
    )

    # Phase 4 (Landing):
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", node=Node.END, weight=weights["4_STATE"], phase=4
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=weights["4_COM_VELOCITY"], phase=4
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=weights["4_TIME"], min_bound=0.2, max_bound=1, phase=4
    )
    objective_functions.add(
        minimize_actuator_torques,
        custom_type=ObjectiveFcn.Lagrange,
        actuators=actuators,
        quadratic=True,
        weight=weights["4_TORQUE_RATIO"],
        phase=4,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=weights["4_COM_POSITION"], axes=Axis.Y, phase=4
    )

    return objective_functions


def add_tau_derivative_objectives(objective_functions, weights: dict = None):

    if weights is None:
        weights = WEIGHTS

    weight_tau_derivative = weights["TAU_DERIVATIVE"]

    # Phase 0 (Propulsion):
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=weight_tau_derivative, phase=0
    )

    # Phase 1 (Flight):
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=weight_tau_derivative, phase=1
    )

    # Phase 2 (Tucked phase):
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=weight_tau_derivative, phase=2
    )

    # Phase 3 (Preparation landing):
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=weight_tau_derivative, phase=3
    )

    # Phase 4 (Landing):
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=weight_tau_derivative, phase=4
    )

    return objective_functions


def add_taudot_objectives(objective_functions, weights: dict = None):
    if weights is None:
        weights = WEIGHTS

    weight_taudot = weights["TAUDOT"]

    # Phase 0 (Propulsion):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="taudot", weight=weight_taudot, phase=0)

    # Phase 1 (Flight):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="taudot", weight=weight_taudot, phase=1)

    # Phase 2 (Tucked phase):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="taudot", weight=weight_taudot, phase=2)

    # Phase 3 (Preparation landing):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="taudot", weight=weight_taudot, phase=3)

    # Phase 4 (Landing):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="taudot", weight=weight_taudot, phase=4)

    return objective_functions
