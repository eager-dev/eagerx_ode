# ROS packages required
from eagerx import Object, Bridge, initialize, log, process

# Environment
from eagerx.core.env import EagerxEnv
from eagerx.core.graph import Graph
from eagerx.wrappers import Flatten

# Implementation specific
import eagerx.nodes  # Registers butterworth_filter # noqa # pylint: disable=unused-import
import eagerx_ode  # Registers OdeBridge # noqa # pylint: disable=unused-import

import tests.pendulum.objects # Registers pendulum # noqa # pylint: disable=unused-import

# Other
import numpy as np

import pytest

NP = process.NEW_PROCESS
ENV = process.ENVIRONMENT

@pytest.mark.parametrize(
    "eps, steps, is_reactive, rtf, p",
    [(3, 3, True, 0, ENV)],
)
def test_ode_bridge(eps, steps, is_reactive, rtf, p):
    # Start roscore
    roscore = initialize("eagerx_core", anonymous=True, log_level=log.WARN)

    # Define unique name for test environment
    name = f"{eps}_{steps}_{is_reactive}_{p}"
    bridge_p = p
    rate = 30

    # Initialize empty graph
    graph = Graph.create()

    # Create pendulum
    pendulum = Object.make(
        "Eagerx_Ode_Pendulum",
        "pendulum",
        sensors=["pendulum_output", "action_applied", "image"],
        states=["model_state", "model_parameters"],
    )
    graph.add(pendulum)

    # Connect the nodes
    graph.connect(action="action", target=pendulum.actuators.pendulum_input)
    graph.connect(source=pendulum.sensors.pendulum_output, observation="observation", window=1)
    graph.connect(source=pendulum.sensors.action_applied, observation="action_applied", window=1)
    graph.render(pendulum.sensors.image, rate=10)

    # Define bridges
    bridge = Bridge.make("OdeBridge", rate=rate, is_reactive=is_reactive, real_time_factor=rtf, process=bridge_p)

    # Define step function
    def step_fn(prev_obs, obs, action, steps):
        # Calculate reward
        if len(obs["observation"][0]) == 2:
            th, thdot = obs["observation"][0]
            sin_th = np.sin(th)
            cos_th = np.cos(th)
        else:
            sin_th, cos_th, thdot = 0, -1, 0
        th = np.arctan2(sin_th, cos_th)
        cost = th ** 2 + 0.1 * (thdot / (1 + 10 * abs(th))) ** 2
        # Determine done flag
        done = steps > 500
        # Set info:
        info = dict()
        return obs, -cost, done, info

    # Initialize Environment
    env = Flatten(EagerxEnv(name=name, rate=rate, graph=graph, bridge=bridge, step_fn=step_fn))
    # env.render("human")

    # First reset
    env.reset()
    action = env.action_space.sample()
    for j in range(eps):
        print("\n[Episode %s]" % j)
        for i in range(steps):
            env.step(action)
        env.reset()
    print("\n[Finished]")
    env.shutdown()
    if roscore:
        roscore.shutdown()
    print("\n[Shutdown]")

