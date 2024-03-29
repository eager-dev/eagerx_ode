import eagerx
from copy import deepcopy
import numpy as np

import pytest

NP = eagerx.NEW_PROCESS
ENV = eagerx.ENVIRONMENT


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "eps, steps, sync, rtf, p",
    [(3, 3, True, 0, ENV)],
)
def test_ode_engine(eps, steps, sync, rtf, p):
    """
    Creates an environment with the dummy Pendulum and OdeEngine.

    :param eps: Number of episodes
    :param steps: Number of steps per episode
    :param sync: If True, the environment is reactive
    :param rtf: Real-time factor
    :param p: Process
    :return:
    """
    eagerx.set_log_level(eagerx.WARN)

    # Define unique name for test environment
    name = f"{eps}_{steps}_{sync}_{p}"
    engine_p = p
    rate = 30

    # Initialize empty graphs
    graph = eagerx.Graph.create()

    # Create pendulum
    try:
        from tests.pendulum.objects import Pendulum
        running_local = False
    except ImportError:
        from pendulum.objects import Pendulum
        running_local = True
    pendulum = Pendulum.make("pendulum")
    graph.add(pendulum)

    if running_local:
        pendulum.config.ode = "pendulum.pendulum_ode/pendulum_ode"
        pendulum.config.Dfun = "pendulum.pendulum_ode/pendulum_dfun"

    # Connect the nodes
    graph.connect(action="action", target=pendulum.actuators.pendulum_input)
    graph.connect(source=pendulum.sensors.pendulum_output, observation="observation", window=1)
    graph.connect(source=pendulum.sensors.action_applied, observation="action_applied", window=1)
    graph.connect(source=pendulum.sensors.theta, observation="theta", window=1)
    graph.connect(source=pendulum.sensors.dtheta, observation="dtheta", window=1)

    graph.render(pendulum.sensors.image, rate=10)

    # Define engines
    from eagerx_ode.engine import OdeEngine
    engine = OdeEngine.make(rate=rate, sync=sync, real_time_factor=rtf, process=engine_p)

    # Define backend
    from eagerx.backends.single_process import SingleProcess
    backend = SingleProcess.make()

    # Define environment
    class TestEnv(eagerx.BaseEnv):
        def __init__(self, name, rate, graph, engine, backend, force_start, render_mode: str = None):
            super().__init__(name, rate, graph, engine, backend=backend, force_start=force_start, render_mode=render_mode)

        def step(self, action):
            obs = self._step(action)
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
            truncated = steps > 500
            terminated = False
            # Set info:
            info = dict()
            if self.render_mode == "human":
                self.render()
            return obs, -cost, terminated, truncated, info

        def reset(self, seed = None, options = None):
            states = self.state_space.sample()
            obs = self._reset(states)
            if self.render_mode == "human":
                self.render()
            return obs, {}

    # Initialize Environment
    env = TestEnv(name, rate, graph, engine, backend, force_start=True)

    # First reset
    env.reset()
    action = env.action_space.sample()
    for j in range(eps):
        print("\n[Episode %s]" % j)
        for i in range(steps):
            obs, _, _, _, _ = env.step(action)
        env.reset()
    print("\n[Finished]")
    env.shutdown()
    print("\n[Shutdown]")


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "eps, steps, sync, rtf, p",
    [(3, 30, True, 0, ENV)],
)
def test_dfun(eps, steps, sync, rtf, p):
    """
    Creates two environments, one uses a Jacobian function (Dfun) and the other not.
    Tests if the observations of the environments are close to eachother within a tolerance.

    :param eps: Number of episodes
    :param steps: Number of steps per episode
    :param sync: If True, the environment is reactive
    :param rtf: Real-time factor
    :param p: Process
    :return:
    """

    eagerx.set_log_level(eagerx.WARN)

    # Define unique name for test environment
    name = f"{eps}_{steps}_{sync}_{p}"
    engine_p = p
    rate = 30

    # Initialize empty graphs
    graph = eagerx.Graph.create()
    graph2 = eagerx.Graph.create()

    # Create pendulum
    try:
        from tests.pendulum.objects import Pendulum
        running_local = False
    except ImportError:
        from pendulum.objects import Pendulum
        running_local = True
    pendulum = Pendulum.make("pendulum")
    graph.add(pendulum)
    pendulum2 = Pendulum.make("pendulum2", Dfun=None)
    graph2.add(pendulum2)

    if running_local:
        pendulum.config.ode = "pendulum.pendulum_ode/pendulum_ode"
        pendulum.config.Dfun = "pendulum.pendulum_ode/pendulum_dfun"
        pendulum2.config.ode = "pendulum.pendulum_ode/pendulum_ode"
        pendulum2.config.Dfun = None

    # Connect the nodes
    graph.connect(action="action", target=pendulum.actuators.pendulum_input)
    graph.connect(source=pendulum.sensors.pendulum_output, observation="observation", window=1)
    graph.connect(source=pendulum.sensors.action_applied, observation="action_applied", window=1)
    graph.render(pendulum.sensors.image, rate=10)

    graph2.connect(action="action", target=pendulum2.actuators.pendulum_input)
    graph2.connect(source=pendulum2.sensors.pendulum_output, observation="observation", window=1)
    graph2.connect(source=pendulum2.sensors.action_applied, observation="action_applied", window=1)
    graph2.render(pendulum2.sensors.image, rate=10)

    # Define engines
    from eagerx_ode.engine import OdeEngine
    engine = OdeEngine.make(rate=rate, sync=sync, real_time_factor=rtf, process=engine_p)

    # Define backend
    from eagerx.backends.single_process import SingleProcess
    backend = SingleProcess.make()

    # Define environment
    class TestEnv(eagerx.BaseEnv):
        def __init__(self, name, rate, graph, engine, backend, force_start, render_mode: str = None):
            super().__init__(name, rate, graph, engine, backend=backend, force_start=force_start, render_mode=render_mode)

        def step(self, action):
            obs = self._step(action)
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
            truncated = steps > 500
            terminated = False
            # Set info:
            info = dict()
            return obs, -cost, terminated, truncated, info

        def reset(self, seed=None, options=None):
            states = self.state_space.sample()
            for key, value in states.items():
                if "model_state" in key:
                    states[key] = value * 0.0
            obs = self._reset(states)
            return obs, {}

    # Initialize Environment
    env = TestEnv(name, rate, graph, deepcopy(engine), backend, force_start=True)
    env2 = TestEnv(name + "_2", rate, graph2, deepcopy(engine), backend, force_start=True)

    # First reset
    obs, info = env.reset()
    obs2, info2 = env2.reset()
    action = env.action_space.sample()
    for j in range(eps):
        print("\n[Episode %s]" % j)
        for i in range(steps):
            obs, _, _, _, _ = env.step(action)
            obs2, _, _, _, _ = env2.step(action)

            # Assert if result is the same with and without Jacobian
            assert np.allclose(obs["observation"], obs2["observation"])
        env.reset()
        env2.reset()
    print("\n[Finished]")
    env.shutdown()
    env2.shutdown()
    print("\n[Shutdown]")


if __name__ == "__main__":
    test_dfun(3, 30, True, 0, ENV)
    test_ode_engine(3, 3, True, 0, ENV)
