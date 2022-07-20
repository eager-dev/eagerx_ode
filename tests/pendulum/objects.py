from eagerx.core.space import Space
from eagerx_ode.engine import OdeEngine
from eagerx.core.entities import Object
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
import eagerx.core.register as register
import numpy as np


class Pendulum(Object):
    @classmethod
    @register.sensors(
        pendulum_output=Space(low=np.array([-3.14, -9], dtype="float32"), high=np.array([3.14, 9], dtype="float32")),
        action_applied=Space(low=np.array([-3], dtype="float32"), high=np.array([3], dtype="float32")),
        image=Space(dtype="uint8"),
        theta=Space(low=-999.0, high=999.0, shape=(), dtype="float32"),
        dtheta=Space(low=-999.0, high=999.0, shape=(), dtype="float32")
    )
    @register.actuators(pendulum_input=Space(low=np.array([-3], dtype="float32"), high=np.array([3], dtype="float32")))
    @register.engine_states(model_state=Space(low=np.array([-3.14, -9], dtype="float32"), high=np.array([3.14, 9], dtype="float32")),
                            model_parameters=Space(dtype="float32"))
    def make(
        cls,
        name: str,
        sensors=None,
        states=None,
        rate=30,
        Dfun="tests.pendulum.pendulum_ode/pendulum_dfun",
    ):
        """Object spec of pendulum"""
        spec = cls.get_specification()

        # Modify default agnostic params
        # Only allow changes to the agnostic params (rates, windows, (space)converters, etc...
        spec.config.name = name
        spec.config.sensors = sensors if sensors else ["pendulum_output", "action_applied", "image", "theta", "dtheta"]
        spec.config.actuators = ["pendulum_input"]
        spec.config.states = states if states else ["model_state"]

        # Add registered agnostic params
        spec.config.render_shape = [480, 480]
        spec.config.Dfun = Dfun

        # Set observation properties: (space_converters, rate, etc...)
        spec.sensors.pendulum_output.rate = rate
        spec.sensors.action_applied.rate = rate
        spec.sensors.dtheta.rate = rate
        spec.sensors.theta.rate = rate
        spec.actuators.pendulum_input.rate = rate
        spec.sensors.image.rate = 15

        # Set image space
        shape = (spec.config.render_shape[0], spec.config.render_shape[1], 3)
        spec.sensors.image.space = Space(low=0, high=255, shape=shape, dtype="uint8")

        # Set model_parameters properties: (space_converters) # [J, m, l, b0, K, R, c, a]
        fixed = [0.000189238, 0.0563641, 0.0437891, 0.000142205, 0.0502769, 9.83536]
        diff = [0, 0, 0, 0.05, 0.05]  # Percentual delta with respect to fixed value
        low = np.array([val - diff * val for val, diff in zip(fixed, diff)], dtype="float32")
        high = np.array([val + diff * val for val, diff in zip(fixed, diff)], dtype="float32")
        spec.states.model_parameters.space = Space(low=low, high=high)

        return spec

    @staticmethod
    @register.engine(OdeEngine)  # This decorator pre-initializes engine implementation with default object_params
    def ode_engine(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (OdeEngine) of the object."""
        # Set object arguments (nothing to set here in this case)
        spec.engine.ode = "tests.pendulum.pendulum_ode/pendulum_ode"
        # Set default params of pendulum ode [J, m, l, b0, K, R].
        spec.engine.ode_params = [
            0.000189238,
            0.0563641,
            0.0437891,
            0.000142205,
            0.0502769,
            9.83536,
        ]
        spec.engine.Dfun = spec.config.Dfun

        # Create engine states (no agnostic states defined in this case)
        from eagerx_ode.engine_states import OdeEngineState, OdeParameters
        spec.engine.states.model_state = OdeEngineState.make()
        spec.engine.states.model_parameters = OdeParameters.make(list(range(5)))

        # Create sensor engine nodes
        from eagerx_ode.engine_nodes import OdeOutput, OdeInput, OdeFloatOutput, OdeRender, ActionApplied
        obs = OdeOutput.make("pendulum_output", rate=spec.sensors.pendulum_output.rate, process=2)
        image = OdeRender.make(
            "image",
            shape=spec.config.render_shape,
            render_fn="tests.pendulum.pendulum_render/pendulum_render_fn",
            rate=spec.sensors.image.rate,
            process=0,
        )
        theta = OdeFloatOutput.make("theta", rate=spec.sensors.theta.rate, process=2, idx=0)

        dtheta = OdeFloatOutput.make("dtheta", rate=spec.sensors.dtheta.rate, process=2, idx=1)

        # Create actuator engine nodes
        action = OdeInput.make(
            "pendulum_actuator",
            rate=spec.actuators.pendulum_input.rate,
            process=2,
            default_action=[0],
        )

        # Connect all engine nodes
        graph.add([obs, image, action, theta, dtheta])
        graph.connect(source=obs.outputs.observation, sensor="pendulum_output")
        graph.connect(actuator="pendulum_input", target=action.inputs.action)
        graph.connect(
            source=action.outputs.action_applied,
            target=image.inputs.action_applied,
            skip=True,
        )
        graph.connect(source=obs.outputs.observation, target=image.inputs.observation)
        graph.connect(source=image.outputs.image, sensor="image")

        # Add action applied
        applied = ActionApplied.make("applied", rate=spec.sensors.action_applied.rate, process=2)
        graph.add(applied)
        graph.connect(
            source=action.outputs.action_applied,
            target=applied.inputs.action_applied,
            skip=True,
        )
        graph.connect(source=applied.outputs.action_applied, sensor="action_applied")
        graph.connect(source=theta.outputs.observation, sensor="theta")
        graph.connect(source=dtheta.outputs.observation, sensor="dtheta")
