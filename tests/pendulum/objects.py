# ROS IMPORTS
from std_msgs.msg import Float32MultiArray, Float32
from sensor_msgs.msg import Image

# EAGERx IMPORTS
from eagerx_ode.engine import OdeEngine
from eagerx.core.entities import (
    Object,
    EngineNode,
    SpaceConverter,
    EngineState,
)
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
import eagerx.core.register as register


class Pendulum(Object):
    entity_id = "Eagerx_Ode_Pendulum"

    @staticmethod
    @register.sensors(
        pendulum_output=Float32MultiArray, action_applied=Float32MultiArray, image=Image, theta=Float32, dtheta=Float32
    )
    @register.actuators(pendulum_input=Float32MultiArray)
    @register.engine_states(model_state=Float32MultiArray, model_parameters=Float32MultiArray)
    @register.config(render_shape=[480, 480], Dfun="tests.pendulum.pendulum_ode/pendulum_dfun")
    def agnostic(spec: ObjectSpec, rate):
        """Agnostic definition of the pendulum"""
        # Register standard converters, space_converters, and processors
        import eagerx.converters  # noqa # pylint: disable=unused-import

        # Set observation properties: (space_converters, rate, etc...)
        spec.sensors.pendulum_output.rate = rate
        spec.sensors.pendulum_output.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", low=[-3.14, -9], high=[3.14, 9], dtype="float32"
        )

        spec.sensors.action_applied.rate = rate
        spec.sensors.action_applied.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", low=[-3], high=[3], dtype="float32"
        )

        spec.sensors.image.rate = 15
        spec.sensors.image.space_converter = SpaceConverter.make(
            "Space_Image",
            low=0,
            high=1,
            shape=spec.config.render_shape,
            dtype="float32",
        )

        spec.sensors.theta.rate = rate
        spec.sensors.theta.space_converter = SpaceConverter.make("Space_Float32", low=-9999.0, high=9999.0, dtype="float32")

        spec.sensors.dtheta.rate = rate
        spec.sensors.dtheta.space_converter = SpaceConverter.make("Space_Float32", low=-9999.0, high=9999.0, dtype="float32")

        # Set actuator properties: (space_converters, rate, etc...)
        spec.actuators.pendulum_input.rate = rate
        spec.actuators.pendulum_input.window = 1
        spec.actuators.pendulum_input.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", low=[-3], high=[3], dtype="float32"
        )

        # Set model_state properties: (space_converters)
        spec.states.model_state.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            low=[-3.14159265359, -9],
            high=[3.14159265359, 9],
            dtype="float32",
        )

        # Set model_parameters properties: (space_converters) # [J, m, l, b0, K, R, c, a]
        fixed = [0.000189238, 0.0563641, 0.0437891, 0.000142205, 0.0502769, 9.83536]
        diff = [0, 0, 0, 0.05, 0.05]  # Percentual delta with respect to fixed value
        low = [val - diff * val for val, diff in zip(fixed, diff)]
        high = [val + diff * val for val, diff in zip(fixed, diff)]
        spec.states.model_parameters.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", low=low, high=high, dtype="float32"
        )

    @staticmethod
    @register.spec(entity_id, Object)
    def spec(
        spec: ObjectSpec,
        name: str,
        sensors=None,
        states=None,
        rate=30,
        Dfun="tests.pendulum.pendulum_ode/pendulum_dfun",
    ):
        """Object spec of pendulum"""
        # Modify default agnostic params
        # Only allow changes to the agnostic params (rates, windows, (space)converters, etc...
        spec.config.name = name
        spec.config.sensors = sensors if sensors else ["pendulum_output", "action_applied", "image", "theta", "dtheta"]
        spec.config.actuators = ["pendulum_input"]
        spec.config.states = states if states else ["model_state"]

        # Add registered agnostic params
        spec.config.render_shape = [480, 480]

        spec.config.Dfun = Dfun

        # Add engine implementation
        Pendulum.agnostic(spec, rate)

    @staticmethod
    @register.engine(entity_id, OdeEngine)  # This decorator pre-initializes engine implementation with default object_params
    def ode_engine(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (OdeEngine) of the object."""
        # Import any object specific entities for this engine

        # Set object arguments (nothing to set here in this case)
        spec.OdeEngine.ode = "tests.pendulum.pendulum_ode/pendulum_ode"
        # Set default params of pendulum ode [J, m, l, b0, K, R].
        spec.OdeEngine.ode_params = [
            0.000189238,
            0.0563641,
            0.0437891,
            0.000142205,
            0.0502769,
            9.83536,
        ]
        spec.OdeEngine.Dfun = spec.config.Dfun

        # Create engine states (no agnostic states defined in this case)
        spec.OdeEngine.states.model_state = EngineState.make("OdeEngineState")
        spec.OdeEngine.states.model_parameters = EngineState.make("OdeParameters", list(range(5)))

        # Create sensor engine nodes
        obs = EngineNode.make(
            "OdeOutput",
            "pendulum_output",
            rate=spec.sensors.pendulum_output.rate,
            process=2,
        )
        image = EngineNode.make(
            "OdeRender",
            "image",
            shape=spec.config.render_shape,
            render_fn="tests.pendulum.pendulum_render/pendulum_render_fn",
            rate=spec.sensors.image.rate,
            process=0,
        )
        theta = EngineNode.make("OdeFloatOutput", "theta", rate=spec.sensors.theta.rate, process=2, idx=0)

        dtheta = EngineNode.make("OdeFloatOutput", "dtheta", rate=spec.sensors.dtheta.rate, process=2, idx=1)

        # Create actuator engine nodes
        action = EngineNode.make(
            "OdeInput",
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
        applied = EngineNode.make("ActionApplied", "applied", rate=spec.sensors.action_applied.rate, process=2)
        graph.add(applied)
        graph.connect(
            source=action.outputs.action_applied,
            target=applied.inputs.action_applied,
            skip=True,
        )
        graph.connect(source=applied.outputs.action_applied, sensor="action_applied")
        graph.connect(source=theta.outputs.observation, sensor="theta")
        graph.connect(source=dtheta.outputs.observation, sensor="dtheta")
