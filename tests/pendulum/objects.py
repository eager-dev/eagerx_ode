# ROS IMPORTS
from std_msgs.msg import Float32MultiArray

# EAGERx IMPORTS
from eagerx_ode.bridge import OdeBridge
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
    @register.sensors(pendulum_output=Float32MultiArray, action_applied=Float32MultiArray)
    @register.actuators(pendulum_input=Float32MultiArray)
    @register.engine_states(model_state=Float32MultiArray, model_parameters=Float32MultiArray)
    @register.config()
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
    def spec(spec: ObjectSpec, name: str, sensors=None, states=None, rate=30):
        """Object spec of pendulum"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        Pendulum.initialize_spec(spec)

        # Modify default agnostic params
        # Only allow changes to the agnostic params (rates, windows, (space)converters, etc...
        spec.config.name = name
        spec.config.sensors = sensors if sensors else ["pendulum_output", "action_applied"]
        spec.config.actuators = ["pendulum_input"]
        spec.config.states = states if states else ["model_state"]

        # Add bridge implementation
        Pendulum.agnostic(spec, rate)

    @staticmethod
    @register.bridge(entity_id, OdeBridge)  # This decorator pre-initializes bridge implementation with default object_params
    def ode_bridge(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (OdeBridge) of the object."""
        # Import any object specific entities for this bridge

        # Set object arguments (nothing to set here in this case)
        spec.OdeBridge.ode = "tests.pendulum.pendulum_ode/pendulum_ode"
        # Set default params of pendulum ode [J, m, l, b0, K, R].
        spec.OdeBridge.ode_params = [
            0.000189238,
            0.0563641,
            0.0437891,
            0.000142205,
            0.0502769,
            9.83536,
        ]

        # Create engine states (no agnostic states defined in this case)
        spec.OdeBridge.states.model_state = EngineState.make("OdeEngineState")
        spec.OdeBridge.states.model_parameters = EngineState.make("OdeParameters", list(range(5)))

        # Create sensor engine nodes
        obs = EngineNode.make(
            "OdeOutput",
            "pendulum_output",
            rate=spec.sensors.pendulum_output.rate,
            process=2,
        )

        # Create actuator engine nodes
        action = EngineNode.make(
            "OdeInput",
            "pendulum_actuator",
            rate=spec.actuators.pendulum_input.rate,
            process=2,
            default_action=[0],
        )

        # Connect all engine nodes
        graph.add([obs, action])
        graph.connect(source=obs.outputs.observation, sensor="pendulum_output")
        graph.connect(actuator="pendulum_input", target=action.inputs.action)

        # Add action applied
        applied = EngineNode.make("ActionApplied", "applied", rate=spec.sensors.action_applied.rate, process=2)
        graph.add(applied)
        graph.connect(
            source=action.outputs.action_applied,
            target=applied.inputs.action_applied,
            skip=True,
        )
        graph.connect(source=applied.outputs.action_applied, sensor="action_applied")
