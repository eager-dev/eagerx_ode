from typing import Optional, List
import numpy as np

# IMPORT EAGERX
from eagerx.core.space import Space
from eagerx.core.constants import process
from eagerx.utils.utils import Msg, load
from eagerx.core.entities import EngineNode
import eagerx.core.register as register


class OdeOutput(EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        process: Optional[int] = process.ENGINE,
        color: Optional[str] = "cyan",
    ):
        """OdeOutput spec"""
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["tick"]
        spec.config.outputs = ["observation"]
        return spec

    def initialize(self, spec, object_spec, simulator):
        # We will probably use self.simulator[self.obj_name] in callback & reset.
        assert (
            spec.config.process == process.ENGINE
        ), "Simulation node requires a reference to the simulator, hence it must be launched in the Engine process"
        self.obj_name = object_spec.config.name
        self.simulator = simulator

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=Space(shape=(), dtype="int64"))
    @register.outputs(observation=Space(dtype="float32"))
    def callback(self, t_n: float, tick: Msg):
        assert isinstance(self.simulator[self.obj_name], dict), (
            'Simulator object "%s" is not compatible with this simulation node.' % self.simulator[self.obj_name]
        )
        data = self.simulator[self.obj_name]["state"]
        return dict(observation=data.astype("float32"))


class ActionApplied(EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        process: Optional[int] = process.ENGINE,
        color: Optional[str] = "cyan",
    ):
        """ActionApplied spec"""
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["tick", "action_applied"]
        spec.config.outputs = ["action_applied"]
        return spec

    def initialize(self, spec, object_spec, simulator):
        pass

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=Space(shape=(), dtype="int64"), action_applied=Space(dtype="float32"))
    @register.outputs(action_applied=Space(dtype="float32"))
    def callback(self, t_n: float, tick: Msg, action_applied: Msg):
        if len(action_applied.msgs) > 0:
            data = action_applied.msgs[-1]
        else:
            data = np.array([0], dtype="float32")
        return dict(action_applied=data)


class OdeInput(EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        default_action: List,
        process: Optional[int] = process.ENGINE,
        color: Optional[str] = "green",
    ):
        """OdeInput spec"""
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["tick", "action"]
        spec.config.outputs = ["action_applied"]

        # Set custom node params
        spec.config.default_action = default_action
        return spec

    def initialize(self, spec, object_spec, simulator):
        # We will probably use self.simulator[self.obj_name] in callback & reset.
        assert (
            spec.config.process == process.ENGINE
        ), "Simulation node requires a reference to the simulator, hence it must be launched in the Engine process"
        self.obj_name = object_spec.config.name
        self.simulator = simulator
        self.default_action = np.array(spec.config.default_action)

    @register.states()
    def reset(self):
        self.simulator[self.obj_name]["input"] = np.squeeze(np.array(self.default_action))

    @register.inputs(tick=Space(shape=(), dtype="int64"), action=Space(dtype="float32"))
    @register.outputs(action_applied=Space(dtype="float32"))
    def callback(self, t_n: float, tick: Msg, action: Msg):
        assert isinstance(self.simulator[self.obj_name], dict), (
            'Simulator object "%s" is not compatible with this simulation node.' % self.simulator[self.obj_name]
        )

        # Set action in simulator for next step.
        self.simulator[self.obj_name]["input"] = np.squeeze(action.msgs[-1].data)

        # Send action that has been applied.
        return dict(action_applied=action.msgs[-1])


class OdeRender(EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        process: Optional[int] = process.ENGINE,
        color: Optional[str] = "cyan",
        shape: list = None,
        render_fn: str = None,
    ):
        """OdeRender spec"""
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.color = color
        spec.config.inputs = ["tick", "observation", "action_applied"]
        spec.config.outputs = ["image"]

        # Modify custom node params
        spec.config.shape = shape if isinstance(shape, list) else [480, 480]
        spec.config.render_fn = render_fn

        # Set image space
        spec.outputs.image.space = Space(low=0, high=255, shape=(spec.config.shape[0], spec.config.shape[1], 3), dtype="uint8")
        return spec

    def initialize(self, spec, object_spec, simulator):
        self.shape = tuple(spec.config.shape)
        self.render_toggle = False
        self.sub_toggle = self.backend.Subscriber("%s/env/render/toggle" % self.ns, "bool", self._set_render_toggle)
        try:
            self.render_fn = load(spec.config.render_fn)
        except (ModuleNotFoundError, TypeError) as e:
            self.backend.logwarn(f"Could not load render_fn `{spec.config.render_fn}`: {e}")
            self.render_fn = lambda img, obs, act: img

    @register.states()
    def reset(self):
        pass

    @register.inputs(
        tick=Space(shape=(), dtype="int64"), observation=Space(dtype="float32"), action_applied=Space(dtype="float32")
    )
    @register.outputs(image=Space(dtype="uint8"))
    def callback(self, t_n: float, tick: Msg, observation: Msg, action_applied: Msg):
        img = np.zeros((self.shape[0], self.shape[1], 3), np.uint8)
        if self.render_toggle:
            img = self.render_fn(img, observation, action_applied)
        return dict(image=img)

    def _set_render_toggle(self, msg):
        if msg:
            self.backend.logdebug("[%s] START RENDERING!" % self.name)
        else:
            self.backend.logdebug("[%s] STOPPED RENDERING!" % self.name)
        self.render_toggle = msg

    def shutdown(self):
        self.backend.logdebug(f"[{self.name}] {self.name}.shutdown() called.")
        self.sub_toggle.unregister()


class OdeFloatOutput(EngineNode):
    """
    EngineNode that can be used to create sensor that outputs the value at specified index of an incoming array.
    """

    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        idx: int = 0,
        process: Optional[int] = process.ENGINE,
        color: Optional[str] = "cyan",
    ):
        """OdeFloatOutput spec
        :param idx: index of the array that will be sent through
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["tick"]
        spec.config.outputs = ["observation"]

        # Set custom node params
        spec.config.idx = idx
        return spec

    def initialize(self, spec, object_spec, simulator):
        # We will probably use self.simulator[self.obj_name] in callback & reset.
        assert (
            self.process == process.ENGINE
        ), "Simulation node requires a reference to the simulator, hence it must be launched in the Engine process"
        self.obj_name = object_spec.config.name
        self.simulator = simulator
        self.idx = spec.config.idx

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=Space(shape=(), dtype="int64"))
    @register.outputs(observation=Space(shape=(), dtype="float32"))
    def callback(self, t_n: float, tick: Msg):
        assert isinstance(self.simulator[self.obj_name], dict), (
            'Simulator object "%s" is not compatible with this simulation node.' % self.simulator[self.obj_name]
        )
        data = self.simulator[self.obj_name]["state"]
        return dict(observation=np.array(data[self.idx], dtype="float32"))
