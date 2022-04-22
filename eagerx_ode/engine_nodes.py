from typing import Optional, List
import numpy as np

# IMPORT ROS
import rospy
import cv_bridge
from std_msgs.msg import UInt64, Float32MultiArray, Bool, Float32
from sensor_msgs.msg import Image


# IMPORT EAGERX
from eagerx.core.constants import process
from eagerx.utils.utils import Msg, get_attribute_from_module
from eagerx.core.entities import EngineNode
import eagerx.core.register as register


class OdeOutput(EngineNode):
    @staticmethod
    @register.spec("OdeOutput", EngineNode)
    def spec(
        spec,
        name: str,
        rate: float,
        process: Optional[int] = process.BRIDGE,
        color: Optional[str] = "cyan",
    ):
        """OdeOutput spec"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(OdeOutput)

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["tick"]
        spec.config.outputs = ["observation"]

    def initialize(self):
        # We will probably use self.simulator[self.obj_name] in callback & reset.
        assert (
            self.process == process.BRIDGE
        ), "Simulation node requires a reference to the simulator, hence it must be launched in the Bridge process"
        self.obj_name = self.config["name"]

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=UInt64)
    @register.outputs(observation=Float32MultiArray)
    def callback(self, t_n: float, tick: Optional[Msg] = None):
        assert isinstance(self.simulator[self.obj_name], dict), (
            'Simulator object "%s" is not compatible with this simulation node.' % self.simulator[self.obj_name]
        )
        data = self.simulator[self.obj_name]["state"]
        return dict(observation=Float32MultiArray(data=data))


class ActionApplied(EngineNode):
    @staticmethod
    @register.spec("ActionApplied", EngineNode)
    def spec(
        spec,
        name: str,
        rate: float,
        process: Optional[int] = process.BRIDGE,
        color: Optional[str] = "cyan",
    ):
        """ActionApplied spec"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(ActionApplied)

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["tick", "action_applied"]
        spec.config.outputs = ["action_applied"]

    def initialize(self):
        pass
        # We will probably use self.simulator[self.obj_name] in callback & reset.
        # assert self.process == process.BRIDGE, 'Simulation node requires a reference to the simulator, hence it must be launched in the Bridge process'
        # self.obj_name = self.config['name']

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=UInt64, action_applied=Float32MultiArray)
    @register.outputs(action_applied=Float32MultiArray)
    def callback(
        self,
        t_n: float,
        tick: Optional[Msg] = None,
        action_applied: Optional[Float32MultiArray] = None,
    ):
        if len(action_applied.msgs) > 0:
            data = action_applied.msgs[-1].data
        else:
            data = [0]
        return dict(action_applied=Float32MultiArray(data=data))


class OdeInput(EngineNode):
    @staticmethod
    @register.spec("OdeInput", EngineNode)
    def spec(
        spec,
        name: str,
        rate: float,
        default_action: List,
        process: Optional[int] = process.BRIDGE,
        color: Optional[str] = "green",
    ):
        """OdeInput spec"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(OdeInput)

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["tick", "action"]
        spec.config.outputs = ["action_applied"]

        # Set custom node params
        spec.config.default_action = default_action

    def initialize(self, default_action):
        # We will probably use self.simulator[self.obj_name] in callback & reset.
        assert (
            self.process == process.BRIDGE
        ), "Simulation node requires a reference to the simulator, hence it must be launched in the Bridge process"
        self.obj_name = self.config["name"]
        self.default_action = np.array(default_action)

    @register.states()
    def reset(self):
        self.simulator[self.obj_name]["input"] = np.squeeze(np.array(self.default_action))

    @register.inputs(tick=UInt64, action=Float32MultiArray)
    @register.outputs(action_applied=Float32MultiArray)
    def callback(
        self,
        t_n: float,
        tick: Optional[Msg] = None,
        action: Optional[Float32MultiArray] = None,
    ):
        assert isinstance(self.simulator[self.obj_name], dict), (
            'Simulator object "%s" is not compatible with this simulation node.' % self.simulator[self.obj_name]
        )

        # Set action in simulator for next step.
        self.simulator[self.obj_name]["input"] = np.squeeze(action.msgs[-1].data)

        # Send action that has been applied.
        return dict(action_applied=action.msgs[-1])


class OdeRender(EngineNode):
    @staticmethod
    @register.spec("OdeRender", EngineNode)
    def spec(
        spec,
        name: str,
        rate: float,
        process: Optional[int] = process.BRIDGE,
        color: Optional[str] = "cyan",
        shape=[480, 480],
        render_fn=None,
    ):
        """OdeRender spec"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(OdeRender)

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.color = color
        spec.config.inputs = ["tick", "observation", "action_applied"]
        spec.config.outputs = ["image"]

        # Modify custom node params
        spec.config.shape = shape
        spec.config.render_fn = render_fn

        # Set component parameter
        spec.inputs.observation.window = 1
        spec.inputs.action_applied.window = 1

    def initialize(self, shape, render_fn):
        self.cv_bridge = cv_bridge.CvBridge()
        self.shape = tuple(shape)
        self.render_toggle = False
        self.render_toggle_pub = rospy.Subscriber("%s/env/render/toggle" % self.ns, Bool, self._set_render_toggle)
        self.render_fn = (lambda img, obs, act: img) if render_fn is None else get_attribute_from_module(render_fn)

    @register.states()
    def reset(self):
        # This sensor is stateless (in contrast to e.g. a PID controller).
        pass

    @register.inputs(tick=UInt64, observation=Float32MultiArray, action_applied=Float32MultiArray)
    @register.outputs(image=Image)
    def callback(
        self, t_n: float, tick: Msg = None, observation: Float32MultiArray = None, action_applied: Float32MultiArray = None
    ):
        if self.render_toggle:
            img = np.zeros((self.shape[0], self.shape[1], 3), np.uint8)
            img = self.render_fn(img, observation, action_applied)
            try:
                msg = self.cv_bridge.cv2_to_imgmsg(img, "bgr8")
            except ImportError as e:
                rospy.logwarn_once("[%s] %s. Using numpy instead." % (self.ns_name, e))
                data = img.tobytes("C")
                msg = Image(data=data, height=self.shape[1], width=self.shape[0], encoding="bgr8")
        else:
            msg = Image()
        return dict(image=msg)

    def _set_render_toggle(self, msg):
        if msg.data:
            rospy.loginfo("[%s] START RENDERING!" % self.name)
        else:
            rospy.loginfo("[%s] STOPPED RENDERING!" % self.name)
        self.render_toggle = msg.data


class OdeFloatOutput(EngineNode):
    """
    EngineNode that can be used to create sensor that outputs the value at specified index of an incoming array.
    """

    @staticmethod
    @register.spec("OdeFloatOutput", EngineNode)
    def spec(
        spec,
        name: str,
        rate: float,
        idx: int = 0,
        process: Optional[int] = process.BRIDGE,
        color: Optional[str] = "cyan",
    ):
        """OdeFloatOutput spec
        :param idx: index of the array that will be sent through
        """
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(OdeFloatOutput)

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["tick"]
        spec.config.outputs = ["observation"]

        # Set custom node params
        spec.config.idx = idx

    def initialize(self, idx):
        # We will probably use self.simulator[self.obj_name] in callback & reset.
        assert (
            self.process == process.BRIDGE
        ), "Simulation node requires a reference to the simulator, hence it must be launched in the Bridge process"
        self.obj_name = self.config["name"]
        self.idx = idx

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=UInt64)
    @register.outputs(observation=Float32)
    def callback(self, t_n: float, tick: Optional[Msg] = None):
        assert isinstance(self.simulator[self.obj_name], dict), (
            'Simulator object "%s" is not compatible with this simulation node.' % self.simulator[self.obj_name]
        )
        data = self.simulator[self.obj_name]["state"]
        return dict(observation=Float32(data[self.idx]))
