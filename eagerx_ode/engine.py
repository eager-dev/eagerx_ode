# OTHER
from typing import Optional, Dict, Union, List
from scipy.integrate import odeint

# ROS IMPORTS
import rospy
from std_msgs.msg import UInt64
from genpy.message import Message

# RX IMPORTS
from eagerx.core.constants import process, ERROR
import eagerx.core.register as register
from eagerx.core.entities import Engine
from eagerx.core.specs import EngineSpec
from eagerx.utils.utils import Msg, get_attribute_from_module


class OdeEngine(Engine):
    @staticmethod
    @register.spec("OdeEngine", Engine)
    def spec(
        spec: EngineSpec,
        rate,
        sync: Optional[bool] = True,
        process: Optional[int] = process.ENVIRONMENT,
        real_time_factor: Optional[float] = 0,
        simulate_delays: Optional[bool] = True,
        log_level: Optional[int] = ERROR,
        rtol: float = 2e-8,
        atol: float = 2e-8,
        hmax: float = 0.0,
        hmin: float = 0.0,
        mxstep: int = 0,
    ):
        """
        Spec of the OdeEngine

        :param spec: Not provided by the user.
        :param rate: Rate of the engine
        :param process: {0: NEW_PROCESS, 1: ENVIRONMENT, 2: ENGINE, 3: EXTERNAL}
        :param sync: Run reactive or async
        :param real_time_factor: simulation speed. 0 == "as fast as possible".
        :param simulate_delays: Boolean flag to simulate delays.
        :param log_level: {0: SILENT, 10: DEBUG, 20: INFO, 30: WARN, 40: ERROR, 50: FATAL}
        :param rtol: The input parameters rtol and atol determine the error control performed by the solver.
        :param atol: The input parameters rtol and atol determine the error control performed by the solver.
        :param hmax: The maximum absolute step size allowed.
        :param hmin: The minimum absolute step size allowed.
        :param mxstep: Maximum number of (internally defined) steps allowed for each integration point in t.
        :return: EngineSpec
        """
        # Modify default engine params
        spec.config.rate = rate
        spec.config.process = process
        spec.config.sync = sync
        spec.config.real_time_factor = real_time_factor
        spec.config.simulate_delays = simulate_delays
        spec.config.log_level = log_level
        spec.config.color = "magenta"

        # Add custom params
        custom = dict(rtol=rtol, atol=atol, hmax=hmax, hmin=hmin, mxstep=mxstep)
        spec.config.update(custom)

    def initialize(self, rtol, atol, hmax, hmin, mxstep):
        # Initialize any simulator here, that is passed as reference to each engine node
        self.odeint_args = dict(rtol=rtol, atol=atol, hmax=hmax, hmin=hmin, mxstep=mxstep)
        self.simulator = dict()

    @register.engine_config(ode=None, Dfun=None, ode_params=list())
    def add_object(self, config, engine_config, node_params, state_params):
        # add object to simulator (we have a ref to the simulator with self.simulator)
        rospy.loginfo(f'Adding object "{config["name"]}" of type "{config["entity_id"]}" to the simulator.')

        # Extract relevant agnostic params
        obj_name = config["name"]
        ode = get_attribute_from_module(engine_config["ode"])
        Dfun = get_attribute_from_module(engine_config["Dfun"]) if "Dfun" in config and config["Dfun"] else None

        # Create new env, and add to simulator
        self.simulator[obj_name] = dict(
            ode=ode,
            Dfun=Dfun,
            state=None,
            input=None,
            ode_params=engine_config["ode_params"],
        )

    def pre_reset(self, **kwargs: Optional[Msg]):
        pass

    @register.states()
    def reset(self, **kwargs: Optional[Msg]):
        pass

    @register.outputs(tick=UInt64)
    def callback(self, t_n: float, **kwargs: Dict[str, Union[List[Message], float, int]]):
        for _obj_name, sim in self.simulator.items():
            input = sim["input"]
            ode = sim["ode"]
            Dfun = sim["Dfun"]
            x = sim["state"]
            ode_params = sim["ode_params"]
            if x is not None and input is not None:
                sim["state"] = odeint(
                    ode,
                    x,
                    [0, 1.0 / self.rate],
                    args=(input, *ode_params),
                    Dfun=Dfun,
                    **self.odeint_args,
                )[-1]
