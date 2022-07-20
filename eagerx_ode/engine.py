# OTHER
from typing import Optional, List
from scipy.integrate import odeint

# RX IMPORTS
from eagerx.core.constants import process, ERROR
import eagerx.core.register as register
from eagerx.core.entities import Engine
from eagerx.utils.utils import load


class OdeEngine(Engine):
    @classmethod
    def make(
        cls,
        rate: float,
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
        spec = cls.get_specification()

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
        return spec

    def initialize(self, spec):
        rtol = spec.config.rtol
        atol = spec.config.atol
        hmax = spec.config.hmax
        hmin = spec.config.hmin
        mxstep = spec.config.mxstep

        # Initialize any simulator here, that is passed as reference to each engine node
        self.odeint_args = dict(rtol=rtol, atol=atol, hmax=hmax, hmin=hmin, mxstep=mxstep)
        self.simulator = dict()

    def add_object(self, spec, ode: str = None, Dfun: Optional[str] = None, ode_params: Optional[List] = None):
        # add object to simulator (we have a ref to the simulator with self.simulator)
        self.backend.loginfo(f'Adding object "{spec.config.name}" of type "{spec.config.entity_id}" to the simulator.')

        # Extract relevant agnostic params
        obj_name = spec.config.name
        ode = load(ode)
        Dfun = load(Dfun) if Dfun is not None else None

        # Create new env, and add to simulator
        self.simulator[obj_name] = dict(
            ode=ode,
            Dfun=Dfun,
            state=None,
            input=None,
            ode_params=ode_params,
        )

    def pre_reset(self):
        pass

    @register.states()
    def reset(self):
        pass

    def callback(self, t_n: float):
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
