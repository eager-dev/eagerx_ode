from eagerx.core.entities import EngineState
import numpy as np


class OdeEngineState(EngineState):
    @classmethod
    def make(cls):
        spec = cls.get_specification()
        return spec

    def initialize(self, spec, simulator):
        self.simulator = simulator

    def reset(self, state):
        self.simulator["state"] = np.squeeze(state.data)


class OdeParameters(EngineState):
    @classmethod
    def make(cls, indices):
        spec = cls.get_specification()
        spec.config.indices = indices
        return spec

    def initialize(self, spec, simulator):
        self.simulator = simulator
        self.indices = spec.config.indices

    def reset(self, state):
        for i in self.indices:
            self.simulator["ode_params"][i] = state.data[i]
