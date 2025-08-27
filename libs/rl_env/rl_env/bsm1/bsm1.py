from dataclasses import dataclass, field
from typing import Literal

import gymnasium as gym
import numpy as np
from lib_config.config import config

from rl_env.group_util import EnvConfig, env_group

from .asm1 import TankModel
from .clarifier import ClarifierModel
from .controllers import PIController
from .flows import (
    FlowModel,
    InfluentModel,
    MixedTimeSeriesInfluent,
    SbhSensorModel,
    SensorModel,
    SteadyStateInfluent,
    TimeSeriesInfluent,
)


@config(frozen=True)
class BSM1Config(EnvConfig):
    name: Literal['BSM1-v0'] = 'BSM1-v0'
    only_so5_action: bool = False
    no_recirc: bool = False
    ox_ctrl_enabled: bool = False
    sno_ctrl_enabled: bool = False
    baseline_duration: float = 0.0 # Duration of the baseline period in days
    direct_action_aeration: bool = True # If True, the action is the flow rate, otherwise it is the setpoint
    direct_action_imlr: bool = True # If True, the action is the IMLR flow rate, otherwise it is the setpoint
    influent_source: Literal['constant', 'dryinfluent', 'raininfluent', 'storminfluent', 'mixed'] = 'constant'
    env_dt_minutes: float = 15.0 # time between env steps in minutes

class BSM1Env(gym.Env):
    """
    Environment based on the BSM1 benchmark model for wastewater treatment plants:
    http://iwa-mia.org/wp-content/uploads/2019/04/BSM_TG_Tech_Report_no_1_BSM1_General_Description.pdf
    The BSM1 model consists of 5 tanks, a clarifier, and an influent flow.

                                               IMLR
           ┌───────────────────────────────────────────────────────────────────┐
           │                                                                   │
           │                                                                   │
      ┌────▼─────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐    ┌─────┴────┐      xxxxxxxxxxxxxxx
      │          │     │          │     │          │     │          │    │          │       x clarifier x──────► Eff
    ──►  Tank 1  ├─────►  Tank 2  ├─────►  Tank 3  ├─────►  Tank 4  ├────►  Tank 5  ├────────►─────────x
      │          │     │          │     │    ▲     │     │    ▲     │    │     ▲    │         x───────x
      └────▲─────┘     └──────────┘     └────┼─────┘     └────┼─────┘    └─────┼────┘          x─────x
           │                                 │                │                │                x───x
           │                                 │                │                │                 xxx
           │           Aeration   ───────────┴────────────────┴────────────────┘                  │
           │                                                                                      │
           └──────────────────────────────────────────────────────────────────────────────────────┤
                                                RAS                                               │
                                                                                                  └────►
                                                                                                    WAS

    Each tank is modeled using the ASM1 activated sludge model.
    The state of the system is the biochemical composition of each tank + each layer of the clarifier.
    The actions are given by aeration to each of tanks 3,4,5, the recycle rates (IMLR and RAS), and the
    wasteage rate (WAS)
    """
    def __init__(self, cfg:  BSM1Config | None = None):
        if cfg is None:
            cfg = BSM1Config()
        self.cfg = cfg
        self.params = BSM1Params()
        self.params.integrator_steps_per_env_step = int(cfg.env_dt_minutes / 60 / 24 /
                                                        self.params.dt_integrator)
        self.t = 0.0
        self.effluent_buffer = np.ones([int(24*60/cfg.env_dt_minutes), 6])*np.nan
        self.effluent_buffer_idx = 0

        if cfg.influent_source == 'constant':
            self.influent = SteadyStateInfluent(self.params.influent_sensors)
        elif cfg.influent_source in ['dryinfluent', 'raininfluent', 'storminfluent']:
            self.influent = TimeSeriesInfluent(cfg.influent_source+'.csv', self.params.influent_sensors)
        elif cfg.influent_source == 'mixed':
            self.influent = MixedTimeSeriesInfluent(self.params.influent_sensors)
        else:
            raise ValueError(f'Unknown influent source {cfg.influent_source}')
        tank_volumes = [1000.0]*2 + [1333.0]*3
        self.tanks = [TankModel(self.params.tank_sensors[i], \
                                param_override={'volume': tank_volumes[i]}) for i in range(5)]
        self.clarifier = ClarifierModel(self.params.effluent_sensors, self.params.sludge_bed_height_sensor)

        if self.cfg.only_so5_action:
            action_range = np.array([self.params.aeration_range])
            if not self.cfg.direct_action_aeration:
                action_range = np.array([[0.,4]])
        else:
            action_range = np.vstack([np.tile(self.params.aeration_range,[3,1]),
                                    self.params.ILMR_range,
                                    self.params.RAS_range,
                                    self.params.WAS_range])
            if not self.cfg.direct_action_aeration:
                action_range[2,:] = np.array([0.,4])
            if not self.cfg.direct_action_imlr:
                action_range[3,:] = np.array([0,2])
        self.action_space = gym.spaces.Box(action_range[:,0], action_range[:,1], dtype=np.float64)

        n_obs = len(self.params.influent_sensors) + \
                sum(len(s) for s in self.params.tank_sensors) + \
                len(self.params.imlr_sensors) + \
                len(self.params.effluent_sensors) + \
                (1 if self.params.sludge_bed_height_sensor is not None else 0) + \
                10 # Number of performance metrics
        self.observation_space = gym.spaces.Box(-np.inf*np.ones(n_obs), np.inf*np.ones(n_obs),
                                                dtype=np.float64)

        self.sno_controller = PIController(10000., 0.025, 1.015, [0, 92230])
        self.ox_controller = PIController(25., 0.002, 0.001, [0, 360])

        self.warmup_plant()
        self.t = 0.0

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs = []

        if self.cfg.only_so5_action:
            action = np.array([240., 240., action[0], 5538.0, 18446, 385])
            if not self.cfg.direct_action_imlr:
                action[3] = 1.0

        for _step in range(max(self.params.integrator_steps_per_env_step, 1)):
            aeration, Q_IMLR, Q_RAS, Q_WAS, action_obs = self.calc_control_values(action)

            # Step the tank models and clarifier
            obs = []

            obs += self.influent.step(dt = self.params.dt_integrator)
            if self.cfg.no_recirc:
                IMLR_flow = FlowModel(self.tanks[-1].tank_flow.X, 0)
                RAS_flow = FlowModel(self.clarifier.under_flow.X, 0)
            else:
                IMLR_flow = FlowModel(self.tanks[-1].tank_flow.X, Q_IMLR)
                RAS_flow = FlowModel(self.clarifier.under_flow.X, Q_RAS)

            # Flow through the 5 tanks
            #   Add influent + RAS + IMLR to get first tank flow
            obs += self.tanks[0].step(self.influent.influent_flow + IMLR_flow + RAS_flow, aeration[0], \
                                      dt = self.params.dt_integrator)
            for i in range(1,5):
                obs += self.tanks[i].step(self.tanks[i-1].tank_flow, aeration[i], dt = self.params.dt_integrator)

            # IMLR sensors
            obs += [s.step(IMLR_flow) for s in self.params.imlr_sensors]

            # Flow through the clarifier
            clarifier_influent = FlowModel(self.tanks[-1].tank_flow.X, self.tanks[-1].tank_flow.Q - Q_IMLR)
            obs += self.clarifier.step(clarifier_influent, Q_RAS + Q_WAS, dt = self.params.dt_integrator)

            self.t += self.params.dt_integrator

        obs += self.calc_performance_metrics(Q_WAS, Q_RAS, Q_IMLR, aeration) # type:ignore

        if self.cfg.only_so5_action:
            action_obs = np.array([action_obs[2]]) # type: ignore

        return np.array(obs), 0.0, False, False, {'action_override': action_obs} # type: ignore

    def calc_control_values(self, action_in: np.ndarray) -> tuple[np.ndarray, float, float, float, np.ndarray]:
        action_obs = action_in.copy()
        is_baseline = self.t < self.cfg.baseline_duration

        # Tanks 3-4 and WAS and RAS are always direct control
        if is_baseline:
            # During the baseline period, we override the actions with the baseline values
            action_obs[:2] = 240.0  # Aeration for tanks 3 and 4
            action_obs[4:] = [18446, 385]  # RAS and WAS flow rates

        aeration = np.zeros(5)
        # First three actions are the aeration rates of tanks 3-5
        aeration[-3:-1] = action_obs[:2]
        if self.cfg.direct_action_aeration:
            if is_baseline:
                if self.cfg.ox_ctrl_enabled:
                    # Use the closed loop controller with fixed setpoint as a baseline comparison
                    error = 2.0 - self.tanks[-1].tank_flow.get('SO')
                    action_obs[2] = self.ox_controller.step(error, self.params.dt_integrator)
                else:
                    # Use a fixed setpoint for the baseline period
                    action_obs[2] = 84.0
            # Use the action as the aeration rate
            aeration[-1] = action_obs[2]
        else:
            # If setpoint control is used, assume that the closed loop controller is active
            if is_baseline:
                # Use the closed loop controller with fixed setpoint as a baseline comparison
                action_obs[2] = 2.0 # Override the action with the fixed controller setpoint
            error = action_obs[2] - self.tanks[-1].tank_flow.get('SO')
            aeration[-1] = self.ox_controller.step(error, self.params.dt_integrator)

        # Next 3 actions are the controlled flow rates for IMLR, RAS, and WAS (see above diagram)
        Q_RAS: float = action_obs[4] # Return activated sludge flow
        Q_WAS: float = action_obs[5] # Waste activated sludge flow
        if self.cfg.direct_action_imlr:
            if is_baseline:
                if self.cfg.sno_ctrl_enabled:
                    # Use the closed loop controller with fixed setpoint as a baseline comparison
                    error = 1.0 - self.tanks[1].tank_flow.get('SNO')
                    action_obs[3] = self.sno_controller.step(error, self.params.dt_integrator)
                else:
                    # Use a fixed setpoint for the baseline period
                    action_obs[3] = 55338.0
            Q_IMLR:float = action_obs[3] # Override the action with the controller output
        else:
            # If setpoint control is used, assume that the closed loop controller is active
            if is_baseline:
                # Use the closed loop controller with fixed setpoint as a baseline comparison
                action_obs[3] = 1.0 # Override the action with the fixed controller setpoint
            error = action_obs[3] - self.tanks[1].tank_flow.get('SNO')
            Q_IMLR:float = self.sno_controller.step(error, self.params.dt_integrator)

        return aeration, Q_IMLR, Q_RAS, Q_WAS, action_obs

    def warmup_plant(self):
        # The tanks need to be populated the tanks with appropriate solids and bacteria. We use the steady state
        # AS tank inlet values with no recirculation to get to this initial condition
        _cfg = self.cfg
        _influent = self.influent

        self.cfg = BSM1Config(
            name='BSM1-v0',
            no_recirc=True,
            influent_source='constant',
            ox_ctrl_enabled=False,
            sno_ctrl_enabled=False,
            baseline_duration=0.0,
            direct_action_aeration=True,
            direct_action_imlr=True,
            env_dt_minutes=120.0,
        )
        Q_IMLR = 55338
        Q_RAS = 18446
        Q_WAS = 385
        steady_state_aeration = np.array([0., 0., 240., 240., 84.])
        warmup_flow = FlowModel(X = np.array([30.0,  14.6116,  1149.1183, 89.3302, 2542.1684, 148.4614, 448.1754, \
                                     0.39275,   8.3321,  7.6987, 1.9406,  5.6137,  4.7005, 3282.9402, 15.0]), \
                                Q = 92230)
        self.influent = InfluentModel(self.params.influent_sensors)
        self.influent.influent_flow = warmup_flow
        action = np.hstack([steady_state_aeration[2:], Q_IMLR, Q_RAS, Q_WAS])

        print('Warming up plant')
        dt = self.params.dt_integrator * self.params.integrator_steps_per_env_step
        obs = []
        for _ in range(int(1/dt)):
            obs = self.step(action)
        print('Plant ready for closed loop evaluation')

        self.cfg = _cfg
        self.influent = _influent
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.influent.reset()
        for t in self.tanks:
            t.reset()
        for s in self.params.imlr_sensors:
            s.reset(FlowModel(self.tanks[-1].tank_flow.X, 0))
        self.clarifier.reset()

        obs, _, _, _, _ = self.warmup_plant()
        self.t = 0.0
        return np.array(obs), {}

    def calc_performance_metrics(self, Q_WAS: float, Q_RAS: float, Q_IMLR: float, aeration: np.ndarray) -> list[float]:
        eff = self.clarifier.effluent_flow
        was = FlowModel(self.clarifier.under_flow.X, Q_WAS)

        # Effluent quality limits
        SNKJ = eff.get('SNH') + eff.get('SND') + eff.get('XND') + \
               0.08 * (eff.get('XBH') + eff.get('XBA')) + 0.06 * (eff.get('XP') + eff.get('XI'))
        N = SNKJ + eff.get('SNO') # Total nitrogen < 18
        COD = eff.get('SS') + eff.get('SI') + eff.get('XS') + eff.get('XI') +\
              eff.get('XBH') + eff.get('XBA') + eff.get('XP')
        SNH = eff.get('SNH')
        TSS = eff.get('TSS')
        BOD = 0.25 * (eff.get('SS') + eff.get('XS') + (1 - 0.08) * (eff.get('XBH') + eff.get('XBA')))

        # Single day rolling average of effluent
        self.effluent_buffer[self.effluent_buffer_idx,:] = eff.get('Q') * np.array([1, N, COD, SNH, TSS, BOD])
        self.effluent_buffer_idx = (self.effluent_buffer_idx + 1) % self.effluent_buffer.shape[0]
        daily_average_effluent = np.nanmean(self.effluent_buffer[:,1:],0) / np.nanmean(self.effluent_buffer[:,0])

        # Effluent quality index (EQI)
        EQI = 1/1000 * ( 2 * eff.get('TSS') + COD + 30 * SNKJ + 10 * eff.get('SNO') + 2 * BOD ) * eff.Q

        # Sludge production
        SPW = 0.75/1000 * (was.get('XS') + was.get('XI') + was.get('XBH') + was.get('XBA') + was.get('XP')) * was.Q
        SPE = 0.75/1000 * (eff.get('XS') + eff.get('XI') + eff.get('XBH') + eff.get('XBA') + eff.get('XP')) * eff.Q

        # Pumping energy
        PE = 0.004 * Q_IMLR + 0.008 * Q_RAS + 0.05 * Q_WAS

        # Aeration energy
        AE = 8/(1800) * sum([t.tank_params.volume * aeration[i] for i,t in enumerate(self.tanks)])

        # Overall cost index
        OCI = AE + PE + 5*SPW
        return [N, COD, SNH, TSS, BOD, EQI, SPW, SPE, PE, AE, OCI, *daily_average_effluent.tolist()]

@dataclass
class BSM1Params:
    # Integrator timestep
    dt_integrator: float = 0.1 / 60 / 24  # Integrator step is 6 seconds (in days)
    # Env timestep
    integrator_steps_per_env_step: int = 150  # Env step is 15 minutes
    # Action ranges
    aeration_range: tuple = (0,360)
    ILMR_range: tuple = (0,92230)
    RAS_range: tuple = (0,36892)
    WAS_range: tuple = (0,1844.6)
    influent_sensors:  tuple[SensorModel, ...]  =        (SensorModel('Q',    [0,1e5], 'A', 2500),
                                                          SensorModel('SNH',  [0,50],  'B', 1.25),
                                                          SensorModel('SO',   [0,10],  'A', 0.25),
                                                          SensorModel('SNO',  [0,20],  'B', 0.5),
                                                          SensorModel('SALK', [0,20],  'B', 0.5))
    tank_sensors: tuple[tuple[SensorModel, ...], ...] = ((),                                         # Tank 1
                                                         (SensorModel('SNO',  [0,20],  'B', 0.5),),  # Tank 2
                                                         (),                                         # Tank 3
                                                         (),                                         # Tank 4
                                                         (SensorModel('SO',   [0,10],  'A', 0.25),)) # Tank 5
    imlr_sensors: tuple[SensorModel, ...] =              (SensorModel('Q',    [0,1e5], 'A', 2500),
                                                          SensorModel('TSS',  [0,1e4], 'A', 250))
    sludge_bed_height_sensor: SbhSensorModel | None = field(default_factory = \
                                                            lambda: SbhSensorModel([0,5], 'A', 0.125, 1000))
    effluent_sensors:  tuple[SensorModel, ...]  =        (SensorModel('SNH',  [0,50],  'B', 1.25),
                                                          SensorModel('SNO',  [0,20],  'B', 0.5))


env_group.dispatcher(BSM1Config(),BSM1Env)
