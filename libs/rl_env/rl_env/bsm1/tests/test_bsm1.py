import time

import numpy as np

from rl_env.bsm1.asm1 import TankModel
from rl_env.bsm1.bsm1 import BSM1Config, BSM1Env
from rl_env.bsm1.clarifier import ClarifierModel
from rl_env.bsm1.flows import FlowModel as FM
from rl_env.bsm1.flows import TimeSeriesInfluent

# Steady state values from BSM1 paper (page 36) for the input to the tanks and each of the 5 tanks
#    SI      SS        XI        XS        XBH       XBA      XP        SO         SNO      SNH        SND     XND     SALK     TSS       TEMP                          # noqa: E501
steady_state_influent = np.array([
     30.0,  69.5,       51.2,   202.32,     28.17,     0.0,      0.0,    0.0,       0.0,    31.56,   6.95,   10.59,    7.0,     211.2675, 15.0,                         # noqa: E501
], dtype=float)
steady_state_values = np.array([
    [30.0,  14.6116,  1149.1183, 89.3302, 2542.1684, 148.4614, 448.1754, 0.39275,   8.3321,  7.6987, 1.9406,  5.6137,  4.7005, 3282.9402, 15.0], # AS Influent          # noqa: E501
    [30.0,   2.8082,  1149.1183, 82.1349, 2551.7631, 148.3886, 448.8459, 0.0042984, 5.3699,  7.9179, 1.2166,  5.2849,  4.9277, 3285.1880, 15.0], # Tank 1               # noqa: E501
    [30.0,   1.4588,  1149.1182, 76.3862, 2553.3824, 148.3083, 449.5167, 0.0000631, 3.6619,  8.3445, 0.8821,  5.0291,  5.0802, 3282.5339, 15.0], # Tank 2               # noqa: E501
    [30.0,   1.1495,  1149.1182, 64.8549, 2557.1288, 148.9404, 450.4123, 1.7184,    6.5408,  5.5480, 0.8289,  4.3924,  4.6748, 3277.8410, 15.0], # Tank 3               # noqa: E501
    [30.0,   0.9953,  1149.1182, 55.6940, 2559.1800, 149.5262, 451.3087, 2.4289,    9.2990,  2.9674, 0.7668,  3.8790,  4.2935, 3273.6203, 15.0], # Tank 4               # noqa: E501
    [30.0,   0.8895,  1149.1182, 49.3056, 2559.3410, 149.7963, 452.2051, 0.49094,  10.4152,  1.7334, 0.6883,  3.5272,  4.1256, 3269.8246, 15.0], # Tank 5               # noqa: E501
    [30.0,   0.88949, 2247.0367, 96.4143, 5004.6489, 292.9183, 884.2618, 0.49094,  10.4152,  1.7334, 0.68828, 6.8972,  4.1256, 6393.9599, 15.0], # Clarifier underflow  # noqa: E501
    [30.0,   0.88949,    4.3918, 0.18844,    9.7815,   0.57251,  1.7283, 0.49094,  10.4152,  1.7334, 0.68828, 0.01348, 4.1256,   12.4969, 15.0],  # Clarifier effluent  # noqa: E501
], dtype=float)

steady_state_aeration = np.array([0., 0., 240., 240., 84.])

def test_tankmodel_steady_state():
    # Initialize the 5 tanks as in the BSM1 model
    volumes = [1000.0]*2 + [1333.0]*3
    tanks = [TankModel(param_override={"volume": vol}) for vol in volumes]

    tank_influents = [FM(X=steady_state_values[i,:], Q=92230) for i in range(5)]  # each tank steady state influent
    start_time = time.time()
    dt = 0.1/60/24
    chain_tanks = True
    for i in range(int(10/dt)):
        # Print progress every 1 day and estimate integration speed
        if i % int(1/dt) == 0:
            print(f"Day {int(i*dt)} / {10} - Integration speed: {i*dt/(time.time() - start_time):.2f} days/sec")

        # Step the tanks with the steady state influent
        if chain_tanks:
            tanks[0].step(tank_influents[0], dt=dt)
            for j in range(1,len(tanks)):
                tanks[j].step(tanks[j-1].tank_flow, aeration=steady_state_aeration[j], dt=dt)
        else:
            for j, tank in enumerate(tanks):
                tank.step(tank_influents[j], aeration=steady_state_aeration[j], dt=dt)

    # Check the final state of each tank
    for i, tank in enumerate(tanks):
        expected = steady_state_values[i+1,:]  # Skip the influent state
        # Allow a small tolerance for numerical error
        np.testing.assert_allclose(tank.tank_flow.X, expected, rtol=1e-2, atol=1e-2)

    print("ASM1 steady state matches expected values.")

def test_clarifiermodel_steady_state():
    # Initialize the clarifier
    clarifier = ClarifierModel()
    influent = FM(X=steady_state_values[5,:], Q=36892)
    underflow_Q = 18446 + 385

    start_time = time.time()
    dt = 0.1/60/24
    for i in range(int(10/dt)):
        # Print progress every 1 day and estimate integration speed
        if i % int(1/dt) == 0:
            print(f"Day {int(i*dt)} / {10} - Integration speed: {i*dt/(time.time() - start_time):.2f} days/sec")

        clarifier.step(influent, underflow_Q, dt)

    # Check the final state
    expected_underflow = steady_state_values[6,:]
    expected_effluent = steady_state_values[7,:]

    # Allow a small tolerance for numerical error
    np.testing.assert_allclose(clarifier.under_flow.X, expected_underflow, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(clarifier.effluent_flow.X, expected_effluent, rtol=1e-2, atol=1e-2)

    print("Clarifier steady state matches expected values.")

def test_bsm1_model_open_loop_no_recirc_steady_state():
    bsm1 = BSM1Env(BSM1Config(no_recirc = True, baseline_duration=0.0))
    bsm1.influent.influent_flow = FM(X=steady_state_values[0,:], Q=92230)
    Q_IMLR = 55338
    Q_RAS = 18446
    Q_WAS = 385
    action = np.hstack([steady_state_aeration[2:], Q_IMLR, Q_RAS, Q_WAS])

    start_time = time.time()
    dt = bsm1.params.dt_integrator * bsm1.params.integrator_steps_per_env_step
    for i in range(int(100/dt)):
        # Print progress every 1 day and estimate integration speed
        if i % int(1/dt) == 0:
            print(f"Day {int(i*dt)} / {100} - Integration speed: {i*dt/(time.time() - start_time):.2f} days/sec")

        bsm1.step(action)

    # AS Tanks
    for i, tank in enumerate(bsm1.tanks):
        expected = steady_state_values[i+1,:]  # Skip the AS influent state
        # Allow a small tolerance for numerical error
        np.testing.assert_allclose(tank.tank_flow.X, expected, rtol=1e-2, atol=1e-2)

    # Clarifier underflow
    expected_UF = steady_state_values[6,:]
    np.testing.assert_allclose(bsm1.clarifier.under_flow.X, expected_UF, rtol=1e-2, atol=1e-2)

    # Effluent
    expected_eff = steady_state_values[7,:]
    np.testing.assert_allclose(bsm1.clarifier.effluent_flow.X, expected_eff, rtol=1e-2, atol=1e-2)

    print("BSM1 system without re-circ steady state matches expected values.")

def test_bsm1_model_open_loop_steady_state():
    bsm1 = BSM1Env(BSM1Config(baseline_duration=0.0))
    Q_IMLR = 55338
    Q_RAS = 18446
    Q_WAS = 385
    action = np.hstack([steady_state_aeration[2:], Q_IMLR, Q_RAS, Q_WAS])
    bsm1.influent.influent_flow = FM(X=steady_state_influent, Q=18446)

    dt = bsm1.params.dt_integrator * bsm1.params.integrator_steps_per_env_step
    start_time = time.time()
    for i in range(int(100/dt)):
        # Print progress every 1 day and estimate integration speed
        if i % int(1/dt) == 0:
            print(f"Day {int(i*dt)} / {100} - Integration speed: {i*dt/(time.time() - start_time):.2f} days/sec")

        bsm1.step(action)

    # Test all steady_state_values (need to recreate some flows here)
    IMLR_flow = FM(bsm1.tanks[-1].tank_flow.X, Q_IMLR)
    RAS_flow = FM(bsm1.clarifier.under_flow.X, Q_RAS)
    AS_Flow = bsm1.influent.influent_flow + IMLR_flow + RAS_flow

    # AS Influent
    expected_AS = steady_state_values[0,:]
    np.testing.assert_allclose(AS_Flow.X, expected_AS, rtol=2e-2, atol=1e-2)

    # AS Tanks
    for i, tank in enumerate(bsm1.tanks):
        expected = steady_state_values[i+1,:]  # Skip the AS influent state
        # Allow a small tolerance for numerical error
        np.testing.assert_allclose(tank.tank_flow.X, expected, rtol=2e-2, atol=1e-2)

    # Clarifier underflow
    expected_UF = steady_state_values[6,:]
    np.testing.assert_allclose(RAS_flow.X, expected_UF, rtol=2e-2, atol=1e-2)

    # Effluent
    expected_eff = steady_state_values[7,:]
    np.testing.assert_allclose(bsm1.clarifier.effluent_flow.X, expected_eff, rtol=2e-2, atol=1e-2)

    print("BSM1 system steady state matches expected values.")

def test_bsm1_model_open_loop_dynamic():
    print("Setting up steady state conditions")
    bsm1 = BSM1Env(BSM1Config(baseline_duration=0.0))
    Q_IMLR = 55338
    Q_RAS = 18446
    Q_WAS = 385
    action = np.hstack([steady_state_aeration[2:], Q_IMLR, Q_RAS, Q_WAS])
    bsm1.influent.influent_flow = FM(X=steady_state_influent, Q=18446)

    dt = bsm1.params.dt_integrator * bsm1.params.integrator_steps_per_env_step
    start_time = time.time()
    for i in range(int(100/dt)):
        # Print progress every 1 day and estimate integration speed
        if i % int(1/dt) == 0:
            print(f"Day {int(i*dt)} / {100} - Integration speed: {i*dt/(time.time() - start_time):.2f} days/sec")

        bsm1.step(action)
    print("Steady state conditions reached")

    # Influent test
    bsm1.influent = TimeSeriesInfluent(file_name = 'dryinfluent.csv')
    start_time = time.time()
    X_ = []
    Q_ = []
    O_ = []
    for i in range(int(14/dt)):
        if i % int(1/dt) == 0:
            print(f"Day {int(i*dt)} / {14} - Integration speed: {i*dt/(time.time() - start_time):.2f} days/sec")
        obs, _, _, _, _ = bsm1.step(action)

        if i * dt >= 7-1e-3:
            X_ += [bsm1.clarifier.effluent_flow.X.copy()]
            Q_ += [bsm1.clarifier.effluent_flow.Q]
            O_ += [obs]


    # Expected values come from appendix 2
    expected_X = np.array([30.0, 0.9725, 4.58, 0.2231, 10.22, 0.5421, 1.757, 0.7462, \
                           8.801, 4.794, 0.7308, 0.01571, 4.46, 12.99, 15])
    expected_Q = 18061.3325
    expected_O = np.array([6700, 2436, 233, 388.2, 3341])
    expected_viol_dur = [0.5761, 0, 4.403, 0, 0]
    expected_viol_num = [5, 0, 7, 0, 0]
    actual_Q = np.average(np.array(Q_))
    # FLow-weighted average of concentrations
    actual_X = np.average(np.array(X_)*np.tile(Q_,[15,1]).T,0)/actual_Q
    # Performance metrics (EQI, Waste Sludge, Effluent Sludge, Pumping Energy, Aeration Energy)
    actual_O = np.average(np.array(O_)[:,-11:-6],0)
    # Constraint violations ()
    ulim = [18, 100, 4, 30, 10]
    viol_dur = np.zeros(5)
    viol_num = np.zeros(5)
    for i, idx in enumerate(range(-16,-11)):
        violations = np.array(O_)[:,idx] > ulim[i]
        viol_dur[i] = dt*np.sum(violations)
        viol_num[i] = sum(np.diff(1*violations) > 0)


    np.testing.assert_allclose(actual_X, expected_X, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(actual_Q, expected_Q, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(actual_O, expected_O, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(viol_dur, expected_viol_dur, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(viol_num, expected_viol_num, rtol = 1e-1, atol = 1)

    print("BSM1 system dynamic results match expected values.")

def test_bsm1_model_closed_loop_dynamic():
    print("Setting up steady state conditions")
    bsm1 = BSM1Env(BSM1Config(ox_ctrl_enabled=True, sno_ctrl_enabled=True,
                              direct_action_aeration=False, direct_action_imlr=False, baseline_duration=0.0))
    so5_sp = 2.0
    sno2_sp = 1.0
    Q_RAS = 18446
    Q_WAS = 385
    action = np.hstack([steady_state_aeration[2:-1], so5_sp, sno2_sp, Q_RAS, Q_WAS])
    bsm1.influent.influent_flow = FM(X=steady_state_influent, Q=18446)

    dt = bsm1.params.dt_integrator * bsm1.params.integrator_steps_per_env_step
    start_time = time.time()
    for i in range(int(100/dt)):
        # Print progress every 1 day and estimate integration speed
        if i % int(1/dt) == 0:
            print(f"Day {int(i*dt)} / {100} - Integration speed: {i*dt/(time.time() - start_time):.2f} days/sec")
        bsm1.step(action)
    print("Steady state conditions reached")

    # Influent test
    bsm1.influent = TimeSeriesInfluent(file_name = 'dryinfluent.csv')
    start_time = time.time()
    X_O_SNO_ = []
    O_ = []
    for i in range(int(14/dt)):
        if i % int(1/dt) == 0:
            print(f"Day {int(i*dt)} / {14} - Integration speed: {i*dt/(time.time() - start_time):.2f} days/sec")
        obs, _, _, _, _ = bsm1.step(action)

        if i * dt >= 7-1e-3:
            X_O_SNO_ += [[bsm1.tanks[-1].tank_flow.X[7], bsm1.tanks[1].tank_flow.X[8]]]
            O_ += [obs]

    expected_X = np.array([so5_sp, sno2_sp])
    expected_O = np.array([6123, 2440, 234.8, 241, 3698])
    expected_viol_dur = [1.2813, 0, 1.1979, 0, 0]
    expected_viol_num = [7, 0, 5, 0, 0]

    # Performance metrics (EQI, Waste Sludge, Effluent Sludge, Pumping Energy, Aeration Energy)
    actual_O = np.average(np.array(O_)[:,-11:-6],0)
    average_N_NO = np.average(np.array(O_)[:,[-14,-16]],0)
    print(average_N_NO)
    # Constraint violations ()
    ulim = [18, 100, 4, 30, 10]
    viol_dur = np.zeros(5)
    viol_num = np.zeros(5)
    for i, idx in enumerate(range(-16,-11)):
        violations = np.array(O_)[:,idx] > ulim[i]
        viol_dur[i] = dt*np.sum(violations)
        viol_num[i] = sum(np.diff(1*violations) > 0)

    # FLow-weighted average of concentrations
    average_err = np.average(np.array(X_O_SNO_) - np.tile(expected_X,[len(X_O_SNO_),1]),0)
    average_abs_err = np.average(np.abs(np.array(X_O_SNO_) - np.tile(expected_X,[len(X_O_SNO_),1])),0)

    # Check that the controlled variables are close to their desired values
    np.testing.assert_allclose(average_err, np.zeros(2), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(average_abs_err, np.zeros(2), atol=0.25)
    # Check that the performance matches expectation
    np.testing.assert_allclose(actual_O, expected_O, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(viol_dur, expected_viol_dur, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(viol_num, expected_viol_num, rtol = 1e-1, atol = 1)

    print("BSM1 system dynamic results with closed loop match expected values.")

if __name__ == "__main__":
    test_tankmodel_steady_state()
    test_clarifiermodel_steady_state()
    test_bsm1_model_open_loop_no_recirc_steady_state()
    test_bsm1_model_open_loop_steady_state()
    test_bsm1_model_open_loop_dynamic()
    test_bsm1_model_closed_loop_dynamic()
