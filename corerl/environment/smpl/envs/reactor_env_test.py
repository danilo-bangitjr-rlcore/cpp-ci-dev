from reactorenv import ReactorEnvGym

# set up the environment
NUM_STEPS = 1000
normalize = False
env = ReactorEnvGym(normalize=normalize)
init_state = [0.8778252, 51.34660837, 0.659] # init with steady state (setpoint)
# init_state = env.sample_initial_state(no_sample=True)
state = env.reset(initial_state=init_state)

state, reward, done, info = env.step([26.85, 0.1]) # feed in the steady action (setpoint action)

print("state", state) # should be [0.8778252, 51.34660837, 0.659], we still get the same steady state

"""
total_reward = 0.0
for step in range(NUM_STEPS):
    state, reward, done, info = env.step(action)
    total_reward += reward
    if step % 1000 == 0:
        print("reward, total_reward:", reward, total_reward)
"""
