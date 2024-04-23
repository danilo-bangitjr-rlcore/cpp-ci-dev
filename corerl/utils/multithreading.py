import threading
import queue


def update_agent_until(agent, interaction_done):
    while not interaction_done.is_set():
        if len(agent.buffer.data) == 0:
            pass
        else:
            agent.update()


def update_agent(agent, interaction_done):
    agent.update()
    interaction_done.wait()


def do_interaction(interaction, state, action, interaction_done, transition_queue):
    next_state, reward, done, truncate, env_info = interaction.step(action)
    transition = (state, action, reward, next_state, done, truncate)
    interaction_done.set()
    transition_queue.put(transition)


def multithreaded_step(agent, interaction, state, action):
    transition_queue = queue.Queue()
    interaction_done = threading.Event()
    agent_thread = threading.Thread(target=update_agent, args=(agent, interaction_done))
    interaction_thread = threading.Thread(target=do_interaction, args=(interaction, state, action,
                                                                       interaction_done, transition_queue))
    interaction_thread.start()
    agent_thread.start()
    interaction_thread.join()
    agent_thread.join()

    transition = transition_queue.get()
    return transition
