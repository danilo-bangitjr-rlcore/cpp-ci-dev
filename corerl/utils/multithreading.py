"""
THIS FILE IS DEPRESCATED. WILL UPDATE IN THE FUTURE AS NEEDED
"""

# import threading
# import queue
#
#
# def get_all_queue(q):
#     result_list = []
#     while not q.empty():
#         result_list.append(q.get())
#
#     return result_list
#
# def update_agent_until(agent, interaction_done):
#     while not interaction_done.is_set():
#         if len(agent.buffer.data) == 0:
#             pass
#         else:
#             agent.do_eval()
#
#
# def update_agent(agent, interaction_done):
#     if len(agent.buffer.data) == 0:
#         pass
#     else:
#         agent.do_eval()
#     interaction_done.wait()
#
#
# def do_interaction(interaction, state, action, interaction_done, transition_queue):
#     transitions, env_infos = interaction.step(action)
#     for transition in transitions:
#         transition_queue.put(transition)
#     interaction_done.set()



# def multithreaded_step(agent, interaction, state, action):
#     transition_queue = queue.Queue()
#     interaction_done = threading.Event()
#     agent_thread = threading.Thread(target=update_agent, args=(agent, interaction_done))
#     interaction_thread = threading.Thread(target=do_interaction, args=(interaction, state, action,
#                                                                        interaction_done, transition_queue))
#     interaction_thread.start()
#     agent_thread.start()
#     interaction_thread.join()
#     agent_thread.join()
#
#     transitions = get_all_queue(transition_queue)
#     return transitions
