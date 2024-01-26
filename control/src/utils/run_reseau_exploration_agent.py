import sched
import time
from datetime import date, timedelta

def run_reseau_exploration_agent(agent, cfg):
    scheduler = sched.scheduler(time.time, time.sleep)

    # Schedule First ORP Sensor Delay Experiment Iteration
    orp_delay_start_time = time.strptime(str(date.today()) + " " + cfg.orp_delay_start_times[0], '%Y-%m-%d %H:%M:%S')
    orp_delay_start_time = time.mktime(orp_delay_start_time)
    scheduler.enterabs(orp_delay_start_time, 1, agent.orp_delay_agent, argument=[scheduler])

    # Schedule First Varying FPM Experiment Iteration
    scheduler.enter(1, 2, agent.fpm_exploration_agent, argument=[scheduler])

    scheduler.run(blocking=True)