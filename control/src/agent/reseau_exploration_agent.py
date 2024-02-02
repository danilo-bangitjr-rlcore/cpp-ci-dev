import sched
import time
import numpy as np
from datetime import date, timedelta

class ReseauExplorationAgent:
	def __init__(self, cfg):
		self.env = cfg.env
		self.reset_fpm = cfg.reset_fpm
		self.reset_duration = cfg.reset_duration

		# ORP Delay Experiment
		# self.orp_threshold_factor = cfg.orp_threshold_factor
		self.orp_delay_fpm = cfg.init_fpm
		self.orp_delay_start_times = cfg.orp_delay_start_times
		self.orp_delay_duration = cfg.orp_delay_duration
		self.orp_delay_iter_num = 0

		# Varying FPM Dataset
		self.reset = cfg.reset
		self.fpm_min = cfg.fpm_min
		self.fpm_max = cfg.fpm_max
		self.num_fpms = cfg.num_fpms
		self.fpm_list = np.linspace(self.fpm_min, self.fpm_max, num=self.num_fpms)
		self.fpm_list = np.round(self.fpm_list, 2)
		np.random.shuffle(self.fpm_list)
		self.varying_fpm_iter_num = 0

	def get_reference_orp(self):
		curr_obs, _, _, _, _ = self.env.get_observation(0)
		# Average ORP readings from last 5 seconds
		orp_ref = curr_obs['ait301_pv'][-5:].mean()
		self.orp_threshold = orp_ref * self.orp_threshold_factor
		print("ORP Threshold =", self.orp_threshold)

	def reset_env(self):
		print("Resetting Environment For " + str(self.reset_duration) + " Seconds")
		self.env.reset()
		time.sleep(self.reset_duration)

	def orp_delay_agent(self, scheduler):
		print("Run ORP Sensor Delay Experiment Iteration")
		self.orp_delay_iter_num += 1
		
		# Reset the environment
		# self.reset_env()
		
		# Update ORP threshold
		# self.get_reference_orp()
		
		# Set the FPM to the current ORP Delay Experiment FPM
		print("Set FPM To " + str(self.orp_delay_fpm) + " For " + str(self.orp_delay_duration) + " Seconds")
		self.env.take_action([self.orp_delay_fpm])
		
		# Keep this FPM constant long enough to ensure Victor will be there to sample the chlorine concentration (~2 hours)
		time.sleep(self.orp_delay_duration)
		
		# Determine if ORP exceeds threshold and update FPM for next iteration
		"""
		curr_obs, _, _, _, _ = self.env.get_observation(0)
		orp_ref = curr_obs['ait301_pv'][-5:].mean()
		print("Current ORP:", orp_ref)
		if orp_ref > self.orp_threshold:
			self.orp_delay_fpm -= 25
			print("Current ORP Above Threshold. New ORP Sensor Delay FPM =", self.orp_delay_fpm)
		else:
			self.orp_delay_fpm += 50
			print("Current ORP Below Threshold. New ORP Sensor Delay FPM =", self.orp_delay_fpm)
		"""
		self.orp_delay_fpm += 5
		
		# Schedule next ORP delay iteration
		if self.orp_delay_iter_num % len(self.orp_delay_start_times) == 0:
			s = str(date.today() + timedelta(days=1)) + " " + self.orp_delay_start_times[0]
			print(s)
			# Schedule next morning
			orp_delay_start_time = time.strptime(str(date.today() + timedelta(days=1)) + " " + self.orp_delay_start_times[0], '%Y-%m-%d %H:%M:%S')
			orp_delay_start_time = time.mktime(orp_delay_start_time)
			scheduler.enterabs(orp_delay_start_time, 1, self.orp_delay_agent, argument=[scheduler])
		else:
			# Schedule this afternoon
			orp_delay_start_time = time.strptime(str(date.today()) + " " + self.orp_delay_start_times[1], '%Y-%m-%d %H:%M:%S')
			orp_delay_start_time = time.mktime(orp_delay_start_time)
			scheduler.enterabs(orp_delay_start_time, 1, self.orp_delay_agent, argument=[scheduler])


	def fpm_exploration_agent(self, scheduler):
		print("Run Varying FPM Experiment Iteration")
		# Reset environment before trying an FPM in self.fpm_list
		"""
		if self.reset:
			self.reset_env()
		"""
		# Perform environment resets only for the first half of tested FPMs
		if self.varying_fpm_iter_num < self.num_fpms:
			self.reset_env()

		# Set the FPM to the current FPM in self.fpm_list
		if self.varying_fpm_iter_num > 3*self.num_fpms:
			curr_fpm = self.reset_fpm
		else:
			curr_fpm = self.fpm_list[self.varying_fpm_iter_num % len(self.fpm_list)]
		print("Set FPM To " + str(curr_fpm) + " For " + str(self.reset_duration) + " Seconds")
		self.env.take_action([curr_fpm])

		# Keep the FPM constant for a long enough duration
		time.sleep(self.reset_duration)

		self.varying_fpm_iter_num += 1

		# Once we've gone through each FPM in the list, reshuffle the list
		if self.varying_fpm_iter_num % len(self.fpm_list) == 0:
			np.random.shuffle(self.fpm_list)
			print("Shuffled FPM List:")
			print(self.fpm_list)

		# Schedule next FPM exploration iteration
		scheduler.enter(1, 2, self.fpm_exploration_agent, argument=[scheduler])


