import torch
import torch.nn as nn
import numpy as np
import src.utils.utils as utils
import src.network.torch_utils as torch_utils
from src.network.networks import FC
from src.component.buffer import Buffer
from src.component.normalizer import init_normalizer
from src.network.factory import init_custom_network, init_optimizer

# Training GVF from offline dataset
class GVF:
	def __init__(self, cfg):
		self.seed = cfg.seed
		self.device = cfg.device
		self.epochs = cfg.epochs
		self.parameters_dir = cfg.parameters_path

		self.gamma = cfg.gamma
		self.batch_size = cfg.batch_size
		self.polyak = cfg.polyak
        self.use_target_network = cfg.use_target_network
        self.target_network_update_freq = 1

        self.train_split = cfg.train_split
        self.validation_split = cfg.validation_split
        self.test_split = cfg.test_split

        self.train_loss = []

        # Initialize normalizers used for constructing state
		self.obs_normalizer = init_normalizer(cfg.obs_normalizer, 
			type('obj', (object,), {'scaler': cfg.obs_scale,  'bias': cfg.obs_bias}))
        self.cumulant_normalizer = init_normalizer(cfg.cumulant_normalizer,
        	type('obj', (object,), {'scaler': cfg.cumulant_scale,  'bias': cfg.cumulant_bias}))

		# Load raw offline dataset
		if cfg.offline_src == "CSV":
			self.dates, self.raw_observations, self.raw_cumulants = utils.load_offline_logs(cfg)
			self.total_samples = len(self.raw_observations)
			self.num_features = len(self.raw_observations.columns)
		else:
			raise NotImplementedError

		# Produce train, validation, test splits
		self.get_train_split()
		self.get_validation_split()
		self.get_test_split()

		# Produce augmented offline dataset (normalize observations and cumulants and construct states from observations)
		self.augment_dataset()

		# Get state dim from state constructor
		self.state_dim = 10

		# Initialize and load replay buffer with augmented training dataset
		self.buffer = Buffer(self.train_samples - 1, self.batch_size, self.seed)
		actions = np.zeros(self.train_samples)
		dones = np.zeros(self.train_samples)
		truncates = np.zeros(self.train_samples)
		self.buffer.load(self.train_states, actions, self.train_cumulants, dones, truncates)

        # Didn't use Target Network in GVF paper but giving us option
		self.gvf = init_custom_network(cfg.gvf, self.device, self.state_dim, cfg.hidden_gvf, 1,
                                             cfg.activation, "None", cfg.layer_init, cfg.layer_norm)
		self.gvf_target = init_custom_network(cfg.gvf, self.device, self.state_dim, cfg.hidden_gvf, 1,
                                             cfg.activation, "None", cfg.layer_init, cfg.layer_norm)
		self.gvf_target.load_state_dict(self.gvf.state_dict())
		self.gvf_optimizer = init_optimizer(cfg.optimizer, list(self.gvf.parameters()), cfg.lr_gvf)

    # Produce new offline dataset with normalized observations
    def augment_dataset(self):
    	# Normalize observations and construct state
    	self.train_states = self.train_raw_obs
    	self.val_states = self.val_raw_cumulants
    	self.test_states = self.test_raw_cumulants

    	# Normalize cumulants
    	self.train_cumulants = self.train_raw_cumulants
    	self.val_cumulants = self.val_raw_cumulants
    	self.test_cumulants = self.test_raw_cumulants

    def get_train_split(self):
    	self.train_samples = self.train_split * self.total_samples
    	self.train_dates = self.dates[:self.train_samples]
    	self.train_raw_obs = self.raw_observations[:self.train_samples]
    	self.train_raw_cumulants = self.raw_cumulants[:self.train_samples]

    def get_validation_split(self):
    	self.validation_samples = self.validation_split * self.total_samples
    	self.val_dates = self.dates[self.train_samples : self.train_samples + self.validation_samples]
    	self.val_raw_obs = self.raw_observations[self.train_samples : self.train_samples + self.validation_samples]
    	self.val_raw_cumulants = self.raw_cumulants[self.train_samples : self.train_samples + self.validation_samples]

    def get_test_split(self):
    	self.test_samples = self.test_split * self.total_samples
    	self.test_dates = self.dates[self.train_samples + self.validation_samples:]
    	self.test_raw_obs = self.raw_observations[self.train_samples + self.validation_samples:]
    	self.test_raw_cumulants = self.raw_cumulants[self.train_samples + self.validation_samples:]

    def get_data(self):
        states, _, cumulants, next_states, _, _ = self.buffer.sample()
        states = torch_utils.tensor(states, self.device)
        cumulants = torch_utils.tensor(cumulants, self.device)
        next_states = torch_utils.tensor(next_states, self.device)
        data = {
            'states': states,
            'cumulants': cumulants,
            'next_states': next_states,
        }
        return data

    def get_gvf_value(self, state, with_grad):
        if with_grad:
            v = self.gvf(state)
        else:
            with torch.no_grad():
                v = self.gvf(state)
        return v

    def get_gvf_target_value(self, state):
    	with torch.no_grad():
    		v = self.gvf_target(state)
        return v

    def update(self):
    	batch = self.get_data()
        state_batch, cumulant_batch, next_state_batch = batch['states'], batch['cumulants'], batch['next_states']

        gvf_current = self.get_gvf_value(state_batch, with_grad=True)

        gvf_target = cumulant_batch + (self.gamma * self.get_gvf_target_value(next_state_batch))

        gvf_loss = nn.functional.mse_loss(gvf_current, gvf_target)
        
        self.gvf_optimizer.zero_grad()
        gvf_loss.backward()
        self.gvf_optimizer.step()

        self.train_loss.append(torch_utils.to_np(gvf_loss))

    def sync_target(self):
        with torch.no_grad():
            for p, p_targ in zip(self.gvf.parameters(), self.gvf_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def train(self):
    	for epoch in range(len(self.epochs)):
    		self.update()
    		self.sync_target()

    	return self.train_loss

    def save(self):
        parameters_dir = self.parameters_dir
        
        path = os.path.join(parameters_dir, "gvf")
        torch.save(self.gvf.state_dict(), path)

        path = os.path.join(parameters_dir, "gvf_target")
        torch.save(self.gvf_target.state_dict(), path)
    
        path = os.path.join(parameters_dir, "gvf_opt")
        torch.save(self.gvf_optimizer.state_dict(), path)
        
        path = os.path.join(parameters_dir, "buffer.pkl")
        with open(path, "wb") as f:
            pkl.dump(self.buffer, f)

    def compute_returns(self, cumulants):
    	returns = []
    	G = 0.0
    	for t in range(len(cumulants) - 1, -1, -1):
    		G = cumulants[t] + (self.gamma * G)
    		returns.insert(0, G)

    	return np.array(returns)

    def prediction_errors(self, states, cumulants):
    	returns = self.compute_returns(cumulants)
    	predictions = self.get_gvf_value(states, with_grad=False)
    	predictions = torch_utils.to_np(predictions)
    	
    	return (np.square(returns - predictions)).mean(axis=0)

    def validation_loss(self):
    	return self.prediction_errors(self.val_states, self.val_cumulants)

    def test_loss(self):
    	return self.prediction_errors(self.test_states, self.test_cumulants)


