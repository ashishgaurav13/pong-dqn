import gym
import numpy as np
from utils import obs2img
from utils import downsample as ds
from utils import ExperienceBuffer
from torchsummary import summary
from torch import nn, optim
import torch
import glob, datetime, sys
from utils import Tee
import sys

# https://github.com/NoListen/ERL/blob/5102d3a8244ccdaec043a4ca4deec95c71a67c13/config.py
GAMMA = 0.99
EXPERIENCE_BUFFER_SIZE = 5000
EXPERIENCE_BUFFER_EXPLORE = 2000
MINIBATCH_SIZE = 8
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
STEP = 1
STEP_MOD = 1000
MAX_EP_STEPS = 1000
SAVE_MODELS = False
LOGGING = False
USE_GPU = True

# if LOGGING:
# 	log_file_name = datetime.datetime.now().strftime("log_%Y_%m_%d_%H_%M_%S.txt")
# 	log_file = open(log_file_name, "w")
# 	backup = sys.stdout
# 	sys.stdout = Tee(sys.stdout, log_file)

if USE_GPU:
	torch.backends.cudnn.benchmark = True
	torch.set_default_tensor_type(torch.cuda.FloatTensor)

class QNetwork(nn.Module):

	def __init__(self, env, lr, eps, decay):
		super().__init__()
		self.env = env
		self.conv = nn.Sequential(
			nn.Conv2d(1, 32, 32),
			nn.ReLU(),
			nn.Conv2d(32, 64, 32),
			nn.ReLU(),
			nn.Conv2d(64, 64, 16),
			nn.ReLU()
		)
		self.fc = nn.Sequential(
			nn.Linear(64*7*7, 512),
			nn.ReLU(),
			nn.Linear(512, 64),
			nn.ReLU(),
			nn.Linear(64, env.action_space.n),
		)
		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()
		self.eps = eps
		self.decay = decay

	def forward(self, x):
		global USE_GPU
		x = x.cuda() if USE_GPU else x
		x = self.conv(x)
		x = x.view(-1, 64*7*7)
		x = self.fc(x)
		return x.cuda() if USE_GPU else x

	def backprop(self, x, y):
		self.optimizer.zero_grad() # clear gradients
		output = self.forward(x)
		loss = self.loss(output, y)
		loss.backward()
		self.optimizer.step()
		self.eps *= self.decay
		return loss.item()

	def predict(self, x):
		if type(x) == list: x = np.array(x)
		x = torch.from_numpy(x).float()
		return self.forward(x)

	def choose_action(self, x):
		if np.random.rand() <= self.eps:
			return np.random.choice(self.env.action_space.n)
		else:
			[Q] = self.predict(np.resize(x, (1, *x.shape)))
			values, indices = torch.max(Q, 0)
			return indices.item()

	def train(self, batch):
		global USE_GPU
		i, o = [], []
		for e in batch:
			state, action, r, next_state, done = e
			[Q] = self.predict([state])
			[Q_ns] = self.predict([next_state])
			target = Q
			values, indices = torch.max(Q_ns, 0)
			target[action] = r+GAMMA*values.item() if not done else r
			i.append(state)
			o.append(target)
		ii = torch.FloatTensor(i).cuda() if USE_GPU else torch.FloatTensor(i)
		oo = torch.stack(o).cuda() if USE_GPU else torch.stack(o)
		return self.backprop(ii, oo)

def play_episode(env, eb, dqn, i):
	global EXPERIENCE_BUFFER_EXPLORE, MINIBATCH_SIZE
	global STEP, STEP_MOD, SAVE_MODELS, MAX_EP_STEPS
	obs = env.reset()
	obs = ds(obs)
	# obs2img(np.reshape(obs, (84, 84)), 'test.png')
	ep_reward = 0
	ep_steps = 0
	losses = []
	print("Ep:%g\tSt:%g\tR:%g" % (i, ep_steps, ep_reward), end='')

	while True:
		# env.render()
		# choose action, eps greedy and step
		action = dqn.choose_action(obs)
		next_obs, r, done, info = env.step(action)
		next_obs = ds(next_obs)
		# obs2img(np.reshape(next_obs, (84, 84)), 'test.png')

		# add experience tuple
		experience = (obs, action, r, next_obs, done)
		eb.add(experience)
		# train
		if len(eb) > EXPERIENCE_BUFFER_EXPLORE:
			batch = eb.sample(MINIBATCH_SIZE)
			loss = dqn.train(batch)
			losses += [loss]
		# save model
		# if SAVE_MODELS and STEP % STEP_MOD == 0:
		# 	print('Saving model: model%d.pt' % (STEP//STEP_MOD))
		# 	torch.save(dqn, 'model%d.pt' % (STEP//STEP_MOD))
		# Misc
		STEP += 1
		obs = next_obs
		ep_reward += r
		ep_steps += 1
		avg_loss = np.mean(losses) if len(losses) > 0 else np.nan
		print("\rEp:%g\tSt:%g\tR:%g\tAL:%g" % (i, ep_steps, ep_reward, avg_loss), end='')
		sys.stdout.flush()

		if done or ep_steps%MAX_EP_STEPS == 0:
			break

	avg_loss = np.mean(losses) if len(losses) > 0 else np.nan
	print("\rEp:%g\tSt:%g\tR:%g\tAL:%g" % (i, ep_steps, ep_reward, avg_loss))
	sys.stdout.flush()

# https://github.com/openai/atari-reset/blob/master/test_atari.py
env = gym.make('PongNoFrameskip-v0')
eb = ExperienceBuffer(size=EXPERIENCE_BUFFER_SIZE)

# Load latest model
# saved_models = glob.glob('model*.pt')
# if len(saved_models) > 0:
# 	nums = map(int, [item.lstrip('model').rstrip('.pt') for item in saved_models])
# 	max_num = max(nums)
# 	print('Loading: model%d.pt' % max_num)
# 	dqn = torch.load('model%d.pt' % max_num)
# 	STEP += max_num*STEP_MOD
# else:
# 	dqn = QNetwork(env, lr=0.001, eps=EPSILON_START, decay=EPSILON_DECAY)
dqn = QNetwork(env, lr=0.001, eps=EPSILON_START, decay=EPSILON_DECAY) #
dqn = dqn.cuda() if USE_GPU else dqn
summary(dqn, (1, 84, 84))
# exit(0)

# Infinite episodes
EP = 1
while True: 
	play_episode(env, eb, dqn, EP)
	EP += 1