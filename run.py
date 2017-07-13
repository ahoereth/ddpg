from src import DDPG

ENV_NAME = 'LunarLanderContinuous-v2'
model = DDPG(ENV_NAME, memory=1e4, min_memory=1e3, update_frequency=1,
             state_stacksize=1, checkpoint=None, simulation_workers=2,
             train_workers=2, feed_workers=1)
model.train(10)
model.demo()
