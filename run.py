import tensorflow as tf

from src import DDPG


ENV_NAME = 'LunarLanderContinuous-v2'
CHECKPOINT = None

model = DDPG(
    ENV_NAME, checkpoint=CHECKPOINT,
    memory=1e6, min_memory=1e4, update_frequency=1,
    state_stacksize=1, simulation_workers=1, train_workers=2, feed_workers=10,
    batchsize=64,
    weight_decay=True,
    bias_decay=True,  # No defined in paper.
    decay_scale=1e-2,
    actor_batch_normalization=False,
    critic_batch_normalization=True,
    gamma=0.99,
    critic_learning_rate=1e-3,
    actor_learning_rate=1e-4,
    tau=1e-3
)
model.train(1000000)
# model.demo()
