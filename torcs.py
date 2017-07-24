import time

import tensorflow as tf

from src import DDPG


ENV_NAME = 'Torcs'
CHECKPOINT = None
# CHECKPOINT = 'logs/170714-1406-107000'
# CHECKPOINT = tf.train.latest_checkpoint('logs')  # not working


start = time.time()

model = DDPG(
    ENV_NAME, checkpoint=CHECKPOINT,
    memory=1e6, min_memory=33,
    update_frequency=1, state_stacksize=1,
    simulation_workers=1, train_workers=1, feed_workers=1,
    batchsize=32,
    weight_decay=True,
    bias_decay=True,
    decay_scale=1e-2,
    actor_batch_normalization=False,
    critic_batch_normalization=False,
    gamma=0.99,
    critic_learning_rate=1e-3,
    actor_learning_rate=1e-4,
    tau=1e-3,
    mu=[0., .1],  # steering, acceleration
    theta=[.6, 1.],  # steering, acceleration
    sigma=[.3, .1],  # steering, acceleration
    exploration_steps=10000,
    config_name=''
)
model.train(1000000)
# model.demo()

print(time.time() - start)
