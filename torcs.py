import tensorflow as tf

from src import DDPG

ENV_NAME = 'Torcs'
CHECKPOINT = None
CHECKPOINT = 'runs/170725-1720/291000'

model = DDPG(
    ENV_NAME,
    checkpoint=CHECKPOINT,
    memory=1e7,
    min_memory=100,
    update_frequency=1,
    state_stacksize=1,
    simulation_workers=1,
    train_workers=1,
    feed_workers=1,
    batchsize=32,
    weight_decay=True,
    bias_decay=False,
    decay_scale=1e-3,
    actor_batch_normalization=True,
    critic_batch_normalization=True,
    gamma=0.99,
    h1=300,
    h2=600,
    critic_learning_rate=1e-3,
    actor_learning_rate=1e-4,
    tau=5e-3,  # 1e-3
    mu=[0., .5],  # steering, acceleration
    theta=[.8, 1.],  # steering, acceleration
    sigma=[.3, .1],  # steering, acceleration
    exploration_steps=10000,
    config_name='')

model.train(1)  # steps to train for
model.demo()
