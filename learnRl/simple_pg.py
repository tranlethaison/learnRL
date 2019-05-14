"""Simple Policy Gradient
implement with tensorflow 2.0.0
"""
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
print('Tensorflow ' + tf.__version__)

import gym


def make_model(obser_dim, n_actions):
    inputs = layers.Input((obser_dim), name="Observation")
    x = layers.Flatten()(inputs)
    x = layers.Dense(32, activation=tf.tanh)(x)
    outputs = layers.Dense(n_actions, activation="linear")
    
    return keras.Model(inputs, outputs)


@tf.function
def optimize(model, optimizer, batch):
    """1 step optimize policy"""
    obsers, actions, weights = batch
    
    with tf.GradientTape() as tape:
        pass


def sample_action(model, obser):
    """sample action from policy"""
    logits = model(np.expand_dims(obser, axis=0))
    return tf.random.categorical(logits=logits, num_samples=1)[0][0]


def train(
    env_name="CartPole-v0",
    lr=1e-2,
    n_epochs=50,
    batch_size=5000,
    do_render=False
):
    # make environment, check spaces, get obser / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, gym.spaces.Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, gym.spaces.Discrete), \
        "This example only works for envs with discrete action spaces."

    obser_dim = env.observation_space.shape
    n_actions = env.action_space.n

    print('Env: {}\nobservation_dim: {}\nn_actions: {}'
        .format(env_name, obser_dim, n_actions))

    # model, optimizer
    model = make_model(obser_dim, n_actions)
    optimizer = optimizers.Adam(lr)

    # trainning policy
    def train_one_batch():
        # batch data
        obsers = []
        actions = []

        # reset episode specific variables
        obser, ep_rewards, done = env.reset(), [], False

        while 1:
            # sample action from policy
            action = sample_action(model, obser)

            # batch data
            obsers.append(obser)
            actions.append(action)

            # get environment feedback
            obser, reward, done, _ = env.step(action)
            ep_rewards.append(reward)

            if done:
                # weights for lobprob(a|s) is R(tau)
                weights = [sum(ep_rewards)] * len(ep_rewards)

                if len(obsers) > batch_size:
                    break

                # reset episode specific variables
                obser, ep_rewards, done = env.reset(), [], False

        # optimize policy
        batch = [obsers, actions, weights]
        loss = optimize(model, optimizer, batch)






