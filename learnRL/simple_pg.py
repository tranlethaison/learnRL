"""Simple Policy Gradient
implement with tensorflow 2.0.0
"""
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
print('Tensorflow ' + tf.__version__)

import gym
import numpy as np
import fire


def make_model(obser_dim, n_actions):
    # activation = lambda x: tf.nn.leaky_relu(x, alpha=0.1)
    activation = tf.nn.relu6

    inputs = layers.Input((obser_dim), name="Observation")
    x = layers.Flatten()(inputs)
    x = layers.Dense(32, activation=activation)(x)
    outputs = layers.Dense(n_actions, activation="linear")(x)

    return keras.Model(inputs, outputs)


@tf.function
def train_step(model, optimizer, batch, n_actions):
    """1 step optimize policy"""
    obsers, actions, weights = batch
    actions_mask = tf.one_hot(actions, n_actions)

    with tf.GradientTape() as tape:
        # log(a|s) R(tau)
        logits = model(obsers)

        log_probs = tf.reduce_sum(
            actions_mask * tf.nn.log_softmax(logits),
            axis=1)

        loss = -tf.reduce_mean(log_probs * weights)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


@tf.function
def sample_action(model, obser):
    """sample action from policy"""
    obser = tf.expand_dims(tf.cast(obser, tf.float32), axis=0)
    logits = model(obser)
    return tf.random.categorical(logits=logits, num_samples=1)[0][0]


def train(
    env_name="CartPole-v0",
    lr=25e-3,
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

    # trainning loop
    for epoch in range(n_epochs):
        # batch data
        obsers, actions = [], []
        weights = [] # weight for each lobprob(a|s) is R(tau)
        returns = [] # episode returns
        lens = [] # episode lens

        # reset episode specific variables
        obser, ep_rewards, done = env.reset(), [], False

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        while 1:
            # sample action from policy
            action = sample_action(model, obser).numpy()

            # batch data
            obsers.append(obser)
            actions.append(action)

            # get environment feedback
            obser, reward, done, _ = env.step(action)
            ep_rewards.append(reward)

            # rendering
            if do_render \
            and not finished_rendering_this_epoch:
                env.render()

            if done:
                ep_return, ep_len = sum(ep_rewards), len(ep_rewards)

                # batch data
                weights += [ep_return] * ep_len
                returns.append(ep_return)
                lens.append(ep_len)

                if len(obsers) > batch_size:
                    break

                # reset episode specific variables
                obser, ep_rewards, done = env.reset(), [], False

                finished_rendering_this_epoch = True

        # optimize policy
        batch = [
            tf.cast(obsers, tf.float32),
            tf.cast(actions, tf.uint8),
            tf.cast(weights, tf.float32)]

        loss = train_step(model, optimizer, batch, n_actions)

        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
            (epoch, loss, np.mean(returns), np.mean(lens)))


if __name__ == '__main__':
    # with small network, running on CPU is faster.
    #os.environ['CUDA_VISIBLE_DEVICES'] = ''

    fire.Fire()

