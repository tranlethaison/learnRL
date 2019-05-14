"""Simple Policy Gradient
implement with tensorflow 2.0.0
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
print('Tensorflow ' + tf.__version__)
# tf.enable_eager_execution()

import gym
import numpy as np
import fire


def make_model(obser_dim, n_actions):
    inputs = layers.Input((obser_dim), name="Observation")
    x = layers.Flatten()(inputs)
    x = layers.Dense(32, activation=tf.tanh)(x)
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


def sample_action(model, obser):
    """sample action from policy"""
    logits = model(np.expand_dims(obser, axis=0).astype(np.float32))
    return tf.random.categorical(logits=logits, num_samples=1).numpy()[0][0]


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

    # trainning loop
    for epoch in range(n_epochs):
        # batch data
        obsers, actions = [], []
        weights = [] # weight for each lobprob(a|s) is R(tau)
        returns = [] # episode returns
        lens = [] # episode lens

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
                ep_return, ep_len = sum(ep_rewards), len(ep_rewards)

                # batch data
                weights += [ep_return] * ep_len
                returns.append(ep_return)
                lens.append(ep_len)

                if len(obsers) > batch_size:
                    break

                # reset episode specific variables
                obser, ep_rewards, done = env.reset(), [], False

        # optimize policy
        batch = [
            np.array(obsers, np.float32),
            np.array(actions, np.uint8),
            np.array(weights, np.float32)]

        loss = train_step(model, optimizer, batch, n_actions)

        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
            (epoch, loss, np.mean(returns), np.mean(lens)))


if __name__ == '__main__':
    fire.Fire()
