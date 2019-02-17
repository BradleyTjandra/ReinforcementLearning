# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 09:55:59 2019

@author: Bradley.Tjandra
"""

import gym
import tensorflow as tf
import numpy as np

total_episodes = 10000
max_steps = 999
update_frequency = 5


env = gym.make('CartPole-v0')
#s = env.reset()

params = {}
params["state_dims"] = env.observation_space.shape[0]
params["num_actions"] = env.action_space.n
params["layers"] = [8]
params["lr"] = 0.01
params["discount_rate"] = 0.99

def discount_rewards(params):
    
    y = params["discount_rate"]
    reward_history = tf.placeholder(tf.float32)
    
    # create a tensor which looks like [1, y, y**2, ...]
    discount_rates = tf.ones_like(reward_history) * y
    discount_rates = tf.cumprod(discount_rates, axis=1)
    
    discounted_rewards = reward_history * discount_rates
    disc_r_history = tf.cumsum(discounted_rewards, axis=1, reverse=True)
    disc_r_history /= discount_rates
    
    return reward_history, disc_r_history
    
def create_nn(params):
    
    state_dims = params["state_dims"]
    num_actions = params["num_actions"]
    layers = params["layers"]
    
    # policy network
    s = tf.placeholder(tf.float32, [None, state_dims], "place_state")
    for l in layers:
        layer = tf.layers.dense(
                s,
                units=l,
                activation=tf.nn.relu,
                use_bias=False)
    
    logits = tf.layers.dense(
            layer,
            units = num_actions,
            activation=None,
            use_bias=False)
    
    Q_estimate = tf.nn.softmax(logits)
    action = tf.multinomial(logits, 1)
    action = tf.reshape(action,[])
    
    placeholders = { "s" : s }   
    tensors = { 
            "action" : action,
            "Q_estimate" : Q_estimate
            }
    
    return placeholders, tensors
    
def create_trainer(params, tensors):
    
    Q_estimate = tensors["Q_estimate"]

    lr = params["lr"]
    num_actions = params["num_actions"]
    
    # loss for training
    chosen_action = tf.placeholder(tf.int32,[None], "place_chosen_action")
    reward = tf.placeholder(tf.float32, [None], "place_reward")    
    
    Q_mask = tf.one_hot(chosen_action, num_actions, 
                         on_value=True, off_value=False)
    responsible_outputs = tf.boolean_mask(Q_estimate, Q_mask)
    loss = - tf.log(responsible_outputs + 1E-4) * reward
    loss = tf.reduce_sum(loss)
        
    optimizer = tf.train.AdamOptimizer(learning_rate=lr) 
    grads = optimizer.compute_gradients(loss)
    train_op = optimizer.minimize(loss)
            
    placeholders = {
            "chosen_action" : chosen_action,
            "reward" : reward
            }
    tensors = {"train_op" : train_op, "grads" : grads}
    return placeholders, tensors
    
def create_graph(params):
    
    tf.reset_default_graph()
    
    placeholders, tensors = create_nn(params)
    train_holders, train_tensors = create_trainer(params, tensors)
    undisc_r, disc_r = discount_rewards(params)
    
    # Output
    placeholders.update(train_holders)
    placeholders["undisc_r"] = undisc_r
    tensors.update(train_tensors)
    tensors["disc_r"] = disc_r
    return placeholders, tensors

class EpHistory():
    
    def __init__(self):

        self.columns = ["s", "a", "r", "disc_r"]        
        self.reset_history()
        
    def append_step(self, step):
        s,a,r = step
        self.history["s"].append(s)
        self.history["a"].append(a)
        self.history["r"].append(r)
        
    def reset_history(self):
        self.history = { c : [] for c in self.columns }
    
    def extend_disc_r(self, disc_r):
        self.history["disc_r"].extend(disc_r)
        
    def pop_history(self):
        
        history = tuple(self.history[c] for c in self.columns)
        self.reset_history()
        return history
    
    def append_history(self, ep_history):
        
        for c in self.columns:
            dat = ep_history.history[c]
            self.history[c].extend(dat)        

placeholders, tensors = create_graph(params)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
training_buffer = EpHistory()
steps_history = []
reward_history = []
for i in range(total_episodes):
    s = env.reset()
    total_reward = 0
    ep_history = EpHistory()
    for steps in range(max_steps):
        a = sess.run(tensors["action"], { placeholders["s"] : [s] })
        new_state, r, is_done, _ = env.step(a)        
        ep_history.append_step((s,a,r))
        
        s = new_state
        total_reward += r
        
        if is_done:
            undisc_r = ep_history.history["r"]
            disc_r = sess.run(tensors["disc_r"], { placeholders["undisc_r"] : [undisc_r] })
            disc_r = disc_r.tolist()[0]
            ep_history.extend_disc_r(disc_r)
            training_buffer.append_history(ep_history)
            break
    
    steps_history.append(steps)
    reward_history.append(total_reward)
            
    if i % update_frequency == 0 and i != 0:
    
        s_hist, a_hist, _, disc_r_hist = training_buffer.pop_history()
        
        feed_dict={
                placeholders["s"] : s_hist,
                placeholders["chosen_action"] : a_hist,
                placeholders["reward"] : disc_r_hist
                }
        sess.run(tensors["train_op"], feed_dict)
        
    if i % 100 == 0:
        avg_reward = np.mean(reward_history[-500:])
        print("Ep #{}, Avg reward {}".format(str(i), str(avg_reward)))
    
sess.close()