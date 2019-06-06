# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 12:16:20 2019

@author: Bradley.Tjandra
"""

import tensorflow as tf
import numpy as np
import sys
#import matplotlib.pyplot as plt

num_episodes = 10000
max_steps = 200
pre_train_steps = 10000

# where I have saved the GridWorld package
sys.path.append(r"C:\Users\bradley.tjandra\Dropbox\2019\Machine Learning_2019\Code\DeepRL_Medium_Tutorial") 
from GridWorldEnv import gameEnv
env = gameEnv(partial=False,size=5)
env.reset()


params = {}
params["state_dims"] = [84,84,3]
params["units"] = [32,64,64,512]
params["kernel_size"] = [8,4,3,7]
params["stride"] = [4,2,1,1]
params["A_layer"] = 512
params["V_layer"] = 512
params["num_actions"] = env.actions
params["lr"] = 1E-4
params["disc_rate"] = 0.99
params["tau"] = 0.001 # rate to update target network towards primary network

class DQN:
    
    def __init__(self, params, name):
        
        self.name = name
        self.params = params
        
        with tf.variable_scope(name):
            
            state_dims = params["state_dims"]
            units = params["units"]
            kernel_sizes = params["kernel_size"]
            strides = params["stride"]
            A_layer = params["A_layer"]
            V_layer = params["V_layer"]
            num_actions = params["num_actions"]
            lr = params["lr"]
            y = params["disc_rate"]
                
            s = tf.placeholder(tf.float32, shape=[None] + state_dims, name="s_holder")
            e = tf.placeholder(tf.float32, shape=[], name="eps_holder")
            
            layer = s 
            for num_units, kernel, stride in zip(units, kernel_sizes, strides):
                layer = tf.layers.conv2d(
                        layer, 
                        filters = num_units,
                        kernel_size=[kernel, kernel],
                        strides=[stride, stride]
                        )
            
            stream_A, stream_V = tf.split(layer, 2, len(state_dims))
            
            stream_A = tf.contrib.layers.flatten(stream_A)
            stream_A = tf.layers.dense(stream_A, 
                                       A_layer,
                                       activation=None)
            stream_A = tf.layers.dense(stream_A, num_actions)            
            stream_A = stream_A - tf.reduce_mean(stream_A, axis=-1, keepdims=True)
                
            stream_V = tf.contrib.layers.flatten(stream_V)
            stream_V = tf.layers.dense(stream_V,
                                       V_layer,
                                       activation=None)
            stream_V = tf.layers.dense(stream_V,1)
            
            Q_estimate = stream_V + stream_A
            best_action = tf.argmax(Q_estimate,1) # may not be right dims
            
            explore_switch = tf.random_uniform(tf.shape(best_action)) < e
            explore_switch = tf.cast(explore_switch, tf.int32)
            explore_switch = tf.one_hot(explore_switch, depth=2, dtype=tf.int64)
            random_action = tf.random_uniform(
                    maxval=num_actions, 
                    dtype=tf.int64,
                    shape=tf.shape(best_action))
            actions = tf.stack([best_action, random_action], axis = -1)
            chosen_action = tf.reduce_sum(explore_switch * actions, axis=-1)
            
            # back prop
            Q_next = tf.placeholder(tf.float32, shape=[None,num_actions], name="Q_holder")
            action_holder = tf.placeholder(tf.int32, shape=[None, 1], name="a_holder")
            reward_holder = tf.placeholder(tf.float32, shape=[None], name="r_holder")
            done_holder = tf.placeholder(tf.bool, shape=[None], name="done_holder")
            
            action_OH = tf.one_hot(action_holder, num_actions)
            not_done = 1 - tf.cast(done_holder, tf.float32)
            Q_target = reward_holder + not_done * tf.reduce_max(Q_next, -1) * y
            # "used" = we received info for the reward from action a
            Q_estimate_used = tf.reduce_sum(Q_estimate * action_OH ,-1)
            
            loss = tf.reduce_mean(tf.squared_difference(Q_estimate_used, Q_target))
            
            optimizer = tf.train.AdamOptimizer(learning_rate = lr)
            train_op = optimizer.minimize(loss)
            
            self.tvars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope=name)
            self.placeholders = {
                    "s" : s,
                    "e" : e,
                    "a" : action_holder,
                    "r" : reward_holder,
                    "d" : done_holder,
                    "Q_next" : Q_next}
            self.tensors = {
                    "action" : chosen_action,
                    "train_op" : train_op,
                    "Q_estimate" : Q_estimate
                    }
            
            
    def copy_network(self, params, DQN):

        # not really a copy, more of a "soft-update" 
        
        tau = params["tau"]
        g = tf.get_default_graph()
        
        source_tensor_names = [DQN.name + "/" + tv.name.replace(self.name + "/", "") 
            for tv in self.tvars]
#        new_vars = [g.get_tensor_by_name()]
        new_vars = [g.get_tensor_by_name(name) for name in source_tensor_names]
        assign_ops = [tf.assign(old_var, old_var * (1-tau) + new_var * tau) 
                        for old_var, new_var  in zip(self.tvars, new_vars)]
        return assign_ops
    
    def get_action(self, sess, state, e):
        
        feed_dict = {
                self.placeholders["s"] : [state],
                self.placeholders["e"] : e
                }
        tensors = self.tensors["action"]
        return sess.run(tensors, feed_dict)
    
    def get_Q(self, sess, state):
                
        feed_dict = {
                self.placeholders["s"] : state,
                self.placeholders["e"] : 0.
                }
        tensors = self.tensors["Q_estimate"]
        return sess.run(tensors, feed_dict)
    
    def update_network(self, sess, s, a, r, d, Q_next):
        
        feed_dict = {
                self.placeholders["s"] : s,
                self.placeholders["a"] : a,
                self.placeholders["r"] : r,
                self.placeholders["d"] : d,
                self.placeholders["Q_next"] : Q_next,
                }
        tensors = self.tensors["train_op"]
        sess.run(tensors, feed_dict)
        return None
        
class EpisodeHistory:
    
    def __init__(self, buffer_size = 10000):
    
        self.history_keys = ["s", "a", "r", "d", "s_next"]
        self.history = {key : [] for key in self.history_keys}
        self.length = 0
        self.buffer_size = buffer_size
    
    def append(self, s, a, r, d, s_next):
        
        if self.length >= self.buffer_size:
            start = self.length - self.buffer_size + 1
            end = self.length
            for k in self.history_keys:
                trunc_hist = self.history[k][start:end] 
                self.history[k] = trunc_hist
        else:
            self.length += 1
        
        for k, v in zip(self.history_keys, [s,a,r,d,s_next]):
            self.history[k].append(v)
            
        return(self.history)
        
    def sample(self, size):
        
        indices = np.random.randint(0, self.length, size)
        eps = tuple([self.history[k][i] for i in indices] for k in self.history_keys)
        return eps
    
    def clear(self):
        
        self.history = {key : [] for key in self.history_keys}
        self.length = 0
        
tf.reset_default_graph()
targetNetwork = DQN(params, "target")
actionNetwork = DQN(params, "action")
copyOps = targetNetwork.copy_network(params, actionNetwork)
epHistory = EpisodeHistory()
total_steps = 0

batch_size = 32
update_frequency = 4

# simulated annealing
initial_e = 1.
final_e = 0.1
annealing_steps = 10000.
change_e = (initial_e-final_e) / annealing_steps
e = initial_e

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(num_episodes):
    s = env.reset()
    episode_r = 0
    for episode_steps in range(1,max_steps+1):
        
        if total_steps < pre_train_steps:
            a = np.random.randint(params["num_actions"],size=1)
        else:
            a = actionNetwork.get_action(sess, s, e)
            
        s_next, r, d = env.step(a)
        epHistory.append(s,a,r,d,s_next)
        s = s_next
        
        episode_r += r
        total_steps += 1        
        
        if total_steps >= pre_train_steps:
            e = max(final_e, e - change_e)
            
            if total_steps % update_frequency == 0:
                ss, a_s, rs, ds, ss_next = epHistory.sample(batch_size)
                Q_next = targetNetwork.get_Q(sess, ss_next)
                actionNetwork.update_network(sess, ss, a_s, rs, ds, Q_next)
                sess.run(copyOps)
        
        # I don't think the game ever "finishes"
#        if d:
#            break
    
           
    if i % 500 == 0:
        print("Episode {}: Reward {}".format(i, episode_r))
        
        
sess.close()
