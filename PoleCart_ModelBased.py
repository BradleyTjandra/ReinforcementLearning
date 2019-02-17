# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:52:49 2019

@author: Bradley.Tjandra
"""

import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import math

num_episodes = 10000
max_steps = 999
update_frequency = 5


env = gym.make('CartPole-v0')
env.reset()

params = {}
params["state_dims"] = env.observation_space.shape[0]
params["num_actions"] = env.action_space.n
params["policy_layers"] = [8]
params["model_layers"] = [256,256]
params["discount_rate"] = 0.99
params["policy_lr"] = 1E-02
params["model_lr"] = 1E-01

class PolicyNetwork:
    
    def discount_rewards(self, reward_history, y):
            
        # create a tensor which looks like [1, y, y**2, ...]
        discount_rates = tf.ones_like(reward_history) * y
        discount_rates = tf.cumprod(discount_rates)
        
        # creates the sum of future discounted rewards
        discounted_rewards = reward_history * discount_rates
        disc_r_history = tf.cumsum(discounted_rewards, reverse=True)
        disc_r_history /= discount_rates
        
        return disc_r_history

    def __init__(self, params):
    
        with tf.variable_scope("PolicyNetwork"):
            state_dims = params["state_dims"]
            hidden_layers = params["policy_layers"]
            num_actions = params["num_actions"]
            lr = params["policy_lr"]
            y = params["discount_rate"]
            
            # forward prop
            s = tf.placeholder(tf.float32, [None, state_dims])
            layer = s
            for lay in hidden_layers:
                layer = tf.layers.dense(
                        layer, 
                        lay,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        use_bias=False,
                        activation=tf.nn.relu)
            logits = tf.layers.dense(
                    layer,
                    num_actions, 
                    use_bias=False,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    activation=None)
            Q_pred = tf.nn.softmax(logits)
            action=tf.reshape(tf.multinomial(logits,1),[])
            
            # back prop
            reward = tf.placeholder(tf.float32, [None])
            chosen_action = tf.placeholder(tf.int32, [None])
            
            Q_mask = tf.one_hot(chosen_action, num_actions, 
                                 on_value=True, off_value=False)
            responsible_outputs = tf.boolean_mask(Q_pred, Q_mask)
            disc_r = self.discount_rewards(reward, y)
            loss = - tf.log(responsible_outputs + 1E-8) * disc_r
            loss = tf.reduce_sum(loss)

            self.tvars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope="PolicyNetwork")            
            optimizer = tf.train.AdamOptimizer(learning_rate=lr) 
            grads = optimizer.compute_gradients(loss, self.tvars)
            
            grad_holders = [tf.placeholder(tf.float32) for g in self.tvars]
            train_op = optimizer.apply_gradients(zip(grad_holders,self.tvars))
            
        # output
        self.placeholders = {
                "s" : s,
                "r" : reward,
                "a" : chosen_action,
                "grads" : grad_holders
                }
        
        self.tensors = {
                "Q" : Q_pred,
                "a" : action,
                "train_op" : train_op,
                "grads" : grads
                }
    
class ModelNetwork:
    
    def __init__(self, params):
        
        with tf.variable_scope("ModelNetwork"):
       
            state_dims = params["state_dims"]
            hidden_layers = params["model_layers"]
            lr = params["model_lr"]
            
            # forward prop
            s = tf.placeholder(tf.float32, [None,state_dims]) 
            a = tf.placeholder(tf.float32, [None]) 
            a_reshaped = tf.expand_dims(a, axis=-1)
            s_and_a = tf.concat([s,a_reshaped],axis=1)
            for lay in hidden_layers:
                layer = tf.layers.dense(
                        s_and_a,
                        lay,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        bias_initializer=tf.initializers.zeros,
                        activation=tf.nn.relu)
            
            pred_state = tf.layers.dense(
                    layer,
                    units = state_dims,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    bias_initializer=tf.initializers.zeros)
            pred_r = tf.layers.dense(
                    layer,
                    units = 1,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    bias_initializer=tf.initializers.zeros)
            done_logits = tf.layers.dense(
                    layer,
                    2,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    bias_initializer=tf.initializers.zeros)
            pred_state  = tf.clip_by_value(pred_state,-1E30, 1E30)
            pred_r      = tf.clip_by_value(pred_r,-1E30, 1E30)
            done_logits = tf.clip_by_value(done_logits,-1E30, 1E30)
            
            pred_done = tf.reshape(tf.multinomial(done_logits, 1),[-1,1])
            pred = (tf.reshape(pred_state,[-1, state_dims]), pred_r, pred_done)
            
            # back prop
            true_state = tf.placeholder(tf.float32,[None,state_dims])
            true_r = tf.placeholder(tf.float32,[None])
            true_done = tf.placeholder(tf.float32,[None])
            
            loss_state = tf.squared_difference(true_state, pred_state)
            loss_state = tf.reduce_sum(loss_state,axis=1)
            loss_state = tf.reduce_mean(loss_state)
            
            loss_r = tf.squared_difference(true_r,pred_r, "loss_r")
            loss_r = tf.reduce_mean(loss_r)
            
            pred_done = tf.cast(pred_done, tf.float32)
            true_done_recast = tf.cast(true_done, tf.int32)
            true_done_OH = tf.one_hot(true_done_recast, 2)
            loss_done = - tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=true_done_OH, 
                    logits=done_logits)
            loss_done  = tf.reduce_mean(loss_done)
            
            loss = loss_state + loss_r + loss_done
            optimizer = tf.train.AdamOptimizer(learning_rate = lr)
            train_op = optimizer.minimize(loss)
        
        # output
        self.placeholders = {
                "s" : s,
                "a" : a,
                "true_state" : true_state,
                "true_r" : true_r,
                "true_done" : true_done
                }
        
        self.tensors = {
                "train_op" : train_op,
                "pred" : pred,
                "loss" : loss,
                "pred_done" : pred_done,
                "loss_r" : loss_r,
                "loss_d" : loss_done,
                "done_logits" : done_logits
                }

        
class DataManager:
    
    def __init__(self):
        
        self.stats = {}
        
    def add_to_stat(self, stat_name, val):
        
        stat = self.stats.get(stat_name,[0,0])
        stat[0] += val
        stat[1] += 1
        self.stats[stat_name] = stat
        
        return stat
    
    def get_stat(self, stat_name):
        
        if type(stat_name) != "list":
            stat_name = [stat_name]    
            stats = [self.stats[n] for n in stat_name]
            stats_with_avg = \
                [[tot, num, tot / num] for tot, num in stats]
            return stats_with_avg
        
        else:
            stat = self.stats[stat_name]
            stat += stat[0] / stat[1]
            
            return stat
    
    
    def reset_stat(self, stat_name):
        
        if stat_name in self.stats[stat_name]:
            self.stats[stat_name] = [0,0]
        
        

tf.reset_default_graph()
policyNetwork = PolicyNetwork(params)
modelNetwork = ModelNetwork(params)
statManager = DataManager()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_model = True
train_policy = True
draw_from_model = False

grad_buffer = sess.run(policyNetwork.tvars)
grad_buffer = [g * 0 for g in grad_buffer]
grad_count = 0
policy_update_frequency = 5

pred_history = None
true_history = None

for ep in range(num_episodes):
    
    ep_history = {
            "s" : [],
            "a" : [],
            "r" : [],
            "d" : []
            }
    s = env.reset()
    
    is_done = False
    ep_reward = 0
    
    for step in range(max_steps):
        
        feed_dict = { policyNetwork.placeholders["s"] : [s] }
        a = sess.run(policyNetwork.tensors["a"], feed_dict)
        
        if draw_from_model:
            feed_dict = {
                    modelNetwork.placeholders["a"] : [a],
                    modelNetwork.placeholders["s"] : [s]
                    }
            next_state, r, is_done = sess.run(
                    modelNetwork.tensors["pred"],
                    feed_dict
                    )
            next_state = np.reshape(next_state,[params["state_dims"]])
            r = float(r)
            is_done = bool(is_done)
        
        else:
            next_state, r, is_done, _ = env.step(a)
            
        ep_reward += r
        d = 1 if is_done else 0
        ep_history["s"].append(s)
        ep_history["a"].append(a)
        ep_history["r"].append(r)
        ep_history["d"].append(d)
        
        s = next_state
        
        if is_done:
            break
        
        
    if is_done:
                
        if draw_from_model == False:    
            
            statManager.add_to_stat("world_r", ep_reward)
        
        if train_policy:
            feed_dict = {
                    policyNetwork.placeholders["s"] : ep_history["s"],
                    policyNetwork.placeholders["a"] : ep_history["a"],
                    policyNetwork.placeholders["r"] : ep_history["r"],
                    }
            grads = sess.run(
                    policyNetwork.tensors["grads"],
                    feed_dict
                    )
            
            for i, grad in enumerate(grads):
                grad_buffer[i] += grad[0]

            grad_count += 1
            if grad_count > policy_update_frequency:
                feed_dict = dict(zip(
                        policyNetwork.placeholders["grads"], 
                        grad_buffer))
                sess.run(
                        policyNetwork.tensors["train_op"],
                        feed_dict
                        )
                
                grad_count = 0
                grad_buffer = [ g*0 for g in grad_buffer ]
                    
        if train_model:
            
            orig_s = ep_history["s"][:-1]
            orig_a = ep_history["a"][:-1]
            true_s = ep_history["s"][1:]
            true_r = ep_history["r"][1:]
            true_d = ep_history["d"][1:]
            
            feed_dict = {
                    modelNetwork.placeholders["s"] : orig_s,
                    modelNetwork.placeholders["a"] : orig_a,
                    modelNetwork.placeholders["true_state"] : true_s,
                    modelNetwork.placeholders["true_r"] : true_r,
                    modelNetwork.placeholders["true_done"] : true_d
                    }
            tensors = tuple(modelNetwork.tensors[t] for t in ["train_op", "pred", "loss"])
            _, pred, loss = sess.run(tensors, feed_dict)    
            
            pred = np.hstack(pred)
            
            if pred_history is None:
                pred_history = pred
            else:
                pred_history = np.vstack([pred_history, pred])
            
            true_ep_history = np.hstack(
                    [np.vstack(true_s),
                    np.vstack(true_r),
                    np.vstack(true_d)]
                    )
            
            if true_history is None:
                true_history = true_ep_history
            else:
                true_history = np.vstack([true_history, true_ep_history])
        
    if (ep % 500) < 2:
    
        if draw_from_model:
            
            print("Episode {}: Model loss {}".format(ep, loss))
            
        else:
            
            world_r_stat = statManager.get_stat("world_r")
            _, _, avg_r = world_r_stat[0]
            print("Episode {}: Average reward {}".format(ep, avg_r))
            statManager.reset_stat("world_r")
        
    # alternate between training policy and model
    if ep > 100:
    
        draw_from_model = not draw_from_model
        train_model = not train_model
        train_policy = not train_policy
        
    if draw_from_model:
        s = np.random.uniform(-0.1,0.1,[4])
    else:
        s = env.reset()
        
plt.figure(figsize=(8, 12))
for i in range(6):
    plt.subplot(6, 2, 2*i + 1)
    plt.plot(true_history[:,i])
    plt.subplot(6,2,2*i+1)
    plt.plot(pred_history[:,i])
plt.tight_layout()        

sess.close()
