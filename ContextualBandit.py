# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 12:30:10 2019

@author: Bradley.Tjandra
"""

import numpy as np
import tensorflow as tf
e = 0.1
total_episodes = 10000
params = {"lr" : 0.01}

class ContextualBandit():
    
    def __init__(self):
        
        self.state=0
        self.bandits=np.array(
                [[0.2,0,0.0,-5],
                 [0.2,-5,1,0.25],
                 [-5,5,5,5]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]
        
    def getBandit(self):
        
        return self.state
    
    def pullBandit(self, action):
        
        bandit = float(self.bandits[self.state, action])
        if float(np.random.randn(1)) > bandit:
            reward =  1
        else:
            reward = -1
        
        self.state = np.random.randint(low=0, high=self.num_bandits)
        
        return (self.state, reward)
    
    def sampleActionSpace(self):
            
        return (np.random.randint(0, self.num_actions, 1)[0])

    
def create_graph(params):
    
    tf.reset_default_graph()
    
    num_bandits = params["num_bandits"]
    num_actions = params["num_actions"]    
    lr          = params["lr"]
    
    e = tf.placeholder(tf.float32, name="explore_prob")
    s = tf.placeholder(tf.int32)
    s_OH = tf.one_hot(s, depth=num_bandits)
    s_OH = tf.reshape(s_OH,[-1,num_bandits])
    Q_estimate =  \
        tf.layers.dense(s_OH,
                        units=num_actions,
                        activation=tf.sigmoid,
                        kernel_initializer=tf.ones_initializer(),
                        use_bias=False)
    best_action = tf.argmax(Q_estimate,1)
    
    explore_switch = tf.random_uniform([1]) < e
    random_action = tf.random_uniform(
            maxval=num_actions, 
            dtype=tf.int64,
            shape=[1])

    
    action = tf.cond(tf.reshape(explore_switch,[]), 
                     lambda: random_action, 
                     lambda: best_action)
    action = tf.reshape(action, [])
#    action = tf.reshape(best_action, [])
    
    # loss for training
    chosen_action = tf.placeholder(tf.int32,name="chosen_action")
    reward = tf.placeholder(tf.float32,name="reward")
    loss = - tf.log(tf.slice(tf.reduce_mean(Q_estimate,axis=0),begin=[chosen_action],size=[1])) * reward
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)
    train_op = optimizer.minimize(loss)
    
    # return arguments
    placeholders = {
            "chosen_action" : chosen_action,
            "reward" : reward,
            "e" : e,
            "state" : s
            }
    
    tensors = {
            "Q_estimate" : Q_estimate,
            "action" : action,
#            "random_action": random_action,
            "best_action": best_action,
            "loss" : loss,
            "train_op" : train_op}
    
    return placeholders, tensors
        
env = ContextualBandit()
params["num_bandits"] = env.num_bandits
params["num_actions"] = env.num_actions

placeholders, tensors = create_graph(params)
reward_history = [[]] * params["num_bandits"]
#action_tensors = [tensors[t] for t in ["action", "random_action","best_action"]]

sess = tf.Session(); sess.run(tf.global_variables_initializer());
state = env.getBandit()
for i in range(total_episodes):
    
    feed_dict = {
            placeholders["e"] : e,
            placeholders["state"] : state}
    action = sess.run(tensors["action"], feed_dict)
#    action, raction, baction = sess.run(action_tensors, feed_dict)
    
#    if np.random.rand(1) < e:
#        action = env.sampleActionSpace()
    
    new_state, reward = env.pullBandit(action)
    reward_for_this_bandit = reward_history[state][:]
    reward_for_this_bandit.append(reward)
    reward_history[state] = reward_for_this_bandit
    
    feed_dict[placeholders["reward"]] = reward
    feed_dict[placeholders["chosen_action"]] = action
    sess.run(tensors["train_op"], feed_dict)
    
    state = new_state
        
    if i % 500 == 0:
        print("Step {}: Just chose action {} for bandit {}. Average bandit scores: {}".format(
                str(i),
                str(action+1),
                str(state+1),
                str([np.mean(rs) for rs in reward_history])))
            
for b in range(params["num_bandits"]):
    best_bandit = sess.run(tensors["best_action"], 
                           feed_dict={placeholders["state"] : b})
    print("For Bandit {}, model thinks that {} is the best action."
          .format(b+1, best_bandit[0]+1))
    

sess.close()    