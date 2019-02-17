# -*ses- coding: utf-8 -*-
"""
Created on Sun Feb 10 11:19:29 2019

@author: Bradley.Tjandra
"""

import numpy as np
import tensorflow as tf

bandits = [0.2, 0, -0.2, -5]

e = 0.1
total_episodes = 1000
params = {"num_bandits" : len(bandits) }

def pullBandit(bandit):
    
    z = np.random.randn(1)
    if z > bandit:
        return 1
    else:
        return -1
    
def create_graph(params):
    
    tf.reset_default_graph()
    
    num_bandits = params["num_bandits"]
    
    e = tf.placeholder(tf.float32, name="explore_prob")
    weights = tf.Variable(tf.ones([num_bandits]))
    action = tf.argmax(weights,0)
    
    explore_switch = tf.reshape(tf.random_uniform([1]) < e,[])
    random_action = tf.random_uniform(maxval=num_bandits-1, 
                                      dtype=tf.int64,
                                      shape=[1])
    
    action = tf.cond(explore_switch, 
                     lambda: tf.reshape(random_action,[]), 
                     lambda: action,
                     name="switch")
    
    chosen_action = tf.placeholder(tf.int32,name="chosen_action")
    reward = tf.placeholder(tf.float32,name="reward")
    loss = - tf.log(tf.slice(weights,[chosen_action],[1])) * reward
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
    train_op = optimizer.minimize(loss)
    
    placeholders = {
            "chosen_action" : chosen_action,
            "reward" : reward,
            "e" : e
            }
    
    tensors = {
            "weights" : weights,
            "action" : action,
            "random_action": random_action,
            "loss" : loss,
            "train_op" : train_op}
    
    return placeholders, tensors

placeholders, tensors = create_graph(params)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    reward_history = [0] * params["num_bandits"]
    for i in range(total_episodes):
        
        feed_dict = {placeholders["e"] : e}
        action = sess.run(tensors["action"], feed_dict)
        
        reward = pullBandit(bandits[action])
        
        feed_dict[placeholders["reward"]] = reward
        feed_dict[placeholders["chosen_action"]] = action
        sess.run(tensors["train_op"],feed_dict)
        
        reward_history[action] += reward
        
        if i % 50 == 0:
            print("Step {}: Just chose action {}. Bandits scores: {}".format(
                    str(i),
                    str(action),
                    str(reward_history)))
            
    weights = sess.run(tensors["weights"])
    best_bandit = str(np.argmax(weights) + 1) 
    print("Model thinks that {} is the best bandit".format(best_bandit))

        
        