# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:27:45 2019

@author: Bradley.Tjandra
"""

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

max_steps = 99
num_episodes = 2000
y = 0.99 # disc factor
e = 0.1 # explore

# set up environment
#env = gym.make('FrozenLake-v0', is_slippery=True)
env = gym.make('FrozenLake-v0')
n_states = env.observation_space.n
n_actions = env.action_space.n

def create_graph():
  
  tf.reset_default_graph()
  
  s = tf.placeholder(tf.int32)
  Q_actual = tf.placeholder(tf.float32, shape=(1,n_actions))
  
  s_onehot = tf.expand_dims(tf.one_hot(s, (n_states), dtype=tf.float32),axis=0)
  s_onehot = tf.reshape(s_onehot,[-1,n_states])
    
  Q_estimate = tf.layers.dense(inputs=s_onehot, 
                               units=n_actions, 
                               kernel_initializer=tf.initializers.random_uniform(0,0.01),
                               use_bias=False)
  next_action = tf.argmax(Q_estimate,1)
  loss = tf.reduce_sum(tf.squared_difference(Q_estimate, Q_actual))
#  optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
  optimizer = tf.train.AdamOptimizer(learning_rate = 0.1)
  train_op = optimizer.minimize(loss)

  
  placeholders = {
      "s" : s,
      "Q_actual" : Q_actual
  }
  
  tensors = {
      "Q_estimate" : Q_estimate,
      "next_action" : next_action,
      "train_op" : train_op,
      "s_onehot" : s_onehot
  }
  
  return placeholders, tensors

placeholders, tensors = create_graph()
tensors_no_train = tuple(tensors[t] for t in ["next_action", "Q_estimate", "s_onehot"])

step_count_history = []
r_history = []

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(num_episodes):
    s = env.reset()
    step_count = 0 # steps in episode
    total_r = 0
    is_done = False
    while step_count < max_steps:
      step_count += 1
      feed_dict = {placeholders["s"] : s}
      next_action, q_estimate, s_onehotval = sess.run(tensors_no_train, feed_dict=feed_dict)
      next_action = next_action[0]
      if np.random.rand(1) < e:
        next_action = env.action_space.sample()
      
      s_next, r, is_done, _ = env.step(next_action)
    
      q_future = sess.run(tensors["Q_estimate"], feed_dict={placeholders["s"]:s_next})  
      
      # if action was not taken, assume actual q was what we estimated
      q_actual = q_estimate
      q_actual[0, next_action] = r + y * np.max(q_future)
      
      feed_dict[placeholders["Q_actual"]] = q_actual
      sess.run(tensors["train_op"], feed_dict=feed_dict)
      s = s_next
      total_r += r
      if is_done == True:
        e = 1./((i/50)+10)
        break
        
    step_count_history.append(step_count)
    r_history.append(total_r)
    
    if i % 100 == 0:
      print("Percent of succesful episodes (after " +str(i)+" episodes): " + str(sum(r_history)/len(r_history) * 100) + "%")
    
print("Percent of succesful episodes: " + str(sum(r_history)/num_episodes * 100) + "%")
      
plt.plot(r_history)

