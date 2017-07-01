import numpy as np
import cPickle as pickle
import tensorflow as tf

import matplotlib.pyplot as plt
import math

import gym
env = gym.make('CartPole-v0')

env.reset()

#Hyperparameters
H_SIZE = 10 #Number of hidden layer neurons
ETA = 1e-2 #Learning Rate
GAMMA = 0.99 #Discount factor

INPUT_DIM = 4 #Input dimensions


#Initializing 
tf.reset_default_graph()

#none is a placeholder for the many observations that would be fed into the nn
input_x = tf.placeholder(tf.float32, [None,INPUT_DIM] , name="input_x")

# if you try to use initializer=tf.zeros_initializer() instead, you dont converge
W1 = tf.get_variable("W1", shape=[INPUT_DIM, H_SIZE],
           initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(input_x,W1))
W2 = tf.get_variable("W2", shape=[H_SIZE, 1],
           initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1,W2)

#TODO: we want to generalize to many outputs, not just a single neuron
#action_probs = sess.run(y,feed_dict={observation: obsrv})
#action = np.argmax(np.multinomial(1,action_probs))
probability = tf.nn.sigmoid(score)

#From here we define the parts of the network needed for learning a good policy.
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
advantages = tf.placeholder(tf.float32,name="reward_signal")

# The loss function. This sends the weights in the direction of making actions 
# that gave good advantage (reward over time) more likely, and actions that didn't less likely.
loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(probability))
loss = -tf.reduce_mean(loglik * advantages) 

adam = tf.train.AdamOptimizer(learning_rate=ETA) # Adam optimizer
# next line returns for each layer the a (grads, vars) pair, but we dont want to use it as-is, we want to accumulate grad
newGrads = adam.compute_gradients(loss, var_list=tvars)
updateGrads = adam.apply_gradients(newGrads)

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add
    return discounted_r

xs,drs,ys = [],[],[]	#Arrays to store parameters till an update happens
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
init = tf.initialize_all_variables()

# Training
with tf.Session() as sess:
    rendering = False
    sess.run(init)
    input_initial = env.reset() # Initial state of the environment
    
    while episode_number <= total_episodes:
                    
        # Format the state for placeholder
        x = np.reshape(input_initial,[1,INPUT_DIM])
        
        # Run policy network 
        tfprob = sess.run(probability,feed_dict={input_x: x})
        action = 1 if np.random.uniform() < tfprob else 0
        
        xs.append(x) #Store x
        y = 1 if action == 0 else 0
        ys.append(y)

        # take action for the state
        input_initial, reward, done, info = env.step(action)
        reward_sum += reward

        drs.append(reward) # store reward after action is taken

        if done: 
            episode_number += 1
            # Stack the memory arrays to feed in session
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            
            xs,drs,ys = [],[],[] #Reset Arrays

            # Compute the discounted reward
            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            
            sess.run(updateGrads,feed_dict={input_x: epx, input_y: epy, advantages: discounted_epr})
            # Print details of the present model
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print 'Episode %d: Average reward for episode in this batch %f.  Running average reward %f.' % (episode_number, reward_sum/batch_size, running_reward/batch_size)
                
            if running_reward/batch_size > 195: 
                print "Task solved in",episode_number,'episodes'
                break
                    
            reward_sum = 0            
            input_initial = env.reset()

        if running_reward > 190 or rendering == True :     #Render environment only after avg reward reaches 100
        #    env.render()
            rendering = True
        
print episode_number,'Episodes completed.'
