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
REG = 0.0
INPUT_DIM = 4 #Input dimensions


#Initializing 
tf.reset_default_graph()

#none is a placeholder for the many observations that would be fed into the nn
input_x = tf.placeholder(tf.float32, [None,INPUT_DIM] , name="input_x")

# if you try to use initializer=tf.zeros_initializer() instead, you dont converge
W1 = tf.get_variable("W1", shape=[INPUT_DIM, H_SIZE],
           initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(input_x,W1))
W2 = tf.get_variable("W2", shape=[H_SIZE, 2],
           initializer=tf.contrib.layers.xavier_initializer())
logits = tf.matmul(layer1,W2)

#probability = tf.nn.sigmoid(score)

#From here we define the parts of the network needed for learning a good policy.
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32,[None,2], name="input_y")
advantages = tf.placeholder(tf.float32,[None,1], name="reward_signal")

# The loss function. This sends the weights in the direction of making actions 
# that gave good advantage (reward over time) more likely, and actions that didn't less likely.

probs = tf.nn.softmax(logits)
probs_log = tf.log(probs)
ce1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_y)
#ce2 = -tf.reduce_sum(input_y * probs_log, reduction_indices=[1])
# following lines do inner product
mul_by_adv = tf.multiply(advantages, tf.reshape(ce1, [-1, 1]))
loss_new = tf.reduce_sum(mul_by_adv) 

adam = tf.train.AdamOptimizer(learning_rate=ETA) # Adam optimizer
# next line returns for each layer the a (grads, vars) pair, but we dont want to use it as-is, we want to accumulate grad
newGrads = adam.compute_gradients(loss_new, var_list=tvars)
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
        tfprob = sess.run(probs,feed_dict={input_x: x})
        
        y = np.random.multinomial(1,tfprob[0])
        action = np.argmax(y)
        xs.append(x) #Store x
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
            print 'Episode %d: Average reward for episode %f.  Running average reward %f.' % (episode_number, reward_sum, running_reward)
                
            if running_reward > 195: 
                print "Task solved in",episode_number,'episodes'
                break
                    
            reward_sum = 0            
            input_initial = env.reset()

        if running_reward > 190 or rendering == True :     #Render environment only after avg reward reaches 100
        #    env.render()
            rendering = True
        
print episode_number,'Episodes completed.'
