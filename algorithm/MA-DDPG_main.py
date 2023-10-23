"""
Deep Deterministic Policy Gradient (DDPG)
-----------------------------------------
An algorithm concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function,
and uses the Q-function to learn the policy.
Reference
---------
Deterministic Policy Gradient Algorithms, Silver et al. 2014
Continuous Control With Deep Reinforcement Learning, Lillicrap et al. 2016
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials/
Environment
-----------
Openai Gym Pendulum-v0, continual action space
Prerequisites
-------------
tensorflow >=2.0.0a0
tensorflow-proactionsbility 0.6.0
tensorlayer >=2.0.0
To run
------
python tutorial_DDPG.py --train/test
"""

import argparse
import os
import random
import time
import math

#import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.io import loadmat

import tensorlayer as tl

# add arguments in command  --train/test
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_ID = 'LargeGridWorld-v0'  # environment id
RANDOM_SEED = 666  # random seed, can be either an int number or None
RENDER = False  # render while training

ALG_NAME = 'DDPG'
TRAIN_EPISODES = 500  # total number of episodes for training
TEST_EPISODES = 10  # total number of episodes for training
MAX_STEPS = 1000  # 20000total number of steps for each episode

LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacemen 
MEMORY_CAPACITY = 20000  # 500000size of replay buffer
BATCH_SIZE = 64  # update action batch size
VAR = 5  # control exploration
#var_real=VAR
###############################  DDPG  ####################################
n_width=93
n_height = 93
m = loadmat("C://Users/Administrator.DESKTOP-NLH290A/Desktop/2302_code/FIGURE_3/mapdata_0717.mat") 
#correct_action=0
MARK= m["MARK_new"]
PL_AP=m["MARK_PL_real"]


class ReplayBuffer:
    """
    a ring buffer for storing transitions and sampling for training
    :state: (state_dim,)
    :action: (action_dim,)
    :reward: (,), scalar
    :next_state: (state_dim,)
    :done: (,), scalar (0 and 1) or bool (True and False)
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

def cosVector(x,y):
    result1=0.0;
    result2=0.0;
    result3=0.0;
    for i in range(len(x)):
        result1+=x[i]*y[i]   #sum(X*Y)
        result2+=x[i]**2     #sum(X*X)
        result3+=y[i]**2     #sum(Y*Y)
    return result1/((result2*result3)**0.5)

class DDPG(object):
    """
    DDPG class
    """
    def __init__(self, action_dim, state_dim, action_range, replay_buffer, agent_num=0):
        self.replay_buffer = replay_buffer
        self.action_dim, self.state_dim, self.action_range = action_dim, state_dim, action_range
        self.var = VAR
        self.agent_num=agent_num

        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        def get_actor(input_state_shape, name=str(self.agent_num)):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: act
            """
            input_layer = tl.layers.Input(input_state_shape, name='A_input'+str(self.agent_num))
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name=str(self.agent_num)+'A_l1')(input_layer)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name=str(self.agent_num)+'A_l2')(layer)
            layer = tl.layers.Dense(n_units=action_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name=str(self.agent_num)+'A_a')(layer)
            layer = tl.layers.Lambda(lambda x: action_range * x)(layer)
            return tl.models.Model(inputs=input_layer, outputs=layer, name='Actor' + name)

        def get_critic(input_state_shape, input_action_shape, name=str(self.agent_num)):
            """
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :param name: name
            :return: Q value Q(s,a)
            """
            state_input = tl.layers.Input(input_state_shape, name=str(self.agent_num)+'C_s_input')
            action_input = tl.layers.Input(input_action_shape, name=str(self.agent_num)+'C_a_input')
            layer = tl.layers.Concat(1)([state_input, action_input])
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name=str(self.agent_num)+'C_l1')(layer)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name=str(self.agent_num)+'C_l2')(layer)
            layer = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name=str(self.agent_num)+'C_out')(layer)
            return tl.models.Model(inputs=[state_input, action_input], outputs=layer, name='Critic' + name)

        self.actor = get_actor([None, state_dim])
        self.critic = get_critic([None, state_dim], [None, action_dim])
        self.actor.train()
        self.critic.train()

        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        self.actor_target = get_actor([None, state_dim], name=str(self.agent_num)+'_target')
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        self.critic_target = get_critic([None, state_dim], [None, action_dim], name=str(self.agent_num)+'_target')
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)

    def ema_update(self):
        """
        Soft updating by exponential smoothing
        :return: None
        """
        paras = self.actor.trainable_weights + self.critic.trainable_weights
        self.ema.apply(paras)
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))

    def get_action(self, state, greedy=False):
        """
        Choose action
        :param s: state
        :param greedy: get action greedy or not
        :return: act
        """
        action = self.actor(np.array([state]))[0]
        if greedy:
            return action
            #return np.random.rand(len(action)).astype(np.float32)- action_range
        return np.clip(
            np.random.normal(action, self.var), -self.action_range, self.action_range
        ).astype(np.float32)  # add randomness to action selection for exploration

    def learn(self, exact_var):
        """，
        Update parameters
        :return: None
        """
        self.var = exact_var
        #print(self.var)
        states, actions, rewards, states_, done = self.replay_buffer.sample(BATCH_SIZE)
        rewards = rewards[:, np.newaxis]
        done = done[:, np.newaxis]

        with tf.GradientTape() as tape:
            actions_ = self.actor_target(states_)
            q_ = self.critic_target([states_, actions_])
            target = rewards + (1 - done) * GAMMA * q_
            q_pred = self.critic([states, actions])
            td_error = tf.losses.mean_squared_error(target, q_pred)
        critic_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_weights))

        with tf.GradientTape() as tape:
            actions = self.actor(states)
            q = self.critic([states, actions])
            actor_loss = -tf.reduce_mean(q)  # maximize the q
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_weights))
        self.ema_update()


    def save(self):
        """
        save trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor_target.hdf5'), self.actor_target)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'critic.hdf5'), self.critic)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'critic_target.hdf5'), self.critic_target)

    def load(self):
        """
        load trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor_target.hdf5'), self.actor_target)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic.hdf5'), self.critic)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic_target.hdf5'), self.critic_target)


if __name__ == '__main__':
    n_mu=3
    n_M=5
    n_o=6*7
    
    #env = gym.make(ENV_ID).unwrapped
    #env = gym.make(ENV_ID).unwrapped

    # reproducible
    # env.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    state_dim = 2
    action_dim = 10
    action_range = 1  # scale action, [-action_range, action_range]
    action_range_su = np.array([1, 1, 1], dtype=np.float32)
    action_range_ris = np.array([1]*(2*n_M+1), dtype=np.float32)
    action_range_n_o = np.array([1]*(2+1), dtype=np.float32)
    
    buffer = ReplayBuffer(MEMORY_CAPACITY) #MU1
    buffer2 = ReplayBuffer(MEMORY_CAPACITY)  #MU2
    buffer3 = ReplayBuffer(MEMORY_CAPACITY)  #MU3
    # buffer4 = ReplayBuffer(MEMORY_CAPACITY)  #su1
    # buffer5 = ReplayBuffer(MEMORY_CAPACITY)  #su2
    # buffer6 = ReplayBuffer(MEMORY_CAPACITY)  #ris
    # buffer7 = ReplayBuffer(MEMORY_CAPACITY)  #commu
    
    
    agent = DDPG(action_dim, state_dim, action_range, buffer, 1) #mu
    agent2 = DDPG(action_dim, state_dim, action_range, buffer2, 2)
    agent3 = DDPG(action_dim, state_dim, action_range, buffer3, 3)
    # agent4 = DDPG(action_dim-2, state_dim, action_range_su, buffer4, 4) #su
    # agent5 = DDPG(action_dim-2, state_dim, action_range_su, buffer5, 5)
    # agent6 = DDPG(2*n_M+1, state_dim*n_mu, action_range_ris, buffer6, 6) #ris
    # agent7 = DDPG(2+1, state_dim*n_mu, action_range_n_o, buffer7, 7) #commu n_o
    
    VAR1=VAR
    VAR2=VAR
    VAR3=VAR
    t0 = time.time()
    if args.train:  # train
        all_episode_reward = []
        all_episode_reward2 = []
        all_episode_reward3 = []
        # all_episode_reward4 = []
        # all_episode_reward5 = []
        # all_episode_reward6 = []
        # all_episode_reward7 = []
        init=0
        
        # end_location = [38*2,11*2]
        # end_location2 = [26*2,18*2]  #[8*2,9*2]
        # end_location3 = [16*2,32*2] #[35*2,9*2]
        end_location = [15*2,32*2]
        end_location2 = [45*2,45*2]
        end_location3 = [47,38*2]
        #end_location = end_location3
        #end_location2 = end_location3
        study=0
        study2=0
        study3=0
        for episode in range(TRAIN_EPISODES):
            #state initialize
            x_k1_array = []
            y_k1_array = []
            x_k2_array = []
            y_k2_array = []
            x_k3_array = []
            y_k3_array = []
            state = np.array([4*2,5*2], dtype=np.float32)  # 环境重置
            state2 = np.array([20*2,20*2], dtype=np.float32)
            state3 = np.array([40*2,10*2], dtype=np.float32)
            state_su1 = np.array([17, 25*2], dtype=np.float32)
            state_su2 = np.array([50, 25*2], dtype=np.float32)
            state_su3 = np.array([84, 25*2], dtype=np.float32)
            episode_reward = 0
            episode_reward2 = 0
            episode_reward3 = 0
            # episode_reward4 = 0
            # episode_reward5 = 0
            # episode_reward6 = 0
            # episode_reward7 = 0
            done1=False
            done2=False
            done3=False
            bobao=0
            bobao2=0
            bobao3=0
            x_k1_array.append(state[0])
            y_k1_array.append(state[1])
            x_k2_array.append(state2[0])
            y_k2_array.append(state2[1])
            x_k3_array.append(state3[0])
            y_k3_array.append(state3[1])
            #greedy0=True
            for steps in range(MAX_STEPS):
                # if RENDER:
                #     env.render()
                # Add exploration noise
                # action selection
                #if len(buffer) >= MEMORY_CAPACITY:
                #    greedy0=False
                action = agent.get_action(state)
                action2 = agent2.get_action(state2)
                action3 = agent3.get_action(state3)
                
                
                # action4 = agent4.get_action(state4)
                # action5 = agent5.get_action(state5)
                # action6 = agent6.get_action(state6)
                # action7 = agent7.get_action(state7)
                # Step
                if not done1:
                    [old_x, old_y] = state
                    new_x, new_y = int(old_x), int(old_y)
                    new_x=int(old_x+action[0])
                    new_y=int(old_y+action[1])
                    if int(new_x) <= 0: 
                        new_x = 1
                    if int(new_x) >= n_width: 
                        new_x = int(n_width)-1
                    if int(new_y) <= 0: 
                        new_y = 1
                    if int(new_y) >= n_height: 
                        new_y = int(n_height)-1
                    if MARK[new_x,new_y] == 2:
                        new_x, new_y = old_x, old_y
                    state_=np.array([new_x, new_y], dtype=np.float32)
                    x_k1_array.append(state_[0])
                    y_k1_array.append(state_[1])
                else:
                    state_ = state
                if not done2:
                    [old_x, old_y] = state2
                    new_x, new_y = int(old_x), int(old_y)
                    new_x=int(old_x+action2[0])
                    new_y=int(old_y+action2[1])
                    if int(new_x) <= 0: 
                        new_x = 1 
                    if int(new_x) >= n_width: 
                        new_x = int(n_width)-1
                    if int(new_y) <= 0: 
                        new_y = 1
                    if int(new_y) >= n_height: 
                        new_y = int(n_height)-1
                    if MARK[new_x,new_y] == 2:
                        new_x, new_y = old_x, old_y
                    state2_=np.array([new_x, new_y], dtype=np.float32)
                    x_k2_array.append(state2_[0])
                    y_k2_array.append(state2_[1])
                else:
                    state2_ = state2
                if not done3:
                    [old_x, old_y] = state3
                    new_x, new_y = int(old_x), int(old_y)
                    new_x=int(old_x+action3[0])
                    new_y=int(old_y+action3[1])
                    if int(new_x) <= 0: 
                        new_x = 1 
                    if int(new_x) >= n_width: 
                        new_x = int(n_width)-1
                    if int(new_y) <= 0: 
                        new_y = 1
                    if int(new_y) >= n_height: 
                        new_y = int(n_height)-1
                    if MARK[new_x,new_y] == 2:
                        new_x, new_y = old_x, old_y
                    state3_=np.array([new_x, new_y], dtype=np.float32)
                    x_k3_array.append(state3_[0])
                    y_k3_array.append(state3_[1])
                else:
                    state3_ = state3
                # state4+5 static
                
                # state6_ = np.array([state_[0], state_[1], state2_[0], state2_[1], state3_[0], state3_[1]])
                # state7_ = np.array([state_[0], state_[1], state2_[0], state2_[1], state3_[0], state3_[1]])
                done_sys = done1 and done2 and done3
                
                if action[8]==-1:
                    action[8]=-0.9999999
                if action2[8]==-1:
                    action2[8]=-0.9999999
                if action3[8]==-1:
                    action3[8]=-0.9999999
                if action[8]==1:
                    action[8]=0.9999999
                if action2[8]==1:
                    action2[8]=0.9999999
                if action3[8]==1:
                    action3[8]=0.9999999
                
                w_1=np.array([action[2]* math.exp(1)**(1j*(1+action[3])*math.pi), action[4]* math.exp(1)**(1j*(1+action[5])*math.pi), action[6]* math.exp(1)**(1j*(1+action[7])*math.pi)])
                w_2=np.array([action2[2]* math.exp(1)**(1j*(1+action2[3])*math.pi), action2[4]* math.exp(1)**(1j*(1+action2[5])*math.pi), action2[6]* math.exp(1)**(1j*(1+action2[7])*math.pi)])
                w_3=np.array([action3[2]* math.exp(1)**(1j*(1+action3[3])*math.pi), action3[4]* math.exp(1)**(1j*(1+action3[5])*math.pi), action3[6]* math.exp(1)**(1j*(1+action3[7])*math.pi)])
                theta_1=cosVector([1,0,0],[state_[0]-50,state_[1]-100, 1-2])
                a_1=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_1)), math.exp(1)**(-2*1j*(math.pi*theta_1))])#
                b_1_AP_LOS=math.sqrt(PL_AP[int(state_[0]), int(state_[1])])
                h_1=b_1_AP_LOS*a_1
                interference_1=10**(-9)
                theta_2=cosVector([1,0,0],[state2_[0]-50,state2_[1]-100, 1-2])
                a_2=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_2)), math.exp(1)**(-2*1j*(math.pi*theta_2))])#
                b_2_AP_LOS=math.sqrt(PL_AP[int(state2_[0]), int(state2_[1])])
                h_2=b_2_AP_LOS*a_2
                interference_2=10**(-9)
                theta_3=cosVector([1,0,0],[state3_[0]-50,state3_[1]-100, 1-2])
                a_3=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_3)), math.exp(1)**(-2*1j*(math.pi*theta_3))])#
                b_3_AP_LOS=math.sqrt(PL_AP[int(state3_[0]), int(state3_[1])])
                h_3=b_3_AP_LOS*a_3
                interference_3=10**(-9)
                theta_4=cosVector([1,0,0],[state_su1[0]-50,state_su1[1]-100, 1-2])
                a_4=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_4)), math.exp(1)**(-2*1j*(math.pi*theta_4))])#
                b_4_AP_LOS=math.sqrt(PL_AP[int(state_su1[0]), int(state_su1[1])])
                h_4=b_4_AP_LOS*a_4
                interference_4=10**(-9)
                theta_5=cosVector([1,0,0],[state_su2[0]-50,state_su2[1]-100, 1-2])
                a_5=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_5)), math.exp(1)**(-2*1j*(math.pi*theta_5))])#
                b_5_AP_LOS=math.sqrt(PL_AP[int(state_su2[0]), int(state_su2[1])])
                h_5=b_5_AP_LOS*a_5
                interference_5=10**(-9)
                theta_6=cosVector([1,0,0],[state_su3[0]-50,state_su3[1]-100, 1-2])
                a_6=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_6)), math.exp(1)**(-2*1j*(math.pi*theta_6))])#
                b_6_AP_LOS=math.sqrt(PL_AP[int(state_su3[0]), int(state_su3[1])])
                h_6=b_6_AP_LOS*a_6
                interference_6=10**(-9)
                action1=action
                order_array=[action1[9], action2[9], action3[9]]
                order_index=[b[0] for b in sorted(enumerate(order_array), key=lambda i:i[1])]
                # action1=action
                # for order_i in order_index:
                #     exec('''if action{}[8]>0.5:
                #         interference_{}+=(1-(action{}[8]+1)/2)*(np.linalg.norm(h_{}*w_{}))**2
                #     else:
                #         interference_4+=((action[8]+1)/2)*(np.linalg.norm(h_4*w_1))**2
                #          ''')
                
                exec('''if action{}[8]>0:
    interference_{}+=(1-(action{}[8]+1)/2)*(np.linalg.norm(h_{}*w_{}))**2
else:
    interference_4+=((action{}[8]+1)/2)*(np.linalg.norm(h_4*w_{}))**2'''.format(order_index[0]+1, order_index[0]+1, order_index[0]+1, order_index[0]+1, order_index[0]+1, order_index[0]+1, order_index[0]+1))
                exec('''if action{}[8]>0:
    interference_{}+=(1-(action{}[8]+1)/2)*(np.linalg.norm(h_{}*w_{}))**2
else:
    interference_5+=((action{}[8]+1)/2)*(np.linalg.norm(h_5*w_{}))**2'''.format(order_index[1]+1, order_index[1]+1, order_index[1]+1, order_index[1]+1, order_index[1]+1, order_index[1]+1, order_index[1]+1))
                exec('''if action{}[8]>0:
    interference_{}+=(1-(action{}[8]+1)/2)*(np.linalg.norm(h_{}*w_{}))**2
else:
    interference_6+=((action{}[8]+1)/2)*(np.linalg.norm(h_6*w_{}))**2'''.format(order_index[2]+1, order_index[2]+1, order_index[2]+1, order_index[2]+1, order_index[2]+1, order_index[2]+1, order_index[2]+1))
    
                SINR_1=((action1[8]+1)/2)*(np.linalg.norm(h_1*w_1))**2/interference_1
                SINR_2=((action2[8]+1)/2)*(np.linalg.norm(h_2*w_2))**2/interference_2
                SINR_3=((action3[8]+1)/2)*(np.linalg.norm(h_3*w_3))**2/interference_3
                exec('''SINR_4=(1-(action{}[8]+1)/2)*(np.linalg.norm(h_4*w_{}))**2/interference_4'''.format(order_index[0]+1, order_index[0]+1))
                exec('''SINR_5=(1-(action{}[8]+1)/2)*(np.linalg.norm(h_5*w_{}))**2/interference_5'''.format(order_index[1]+1, order_index[1]+1))
                exec('''SINR_6=(1-(action{}[8]+1)/2)*(np.linalg.norm(h_6*w_{}))**2/interference_6'''.format(order_index[2]+1, order_index[2]+1))

                
                # calculate reward
                distance_01_2=(state_[0]-end_location[0])*(state_[0]-end_location[0])/4+(state_[1]-end_location[1])*(state_[1]-end_location[1])/4
                distance_01 = math.sqrt(distance_01_2)
                #print(distance_01)
                exec('''reward = -(distance_01/50)+max(0.01, min(SINR_1, SINR_{})/1000)-0.01'''.format(order_index.index(0)+4))
                #reward = -1
                #reward=0
                if distance_01==0:
                    reward = 1
                if not done1:
                    episode_reward += reward
                distance_02_2=(state2_[0]-end_location2[0])*(state2_[0]-end_location2[0])/4+(state2_[1]-end_location2[1])*(state2_[1]-end_location2[1])/4
                distance_02 = math.sqrt(distance_02_2)
                exec('''reward2 = -(distance_02/50)+max(0.01, min(SINR_2, SINR_{})/1000)-0.01'''.format(order_index.index(1)+4))
                if distance_02==0:
                    reward2 = 1
                if not done2:
                    episode_reward2 += reward2
                distance_03_2=(state3_[0]-end_location3[0])*(state3_[0]-end_location3[0])/4+(state3_[1]-end_location3[1])*(state3_[1]-end_location3[1])/4
                distance_03 = math.sqrt(distance_03_2)
                exec('''reward3 = -(distance_03/50)+max(0.01, min(SINR_3, SINR_{})/1000)-0.01'''.format(order_index.index(2)+4))
                if distance_03==0:
                    reward3 = 1
                if not done3:
                    episode_reward3 += reward3
                state_ = np.array(state_, dtype=np.float32)
                state2_ = np.array(state2_, dtype=np.float32)
                state3_ = np.array(state3_, dtype=np.float32)

                # if  len(buffer) >= MEMORY_CAPACITY and steps%100==0:
                #     VAR *= .99995
                #print(state)
                #done = 1 if done is True else 0
                buffer.push(state, action, reward, state_, done1)
                buffer2.push(state2, action2, reward2, state2_, done2)
                buffer3.push(state3, action3, reward3, state3_, done3)
                if not done1:
                    study=study+1
                if not done2:
                    study2=study2+1 
                if not done3:
                    study3=study3+1 
                if len(buffer) >= MEMORY_CAPACITY and not done1 and episode >= MEMORY_CAPACITY/MAX_STEPS:
                    #print("in")
                    #for i in range(20):
                    # if study>=10:
                    VAR1 *= math.sqrt(.99995)
                        # study=-1
                    agent.learn(VAR1)
                    
                if len(buffer2) >= MEMORY_CAPACITY and not done2 and episode>=MEMORY_CAPACITY/MAX_STEPS:
                    # if study2>=10:
                    #     study2=-1
                    VAR2 *= math.sqrt(.99995)
                   # for i in range(20):
                    agent2.learn(VAR2)
                    
                if len(buffer3) >= MEMORY_CAPACITY and not done3 and episode>=MEMORY_CAPACITY/MAX_STEPS:
                    #
                    # if study3>=10:
                    #     study3=-1
                    VAR3 *= math.sqrt(.99995)
                    # for i in range(20):
                    agent3.learn(VAR3)
                    
                if distance_01==0 and bobao==0:
                    done1=True
                    if steps<100:
                        for x in range(len(x_k1_array)):
                            filename = 'x_k1'+str(episode)+"_"+str(steps)+'.txt'
                            with open(filename,'a') as fileobject: #使用‘a'来提醒python用附加模式的方式打开
                                  fileobject.write(str(x_k1_array[x])+'\n')
                        for y in range(len(y_k1_array)):
                            filename = 'y_k1'+str(episode)+"_"+str(steps)+'.txt'
                            with open(filename,'a') as fileobject: #使用‘a'来提醒python用附加模式的方式打开
                                  fileobject.write(str(y_k1_array[y])+'\n')
                    print("1 arrive success!!!!!!!!!!!!!!")
                    bobao=1
                if distance_02==0 and bobao2==0:
                    if steps<100:
                        for x in range(len(x_k2_array)):
                            filename = 'x_k2'+str(episode)+"_"+str(steps)+'.txt'
                            with open(filename,'a') as fileobject: #使用‘a'来提醒python用附加模式的方式打开
                                  fileobject.write(str(x_k2_array[x])+'\n')
                        for y in range(len(y_k2_array)):
                            filename = 'y_k2'+str(episode)+"_"+str(steps)+'.txt'
                            with open(filename,'a') as fileobject: #使用‘a'来提醒python用附加模式的方式打开
                                  fileobject.write(str(y_k2_array[y])+'\n')

                    done2=True
                    print("2 arrive success!!!!!!!!!!!!!!")
                    bobao2=1
                if distance_03==0 and bobao3==0:
                    if steps<100:
                        for x in range(len(x_k3_array)):
                            filename = 'x_k3'+str(episode)+"_"+str(steps)+'.txt'
                            with open(filename,'a') as fileobject: #使用‘a'来提醒python用附加模式的方式打开
                                  fileobject.write(str(x_k3_array[x])+'\n')
                        for y in range(len(y_k3_array)):
                            filename = 'y_k3'+str(episode)+"_"+str(steps)+'.txt'
                            with open(filename,'a') as fileobject: #使用‘a'来提醒python用附加模式的方式打开
                                  fileobject.write(str(y_k3_array[y])+'\n')

                    done3=True
                    print("3 arrive success!!!!!!!!!!!!!!")
                    bobao3=1
                if done1 and done2 and done3:
                    break
                
                state = state_
                state2 = state2_
                state3 = state3_
                
                
            if episode == 0:
                all_episode_reward.append(episode_reward)
                all_episode_reward2.append(episode_reward2)
                all_episode_reward3.append(episode_reward3)
                # filename='Reward_v2_agent1.txt'
                # with open (filename, 'a') as fileobject:
                #     fileobject.write(str(episode_reward)+'\n')
                # filename='Reward_v2_agent2.txt'
                # with open (filename, 'a') as fileobject:
                #      fileobject.write(str(episode_reward2)+'\n')
                # filename='Reward_v2_agent3.txt'
                # with open (filename, 'a') as fileobject:
                #     fileobject.write(str(episode_reward3)+'\n')   
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
                all_episode_reward2.append(all_episode_reward2[-1] * 0.9 + episode_reward2 * 0.1)
                all_episode_reward3.append(all_episode_reward3[-1] * 0.9 + episode_reward3 * 0.1)
                # filename='Reward_v2_agent1.txt'
                # with open (filename, 'a') as fileobject:
                #     fileobject.write(str(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)+'\n')
                # filename='Reward_v2_agent2.txt'
                # with open (filename, 'a') as fileobject:
                #      fileobject.write(str(all_episode_reward2[-1] * 0.9 + episode_reward2 * 0.1)+'\n')
                # filename='Reward_v2_agent3.txt'
                # with open (filename, 'a') as fileobject:
                #     fileobject.write(str(all_episode_reward3[-1] * 0.9 + episode_reward3 * 0.1)+'\n')   
            #print(var_real)
            print(
                ' Episode: {}/{} | Reward: {:.4f} & {:.4f} & {:.4f}  | Step: {:.4f}| END: {}-{} {}-{} {}-{}'.format(
                    episode + 1, TRAIN_EPISODES, episode_reward, episode_reward2, episode_reward3,
                    steps, end_location, state, end_location2, state2, end_location3, state3
                ))
            #print(len(buffer3))
            
        #env.close()
        #agent.save()
        # filename = os.path.basename(path) 
        plt.plot(all_episode_reward)
        plt.plot(all_episode_reward2)
        plt.plot(all_episode_reward3)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))

    # if args.test:
    #     # test
    #     agent.load()
    #     for episode in range(TEST_EPISODES):
    #         state = env.reset().astype(np.float32)
    #         episode_reward = 0
    #         for step in range(MAX_STEPS):
    #             env.render()
    #             state, reward, done, info = env.step(agent.get_action(state, greedy=True))
    #             state = state.astype(np.float32)
    #             episode_reward += reward
    #             if done:
    #                 break
    #         print(
    #             'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f} '.format(
    #                 episode + 1, TEST_EPISODES, episode_reward,
    #                 time.time() - t0
    #             )
    #         )
    #     env.close()