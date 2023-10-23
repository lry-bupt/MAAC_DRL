import tensorflow as tf
from tensorflow import keras
from keras.layers import *
import numpy as np
import gym
import matplotlib.pyplot as plt
import math
from scipy.io import loadmat
np.random.seed(2)
tf.random.set_seed(2)

EP_MAX = 500
BATCH = 32
EP_LEN = 1000
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0005

A_UPDATE_STEPS = 20
C_UPDATE_STEPS = 20
S_DIM, A_DIM = 2, 10
epsilon=0.2

n_width=100
n_height = 100
m = loadmat("mapdata_0717.mat") 
#correct_action=0
MARK= m["MARK_new"]
PL_AP=m["MARK_PL_real"]
n_mu=3
n_M=5
n_o=6*7
a_bound=1

class PPO(object):

    def __init__(self):
        self.opt_a = tf.compat.v1.train.AdamOptimizer(A_LR)
        self.opt_c = tf.compat.v1.train.AdamOptimizer(C_LR)

        self.model_a = self._build_anet(trainable=True)
        self.model_a_old = self._build_anet(trainable=False)
        self.model_c = self._build_cnet()

    def _build_anet(self,trainable=True):
        tfs_a = Input([S_DIM], )
        l1 = Dense(100, 'relu',trainable=trainable)(tfs_a)
        mu = a_bound * Dense(A_DIM, 'tanh',trainable=trainable)(l1)
        sigma = Dense(A_DIM, 'softplus',trainable=trainable)(l1)
        model_a = keras.models.Model(inputs=tfs_a, outputs=[mu, sigma])
        return model_a

    def _build_cnet(self):
        tfs_c = Input([S_DIM], )
        l1 = Dense(100, 'relu')(tfs_c)
        v = Dense(1)(l1)
        model_c = keras.models.Model(inputs=tfs_c, outputs=v)
        model_c.compile(optimizer=self.opt_c, loss='mse')
        return model_c

    def update(self, s, a, r):
        self.model_a_old.set_weights(self.model_a.get_weights())

        mu, sigma = self.model_a_old(s)
        oldpi = tf.compat.v1.distributions.Normal(loc=mu, scale=sigma)
        old_prob_a = oldpi.prob(a)

        v = self.get_v(s)
        adv = r - v

        for i in range(A_UPDATE_STEPS):
            with tf.GradientTape() as tape:
                mu, sigma = self.model_a(s)
                pi = tf.compat.v1.distributions.Normal(loc=mu, scale=sigma)
                ratio = pi.prob(a) / (old_prob_a + 1e-5)
                surr = ratio * adv
                x2 = tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon) * adv
                x3 = tf.minimum(surr, x2)
                aloss = -tf.reduce_mean(x3)

            a_grads = tape.gradient(aloss, self.model_a.trainable_weights)
            a_grads_and_vars = zip(a_grads, self.model_a.trainable_weights)
            self.opt_a.apply_gradients(a_grads_and_vars)

        self.model_c.fit(s, r, verbose=0, shuffle=False,epochs=C_UPDATE_STEPS)

    def choose_action(self, s):
        mu, sigma = self.model_a(s)
        pi = tf.compat.v1.distributions.Normal(loc=mu, scale=sigma)
        a = tf.squeeze(pi.sample(1), axis=0)
        return np.clip(a, -2, 2)

    def get_v(self, s):
        v = self.model_c(s)
        return v

def cosVector(x,y):
    result1=0.0;
    result2=0.0;
    result3=0.0;
    for i in range(len(x)):
        result1+=x[i]*y[i]   #sum(X*Y)
        result2+=x[i]**2     #sum(X*X)
        result3+=y[i]**2     #sum(Y*Y)
    return result1/((result2*result3)**0.5)

ppo = PPO()
end_location = [15*2,32*2]
all_ep_r = []
all_ep_reward_p=[]
for ep in range(EP_MAX):                    #train
    s = np.array([4*2,5*2], dtype=np.float32) 
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    done1 = False
    distance_01_max=math.sqrt((s[0]-end_location[0])*(s[0]-end_location[0])/4+(s[1]-end_location[1])*(s[1]-end_location[1])/4)

    s = np.reshape(s, (-1, S_DIM))
    observation_su1 = np.array([17, 25*2], dtype=np.float32)    
    for t in range(EP_LEN):  # in one episode
        a = ppo.choose_action(s)
        if not done1:
            [old_x, old_y] = s[0]
            new_x, new_y = int(old_x), int(old_y)
            new_x=int(old_x+a[0,0])
            new_y=int(old_y+a[0,1])
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
            s_=np.array([new_x, new_y], dtype=np.float32)
        else:
            s_ = s
        a=a[0]
        if a[8]==-1:
            a[8]=-0.9999999
        # if action2[8]==-1:
        #     action2[8]=-0.9999999
        # if action3[8]==-1:
        #     action3[8]=-0.9999999
        if a[8]==1:
            a[8]=0.9999999
        
        w_1=np.array([a[2]* math.exp(1)**(1j*(1+a[3])*math.pi), a[4]* math.exp(1)**(1j*(1+a[5])*math.pi), a[6]* math.exp(1)**(1j*(1+a[7])*math.pi)])
        # w_2=np.array([action2[2]* math.exp(1)**(1j*(1+action2[3])*math.pi), action2[4]* math.exp(1)**(1j*(1+action2[5])*math.pi), action2[6]* math.exp(1)**(1j*(1+action2[7])*math.pi)])
        # w_3=np.array([action3[2]* math.exp(1)**(1j*(1+action3[3])*math.pi), action3[4]* math.exp(1)**(1j*(1+action3[5])*math.pi), action3[6]* math.exp(1)**(1j*(1+action3[7])*math.pi)])
        theta_1=cosVector([1,0,0],[s_[0]-50,s_[1]-100, 1-2])
        a_1=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_1)), math.exp(1)**(-2*1j*(math.pi*theta_1))])#
        b_1_AP_LOS=math.sqrt(PL_AP[int(s_[0]), int(s_[1])])
        h_1=b_1_AP_LOS*a_1
        interference_1=10**(-9)
        # theta_2=cosVector([1,0,0],[observation2_[0]-50,observation2_[1]-100, 1-2])
        # a_2=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_2)), math.exp(1)**(-2*1j*(math.pi*theta_2))])#
        # b_2_AP_LOS=math.sqrt(PL_AP[int(observation2_[0]), int(observation2_[1])])
        # h_2=b_2_AP_LOS*a_2
        # interference_2=10**(-9)
        # theta_3=cosVector([1,0,0],[observation3_[0]-50,observation3_[1]-100, 1-2])
        # a_3=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_3)), math.exp(1)**(-2*1j*(math.pi*theta_3))])#
        # b_3_AP_LOS=math.sqrt(PL_AP[int(observation3_[0]), int(observation3_[1])])
        # h_3=b_3_AP_LOS*a_3
        # interference_3=10**(-9)
        theta_4=cosVector([1,0,0],[observation_su1[0]-50,observation_su1[1]-100, 1-2])
        a_4=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_4)), math.exp(1)**(-2*1j*(math.pi*theta_4))])#
        b_4_AP_LOS=math.sqrt(PL_AP[int(observation_su1[0]), int(observation_su1[1])])
        h_4=b_4_AP_LOS*a_4
        interference_4=10**(-9)
        
        if a[8]>0:
            interference_1+=(1-(a[8]+1)/2)*(np.linalg.norm(h_1*w_1))**2
        else:
            interference_4+=((a[8]+1)/2)*(np.linalg.norm(h_4*w_1))**2
            
        SINR_1=((a[8]+1)/2)*(np.linalg.norm(h_1*w_1))**2/interference_1
         # SINR_2=((action2[8]+1)/2)*(np.linalg.norm(h_2*w_2))**2/interference_2
         # SINR_3=((action3[8]+1)/2)*(np.linalg.norm(h_3*w_3))**2/interference_3
        SINR_4=(1-(a[8]+1)/2)*(np.linalg.norm(h_4*w_1))**2/interference_4

        
        buffer_s.append(s)
        buffer_a.append(a)
        distance_01_2=(s_[0]-end_location[0])*(s_[0]-end_location[0])/4+(s_[1]-end_location[1])*(s_[1]-end_location[1])/4
        distance_01 = math.sqrt(distance_01_2)
        s_ = np.reshape(s_, (-1, S_DIM))
        r= -(distance_01/50)
        if distance_01==0:
            done1 = True
            #os.system("pause")
            r=1
        r = np.reshape(r, (-1, 1))
        buffer_r.append(r)  # normalize reward, find to be useful
        a = np.reshape(a, (-1, A_DIM))
        s_ = np.reshape(s_, (-1, S_DIM))
        s = s_
        ep_r += r[0]

        # update ppo
        if (t + 1) % BATCH == 0 or t == EP_LEN - 1 or done1:
            #print("here")
            v_s_ = ppo.get_v(s_)[0,0]
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs = np.vstack(buffer_s)
            ba = np.vstack(buffer_a)
            br = np.array(discounted_r)
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br)
        if done1:
            print("success!!!!!!!!!!!!")
            break
    if ep == 0:
        # all_ep_r.append(ep_r)
        all_ep_reward_p.append(ep_r)
    else:
        # all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
        all_ep_reward_p.append(all_ep_reward_p[-1] * 0.9 + ep_r * 0.1)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
    )

plt.plot(all_ep_reward_p)



# while 1:                        #play
#     s = env.reset()
#     for t in range(EP_LEN):
#         s = s.reshape([-1, S_DIM])
#         env.render()
#         s, r, done, info = env.step(ppo.choose_action(s))
#         if done:
#             break
