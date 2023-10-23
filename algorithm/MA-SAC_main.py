 # -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
import math
# import gym
import sympy
from matplotlib import pyplot as plt
from scipy.io import loadmat
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# env = gym.make("LargeGridWorld-v0").unwrapped
state_number=2
action_number=10 #9
max_action = 1
min_action = -1
RENDER=False
EP_MAX = 500
EP_LEN = 1000
GAMMA = 0.9
q_lr = 5e-5#3e-4
value_lr = 5e-4#3e-3
policy_lr = 1.5e-4#3
BATCH = 128
tau = 1e-2
MemoryCapacity=20000
Switch=0
n_width=100
n_height = 100
m = loadmat("mapdata_0717.mat") 
#correct_action=0
MARK= m["MARK_new"]
PL_AP=m["MARK_PL_real"]


class ActorNet(nn.Module):
    def __init__(self,inp,outp):
        super(ActorNet, self).__init__()
        self.in_to_y1=nn.Linear(inp,256)
        self.in_to_y1.weight.data.normal_(0,0.1)
        self.y1_to_y2=nn.Linear(256,256)
        self.y1_to_y2.weight.data.normal_(0,0.1)
        self.out=nn.Linear(256,outp)
        self.out.weight.data.normal_(0,0.1)
        self.std_out = nn.Linear(256, outp)
        self.std_out.weight.data.normal_(0, 0.1)

    def forward(self,inputstate):
        inputstate=self.in_to_y1(inputstate)
        inputstate=F.relu(inputstate)
        inputstate=self.y1_to_y2(inputstate)
        inputstate=F.relu(inputstate)
        mean=max_action*torch.tanh(self.out(inputstate))#输出概率分布的均值mean
        log_std=self.std_out(inputstate)#softplus激活函数的值域>0
        log_std=torch.clamp(log_std,-20,2)
        std=log_std.exp()
        return mean,std

class CriticNet(nn.Module):
    def __init__(self,input,output):
        super(CriticNet, self).__init__()
        #q1
        self.in_to_y1=nn.Linear(input+output,256)
        self.in_to_y1.weight.data.normal_(0,0.1)
        self.y1_to_y2=nn.Linear(256,256)
        self.y1_to_y2.weight.data.normal_(0,0.1)
        self.out=nn.Linear(256,1)
        self.out.weight.data.normal_(0,0.1)
        #q2
        self.q2_in_to_y1 = nn.Linear(input+output, 256)
        self.q2_in_to_y1.weight.data.normal_(0, 0.1)
        self.q2_y1_to_y2 = nn.Linear(256, 256)
        self.q2_y1_to_y2.weight.data.normal_(0, 0.1)
        self.q2_out = nn.Linear(256, 1)
        self.q2_out.weight.data.normal_(0, 0.1)
    def forward(self,s,a):
        inputstate = torch.cat((s, a), dim=1)
        #q1
        q1=self.in_to_y1(inputstate)
        q1=F.relu(q1)
        q1=self.y1_to_y2(q1)
        q1=F.relu(q1)
        q1=self.out(q1)
        #q2
        q2 = self.in_to_y1(inputstate)
        q2 = F.relu(q2)
        q2 = self.y1_to_y2(q2)
        q2 = F.relu(q2)
        q2 = self.out(q2)
        return q1,q2

class Memory():
    def __init__(self,capacity,dims,type_m):
        self.capacity=capacity
        self.mem=np.zeros((capacity,dims))
        self.memory_counter=0
        self.type_m=type_m
    '''存储记忆'''
    def store_transition(self,s,a,r,s_):
        if self.type_m==1:
            tran = np.hstack((s, [a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],r], s_))  # 把s,a,r,s_困在一起，水平拼接
        else:
            tran = np.hstack((s, [a[0],r], s_))  # 把s,a,r,s_困在一起，水平拼接

        index = self.memory_counter % self.capacity#除余得索引
        self.mem[index, :] = tran  # 给索引存值，第index行所有列都为其中一次的s,a,r,s_；mem会是一个capacity行，（s+a+r+s_）列的数组
        self.memory_counter+=1
    '''随机从记忆库里抽取'''
    def sample(self,n):
        assert self.memory_counter>=self.capacity,'记忆库没有存满记忆'
        sample_index = np.random.choice(self.capacity, n)#从capacity个记忆里随机抽取n个为一批，可得到抽样后的索引号
        new_mem = self.mem[sample_index, :]#由抽样得到的索引号在所有的capacity个记忆中  得到记忆s，a，r，s_
        return new_mem
class Actor():
    def __init__(self):
        self.action_net=ActorNet(state_number,action_number)#这只是均值mean
        self.optimizer=torch.optim.Adam(self.action_net.parameters(),lr=policy_lr)

    def choose_action(self,s):
        inputstate = torch.FloatTensor(s)
        mean,std=self.action_net(inputstate)
        dist = torch.distributions.Normal(mean, std)
        action=dist.sample()
        action=torch.clamp(action,min_action,max_action)
        return action.detach().numpy()
    def evaluate(self,s):
        inputstate = torch.FloatTensor(s)
        mean,std=self.action_net(inputstate)
        dist = torch.distributions.Normal(mean, std)
        noise = torch.distributions.Normal(0, 1)
        z = noise.sample()
        action=torch.tanh(mean+std*z)
        action=torch.clamp(action,min_action,max_action)
        action_logprob=dist.log_prob(mean+std*z)-torch.log(1-action.pow(2)+1e-6)
        return action,action_logprob,z,mean,std

    def learn(self,actor_loss):
        loss=actor_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Entroy():
    def __init__(self):
        self.target_entropy = -action_number
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=q_lr)

    def learn(self,entroy_loss):
        loss=entroy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Critic():
    def __init__(self):
        self.critic_v,self.target_critic_v=CriticNet(state_number,action_number),CriticNet(state_number,action_number)#改网络输入状态，生成一个Q值
        self.optimizer = torch.optim.Adam(self.critic_v.parameters(), lr=value_lr,eps=1e-5)
        self.lossfunc = nn.MSELoss()
    def soft_update(self):
        for target_param, param in zip(self.target_critic_v.parameters(), self.critic_v.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_v(self,s,a):
        return self.critic_v(s,a)

    def learn(self,current_q1,current_q2,target_q):
        loss = self.lossfunc(current_q1, target_q) + self.lossfunc(current_q2, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def cosVector(x,y):
    result1=0.0;
    result2=0.0;
    result3=0.0;
    for i in range(len(x)):
        result1+=x[i]*y[i]   #sum(X*Y)
        result2+=x[i]**2     #sum(X*X)
        result3+=y[i]**2     #sum(Y*Y)
    return result1/((result2*result3)**0.5)


if Switch==0:
    print('SAC训练中...')
    actor = Actor()
    critic = Critic()
    entroy=Entroy()
    M = Memory(MemoryCapacity, 2 * state_number + action_number + 1,1)
    all_ep_r = []
    # actor2 = Actor()
    # critic2 = Critic()
    # entroy2=Entroy()
    # M2 = Memory(MemoryCapacity, 2 * state_number + action_number + 1,1)
    # all_ep_r2 = []
    # actor3 = Actor()
    # critic3 = Critic()
    # entroy3=Entroy()
    # M3 = Memory(MemoryCapacity, 2 * state_number + action_number + 1,1)
    # all_ep_r3 = []
    # state_number=6
    # action_number=1 #9
    # actor4 = Actor()
    # critic4 = Critic()
    # entroy4=Entroy()
    # M4 = Memory(MemoryCapacity, 2 * state_number + 1+ 1,2)
    # all_ep_r4 = []
    # state_number=2
    # action_number=9 #9
    # end_location = [26*2,18*2] #8*2,9*2[]
    
    end_location = [15*2,32*2]
    end_location2 = [45*2,45*2]
    end_location3 = [47,38*2]
    D=100
    m_d=100
    lambda_q=10

    for episode in range(EP_MAX):
        observation = np.array([4*2,5*2], dtype=np.float32)  # 环境重置
        # observation2 = np.array([20*2,20*2], dtype=np.float32)
        # observation3 = np.array([40*2,10*2], dtype=np.float32)
        observation_su1 = np.array([17, 25*2], dtype=np.float32)
        observation_su2 = np.array([50, 25*2], dtype=np.float32)
        observation_su3 = np.array([84, 25*2], dtype=np.float32)
        # observation4 = np.array([observation[0], observation[1], observation2[0], observation2[1], observation3[0], observation3[1]])

        reward_totle = 0
        reward_totle2 = 0
        reward_totle3 = 0
        reward_totle4 = 0
        done1=False
        done2=False
        done3=False
        bobao=0
        bobao2=0
        bobao3=0
        for timestep in range(EP_LEN):
            # if RENDER:
            #     env.render()
            action = actor.choose_action(observation)
            # action2 = actor2.choose_action(observation2)
            # action3 = actor3.choose_action(observation3)
            # action4 = actor4.choose_action(observation4)
            if not done1:
                [old_x, old_y] = observation
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
                observation_=np.array([new_x, new_y], dtype=np.float32)
            else:
                observation_ = observation
            # if not done2:
            #     [old_x, old_y] = observation2
            #     new_x, new_y = int(old_x), int(old_y)
            #     new_x=int(old_x+action2[0])
            #     new_y=int(old_y+action2[1])
            #     if int(new_x) <= 0: 
            #         new_x = 1 
            #     if int(new_x) >= n_width: 
            #         new_x = int(n_width)-1
            #     if int(new_y) <= 0: 
            #         new_y = 1
            #     if int(new_y) >= n_height: 
            #         new_y = int(n_height)-1
            #     if MARK[new_x,new_y] == 2:
            #         new_x, new_y = old_x, old_y
            #     observation2_=np.array([new_x, new_y], dtype=np.float32)
            # else:
            #     observation2_ = observation2
            # if not done3:
            #     [old_x, old_y] = observation3
            #     new_x, new_y = int(old_x), int(old_y)
            #     new_x=int(old_x+action3[0])
            #     new_y=int(old_y+action3[1])
            #     if int(new_x) <= 0: 
            #         new_x = 1 
            #     if int(new_x) >= n_width: 
            #         new_x = int(n_width)-1
            #     if int(new_y) <= 0: 
            #         new_y = 1
            #     if int(new_y) >= n_height: 
            #         new_y = int(n_height)-1
            #     if MARK[new_x,new_y] == 2:
            #         new_x, new_y = old_x, old_y
            #     observation3_=np.array([new_x, new_y], dtype=np.float32)
            # else:
            #     observation3_ = observation3
            # observation_ = env.step(observation, 1, action)  # 单步交互
            # observation4_ = np.array([observation_[0], observation_[1], observation2_[0], observation2_[1], observation3_[0], observation3_[1]])
            # state7_ = np.array([state_[0], state_[1], state2_[0], state2_[1], state3_[0], state3_[1]])
            # done_sys = done1 and done2 and done3
            
            if action[8]==-1:
                action[8]=-0.9999999
            # if action2[8]==-1:
            #     action2[8]=-0.9999999
            # if action3[8]==-1:
            #     action3[8]=-0.9999999
            if action[8]==1:
                action[8]=0.9999999
            # if action2[8]==1:
            #     action2[8]=0.9999999
            # if action3[8]==1:
            #     action3[8]=0.9999999

            w_1=np.array([action[2]* math.exp(1)**(1j*(1+action[3])*math.pi), action[4]* math.exp(1)**(1j*(1+action[5])*math.pi), action[6]* math.exp(1)**(1j*(1+action[7])*math.pi)])
            # w_2=np.array([action2[2]* math.exp(1)**(1j*(1+action2[3])*math.pi), action2[4]* math.exp(1)**(1j*(1+action2[5])*math.pi), action2[6]* math.exp(1)**(1j*(1+action2[7])*math.pi)])
            # w_3=np.array([action3[2]* math.exp(1)**(1j*(1+action3[3])*math.pi), action3[4]* math.exp(1)**(1j*(1+action3[5])*math.pi), action3[6]* math.exp(1)**(1j*(1+action3[7])*math.pi)])
            theta_1=cosVector([1,0,0],[observation_[0]-50,observation_[1]-100, 1-2])
            a_1=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_1)), math.exp(1)**(-2*1j*(math.pi*theta_1))])#
            b_1_AP_LOS=math.sqrt(PL_AP[int(observation_[0]), int(observation_[1])])
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
            theta_5=cosVector([1,0,0],[observation_su2[0]-50,observation_su2[1]-100, 1-2])
            a_5=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_5)), math.exp(1)**(-2*1j*(math.pi*theta_5))])#
            b_5_AP_LOS=math.sqrt(PL_AP[int(observation_su2[0]), int(observation_su2[1])])
            h_5=b_5_AP_LOS*a_5
            interference_5=10**(-9)
            theta_6=cosVector([1,0,0],[observation_su3[0]-50,observation_su3[1]-100, 1-2])
            a_6=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_6)), math.exp(1)**(-2*1j*(math.pi*theta_6))])#
            b_6_AP_LOS=math.sqrt(PL_AP[int(observation_su3[0]), int(observation_su3[1])])
            h_6=b_6_AP_LOS*a_6
            interference_6=10**(-9)
            if action[8]>0:
                interference_1+=(1-(action[8]+1)/2)*(np.linalg.norm(h_1*w_1))**2
            else:
                interference_4+=((action[8]+1)/2)*(np.linalg.norm(h_4*w_1))**2
            # if action2[8]>0.5:
            #     interference_2+=(1-(action2[8]+1)/2)*(np.linalg.norm(h_2*w_2))**2
            # else:
            #     interference_5+=((action2[8]+1)/2)*(np.linalg.norm(h_5*w_2))**2
            # if action3[8]>0.5:
            #     interference_3+=(1-(action3[8]+1)/2)*(np.linalg.norm(h_3*w_3))**2
            # else:
            #     interference_6+=((action3[8]+1)/2)*(np.linalg.norm(h_6*w_3))**2
            SINR_1=((action[8]+1)/2)*(np.linalg.norm(h_1*w_1))**2/interference_1
            # SINR_2=((action2[8]+1)/2)*(np.linalg.norm(h_2*w_2))**2/interference_2
            # SINR_3=((action3[8]+1)/2)*(np.linalg.norm(h_3*w_3))**2/interference_3
            SINR_4=(1-(action[8]+1)/2)*(np.linalg.norm(h_4*w_1))**2/interference_4
            # SINR_5=(1-(action2[8]+1)/2)*(np.linalg.norm(h_5*w_2))**2/interference_5
            # SINR_6=(1-(action3[8]+1)/2)*(np.linalg.norm(h_6*w_3))**2/interference_6
            # calculate reward
            # V_sinr_1=1-(1+SINR_1)**(-2)
            # # integrate(x**2, (x, 1, 2))
            # f_x=math.log(2)*math.sqrt(m_d/V_sinr_1)*(math.log(1+SINR_1, 2)-D/m_d)
            # x=sympy.Symbol('x')
            # f = sympy.exp(-x**2/2)
            # epsilon_d_1=1/(math.sqrt(2*math.pi))*(sympy.integrate(f, (x, f_x, sympy.oo)))
            # # print(epsilon_d_1)
            # #sympy.integrate()*math.exp(1)**(-u_var**2)/2, (u_var, f_x, float('inf')))
            # #ue 2
            # V_sinr_2=1-(1+SINR_2)**(-2)
            # f_x=math.log(2)*math.sqrt(m_d/V_sinr_2)*(math.log(1+SINR_2, 2)-D/m_d)
            # x=sympy.Symbol('x')
            # f = sympy.exp(-x**2/2)
            # epsilon_d_2=1/(math.sqrt(2*math.pi))*(sympy.integrate(f, (x, f_x, sympy.oo)))
            # #ue 3
            # V_sinr_3=1-(1+SINR_3)**(-2)
            # f_x=math.log(2)*math.sqrt(m_d/V_sinr_3)*(math.log(1+SINR_3, 2)-D/m_d)
            # x=sympy.Symbol('x')
            # f = sympy.exp(-x**2/2)
            # epsilon_d_3=1/(math.sqrt(2*math.pi))*(sympy.integrate(f, (x, f_x, sympy.oo)))
            # #ue 4
            # V_sinr_4=1-(1+SINR_4)**(-2)
            # f_x=math.log(2)*math.sqrt(m_d/V_sinr_4)*(math.log(1+SINR_4, 2)-D/m_d)
            # x=sympy.Symbol('x')
            # f = sympy.exp(-x**2/2)
            # epsilon_d_4=1/(math.sqrt(2*math.pi))*(sympy.integrate(f, (x, f_x, sympy.oo)))
            # #ue 4
            # V_sinr_5=1-(1+SINR_5)**(-2)
            # f_x=math.log(2)*math.sqrt(m_d/V_sinr_5)*(math.log(1+SINR_5, 2)-D/m_d)
            # x=sympy.Symbol('x')
            # f = sympy.exp(-x**2/2)
            # epsilon_d_5=1/(math.sqrt(2*math.pi))*(sympy.integrate(f, (x, f_x, sympy.oo)))
            # V_sinr_6=1-(1+SINR_6)**(-2)
            # f_x=math.log(2)*math.sqrt(m_d/V_sinr_6)*(math.log(1+SINR_6, 2)-D/m_d)
            # x=sympy.Symbol('x')
            # f = sympy.exp(-x**2/2)
            # epsilon_d_6=1/(math.sqrt(2*math.pi))*(sympy.integrate(f, (x, f_x, sympy.oo)))
            


            distance_01_2=(observation_[0]-end_location[0])*(observation_[0]-end_location[0])/4+(observation_[1]-end_location[1])*(observation_[1]-end_location[1])/4
            distance_01 = math.sqrt(distance_01_2)
            #print(distance_01)
            # if epsilon_d_1<10**(-14):
            #     epsilon_d_1=10**(-14)
            reward = -(distance_01/50)+max(0.01, min(SINR_1, SINR_4)/1000)-0.01
            #reward = -1
            #reward=0
            if distance_01==0:
                reward = 1
            if not done1:
                reward_totle += reward
            # if epsilon_d_2<10**(-14):
            #     epsilon_d_2=10**(-14)
            # distance_02_2=(observation2_[0]-end_location2[0])*(observation2_[0]-end_location2[0])/4+(observation2_[1]-end_location2[1])*(observation2_[1]-end_location2[1])/4
            # distance_02 = math.sqrt(distance_02_2)
            # reward2 = -(distance_02/50)#+max(0.2, min(SINR_2, SINR_5)/50)
            # if distance_02==0:
            #     reward2 = 1
            # if not done2:
            #     reward_totle2 += reward2
            # distance_03_2=(observation3_[0]-end_location3[0])*(observation3_[0]-end_location3[0])/4+(observation3_[1]-end_location3[1])*(observation3_[1]-end_location3[1])/4
            # distance_03 = math.sqrt(distance_03_2)
            # # if epsilon_d_3<10**(-14):
            # #     epsilon_d_3=10**(-14)
            # reward3 = -(distance_03/50)#+max(0.2, min(SINR_3, SINR_6)/50)
            # if distance_03==0:
            #     reward3 = 1
            # if not done3:
            #     reward_totle3 += reward3
            
            # reward4=(reward+reward2+reward3)/3
            
            
            # distance_01_2=(observation_[0]-end_location[0])*(observation_[0]-end_location[0])/4+(observation_[1]-end_location[1])*(observation_[1]-end_location[1])/4
            # distance_01 = math.sqrt(distance_01_2)
            # reward= -(distance_01/10)
            # if distance_01==0:
            #     done1 = True
            #     #os.system("pause")
            #     reward=10
            #print(observation, action, observation_)
            M.store_transition(observation, action, reward, observation_)
            # M2.store_transition(observation2, action2, reward2, observation2_)
            # M3.store_transition(observation3, action3, reward3, observation3_)
            # M4.store_transition(observation4, action4, reward4, observation4_)


            # 记忆库存储
            # 有的2000个存储数据就开始学习
            if M.memory_counter > MemoryCapacity and not done1:
                b_M = M.sample(BATCH)
                b_s = b_M[:, :state_number]
                b_a = b_M[:, state_number: state_number + action_number]
                b_r = b_M[:, -state_number - 1: -state_number]
                b_s_ = b_M[:, -state_number:]
                b_s = torch.FloatTensor(b_s)
                b_a = torch.FloatTensor(b_a)
                b_r = torch.FloatTensor(b_r)
                b_s_ = torch.FloatTensor(b_s_)
                new_action, log_prob_, z, mean, log_std = actor.evaluate(b_s_)
                target_q1,target_q2=critic.get_v(b_s_,new_action)
                target_q=b_r+GAMMA*(torch.min(target_q1,target_q2)-entroy.alpha*log_prob_)
                current_q1, current_q2 = critic.get_v(b_s, b_a)
                critic.learn(current_q1,current_q2,target_q.detach())
                a,log_prob,_,_,_=actor.evaluate(b_s)
                q1,q2=critic.get_v(b_s,a)
                q=torch.min(q1,q2)
                actor_loss = (entroy.alpha * log_prob - q).mean()
                actor.learn(actor_loss)
                alpha_loss = -(entroy.log_alpha.exp() * (log_prob + entroy.target_entropy).detach()).mean()
                entroy.learn(alpha_loss)
                entroy.alpha=entroy.log_alpha.exp()
                # 软更新
                critic.soft_update()
            observation = observation_
            # reward_totle += reward
            if distance_01==0:
                done1=True
                # print("arrive success!!!!!!!!!!!!!!")
            # if M2.memory_counter > MemoryCapacity and not done2:
            #     b_M = M2.sample(BATCH)
            #     b_s = b_M[:, :state_number]
            #     b_a = b_M[:, state_number: state_number + action_number]
            #     b_r = b_M[:, -state_number - 1: -state_number]
            #     b_s_ = b_M[:, -state_number:]
            #     b_s = torch.FloatTensor(b_s)
            #     b_a = torch.FloatTensor(b_a)
            #     b_r = torch.FloatTensor(b_r)
            #     b_s_ = torch.FloatTensor(b_s_)
            #     new_action, log_prob_, z, mean, log_std = actor2.evaluate(b_s_)
            #     target_q1,target_q2=critic2.get_v(b_s_,new_action)
            #     target_q=b_r+GAMMA*(torch.min(target_q1,target_q2)-entroy2.alpha*log_prob_)
            #     current_q1, current_q2 = critic2.get_v(b_s, b_a)
            #     critic2.learn(current_q1,current_q2,target_q.detach())
            #     a,log_prob,_,_,_=actor2.evaluate(b_s)
            #     q1,q2=critic2.get_v(b_s,a)
            #     q=torch.min(q1,q2)
            #     actor_loss = (entroy2.alpha * log_prob - q).mean()
            #     actor2.learn(actor_loss)
            #     alpha_loss = -(entroy2.log_alpha.exp() * (log_prob + entroy2.target_entropy).detach()).mean()
            #     entroy2.learn(alpha_loss)
            #     entroy2.alpha=entroy2.log_alpha.exp()
            #     # 软更新
            #     critic2.soft_update()
            # observation2 = observation2_
            # # reward_totle2 += reward2
            # if distance_02==0:
            #     done2=True
            #     # print("arrive success 2 !!!!!!!!!!!!!!")
            # if M3.memory_counter > MemoryCapacity and not done3:
            #     b_M = M3.sample(BATCH)
            #     b_s = b_M[:, :state_number]
            #     b_a = b_M[:, state_number: state_number + action_number]
            #     b_r = b_M[:, -state_number - 1: -state_number]
            #     b_s_ = b_M[:, -state_number:]
            #     b_s = torch.FloatTensor(b_s)
            #     b_a = torch.FloatTensor(b_a)
            #     b_r = torch.FloatTensor(b_r)
            #     b_s_ = torch.FloatTensor(b_s_)
            #     new_action, log_prob_, z, mean, log_std = actor3.evaluate(b_s_)
            #     target_q1,target_q3=critic3.get_v(b_s_,new_action)
            #     target_q=b_r+GAMMA*(torch.min(target_q1,target_q3)-entroy3.alpha*log_prob_)
            #     current_q1, current_q3 = critic3.get_v(b_s, b_a)
            #     critic3.learn(current_q1,current_q3,target_q.detach())
            #     a,log_prob,_,_,_=actor3.evaluate(b_s)
            #     q1,q3=critic3.get_v(b_s,a)
            #     q=torch.min(q1,q3)
            #     actor_loss = (entroy3.alpha * log_prob - q).mean()
            #     actor3.learn(actor_loss)
            #     alpha_loss = -(entroy3.log_alpha.exp() * (log_prob + entroy3.target_entropy).detach()).mean()
            #     entroy3.learn(alpha_loss)
            #     entroy3.alpha=entroy3.log_alpha.exp()
            #     # 软更新
            #     critic3.soft_update()
            # observation3 = observation3_
            # # reward_totle += reward
            # if distance_03==0:
            #     done3=True
                # print("arrive success 3!!!!!!!!!!!!!!")
            # state_number=6
            # action_number=1 
            # if M4.memory_counter > MemoryCapacity:
            #     b_M = M4.sample(BATCH)
            #     b_s = b_M[:, :state_number]
            #     b_a = b_M[:, state_number: state_number + action_number]
            #     b_r = b_M[:, -state_number - 1: -state_number]
            #     b_s_ = b_M[:, -state_number:]
            #     b_s = torch.FloatTensor(b_s)
            #     b_a = torch.FloatTensor(b_a)
            #     b_r = torch.FloatTensor(b_r)
            #     b_s_ = torch.FloatTensor(b_s_)
            #     new_action, log_prob_, z, mean, log_std = actor4.evaluate(b_s_)
            #     target_q1,target_q4=critic4.get_v(b_s_,new_action)
            #     target_q=b_r+GAMMA*(torch.min(target_q1,target_q4)-entroy4.alpha*log_prob_)
            #     current_q1, current_q4 = critic4.get_v(b_s, b_a)
            #     critic4.learn(current_q1,current_q4,target_q.detach())
            #     a,log_prob,_,_,_=actor4.evaluate(b_s)
            #     q1,q4=critic4.get_v(b_s,a)
            #     q=torch.min(q1,q4)
            #     actor_loss = (entroy4.alpha * log_prob - q).mean()
            #     actor4.learn(actor_loss)
            #     alpha_loss = -(entroy4.log_alpha.exp() * (log_prob + entroy4.target_entropy).detach()).mean()
            #     entroy4.learn(alpha_loss)
            #     entroy4.alpha=entroy4.log_alpha.exp()
            #     # 软更新
            #     critic4.soft_update()
            # observation4 = observation4_
            if done1:
                # print("arrive success!!!!!!!!!!!!!!")
                break
        print("Ep: {} | rewards: {} {} {} {} | Step: {:.4f} | END: {}".format(episode, reward_totle, reward_totle2, reward_totle3, reward_totle4, timestep, observation))
        # if reward_totle > -10: RENDER = True
        all_ep_r.append(reward_totle)
        # all_ep_r2.append(reward_totle2)
        # all_ep_r3.append(reward_totle3)
        # all_ep_r4.append(reward_totle4)
        #if episode % 20 == 0 and episode > 200:#保存神经网络参数
         #   save_data = {'net': actor.action_net.observation_dict(), 'opt': actor.optimizer.state_dict(), 'i': episode}
            #torch.save(save_data, "C:\\Users\Administrator.DESKTOP-NLH290A\Desktop\2302_code\0606\model_SAC.pth")
    # env.close()
    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    # plt.plot(np.arange(len(all_ep_r2)), all_ep_r2)
    # plt.plot(np.arange(len(all_ep_r3)), all_ep_r3)
    # plt.plot(np.arange(len(all_ep_r4)), all_ep_r4)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()
else:
    print('SAC测试中...')
    aa=Actor()
    checkpoint_aa = torch.load("C:\\Users\Administrator.DESKTOP-NLH290A\Desktop\2302_code\0606\model_SAC.pth")
    aa.action_net.load_state_dict(checkpoint_aa['net'])
    for j in range(10):
        # state = env.reset()
        total_rewards = 0
        for timestep in range(EP_LEN):
            # env.render()
            # action = aa.choose_action(state)
            # new_state, reward, done, info = env.step(action)  # 执行动作
            total_rewards += reward
            # state = new_state
        print("Score：", total_rewards)
    # env.close()
