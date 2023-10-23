from td3 import TD3
# import gym
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import math

# import random
# def random_int_list(start, stop, length):
#     start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
#     length = int(abs(length)) if length else 0
#     random_list = []
#     for i in range(length):
#         random_list.append(random.randint(start, stop))
#     return random_list


def cosVector(x,y):
    result1=0.0;
    result2=0.0;
    result3=0.0;
    for i in range(len(x)):
        result1+=x[i]*y[i]   #sum(X*Y)
        result2+=x[i]**2     #sum(X*X)
        result3+=y[i]**2     #sum(Y*Y)
    return result1/((result2*result3)**0.5)


if __name__ == '__main__':
    # env = gym.make('Pendulum-v1')
    obs_dim = 2
    act_dim = 10
    td3_1 = TD3(obs_dim,act_dim)
    td3_2 = TD3(obs_dim,act_dim)
    td3_3 = TD3(obs_dim,act_dim)
    n_width=100
    n_height = 100
    m = loadmat("mapdata_0717.mat") 
    #correct_action=0
    MARK= m["MARK_new"]
    PL_AP=m["MARK_PL_real"]
    n_mu=3
    n_M=5
    n_o=6*7
    all_episode_reward = []
    all_episode_reward2 = []
    all_episode_reward3 = []
    MAX_EPISODE = 500
    MAX_STEP = 1000
    update_every = 50
    batch_size = 100
    # rewardList = []
    # rewardList2 = []
    # rewardList3 = []
    end_location = [15*2,32*2]
    end_location2 = [45*2,45*2]
    end_location3 = [47,38*2]
    # RANDOM_SEED=2
    # random.seed(RANDOM_SEED)
    # np.random.seed(RANDOM_SEED)
    # torch.seed(RANDOM_SEED)
    for episode in range(MAX_EPISODE):
        o_1 = np.array([4*2,5*2], dtype=np.float32)  # 环境重置
        o_2 = np.array([20*2,20*2], dtype=np.float32)
        o_3 = np.array([40*2,10*2], dtype=np.float32)
        state_su1 = np.array([17, 25*2], dtype=np.float32)
        state_su2 = np.array([50, 25*2], dtype=np.float32)
        state_su3 = np.array([84, 25*2], dtype=np.float32)
        ep_reward=0
        ep_reward2=0
        ep_reward3=0
        done_record3=False
        done_record1=False
        done_record2=False
        done1=False
        done2=False
        done3=False
        distance_01_max=math.sqrt((o_1[0]-end_location[0])*(o_1[0]-end_location[0])/4+(o_1[1]-end_location[1])*(o_1[1]-end_location[1])/4)
        distance_02_max=math.sqrt((o_2[0]-end_location[0])*(o_2[0]-end_location[0])/4+(o_2[1]-end_location[1])*(o_2[1]-end_location[1])/4)
        distance_03_max=math.sqrt((o_3[0]-end_location[0])*(o_3[0]-end_location[0])/4+(o_3[1]-end_location[1])*(o_3[1]-end_location[1])/4)
        x_k1_array = []
        y_k1_array = []
        x_k2_array = []
        y_k2_array = []
        x_k3_array = []
        y_k3_array = []
        x_k1_array.append(o_1[0])
        y_k1_array.append(o_1[1])
        x_k2_array.append(o_2[0])
        y_k2_array.append(o_2[1])
        x_k3_array.append(o_3[0])
        y_k3_array.append(o_3[1])
        for j in range(MAX_STEP):
            if episode > 20:
                a_1 = td3_1.get_action(o_1, td3_1.act_noise)*2
                a_2 = td3_2.get_action(o_2, td3_2.act_noise)*2
                a_3 = td3_3.get_action(o_3, td3_3.act_noise)*2
            else:
                a_1 = np.random.rand(10)*2-1
                a_2 = np.random.rand(10)*2-1
                a_3 = np.random.rand(10)*2-1
            if not done1:
                [old_x, old_y] = o_1
                new_x, new_y = int(old_x), int(old_y)
                new_x=int(old_x+a_1[0])
                new_y=int(old_y+a_1[1])
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
                o2_1=np.array([new_x, new_y], dtype=np.float32)
                x_k1_array.append(o2_1[0])
                y_k1_array.append(o2_1[1])

            else:
                o2_1 = o_1
            if not done2:
                [old_x, old_y] = o_2
                new_x, new_y = int(old_x), int(old_y)
                new_x=int(old_x+a_2[0])
                new_y=int(old_y+a_2[1])
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
                o2_2=np.array([new_x, new_y], dtype=np.float32)
                x_k2_array.append(o2_2[0])
                y_k2_array.append(o2_2[1])

            else:
                o2_2 = o_2
            if not done3:
                [old_x, old_y] = o_3
                new_x, new_y = int(old_x), int(old_y)
                new_x=int(old_x+a_3[0])
                new_y=int(old_y+a_3[1])
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
                o2_3=np.array([new_x, new_y], dtype=np.float32)
                x_k3_array.append(o2_3[0])
                y_k3_array.append(o2_3[1])

            else:
                o2_3 = o_3
                
                
            if a_1[8]==-1:
                a_1[8]=-0.9999999
            if a_2[8]==-1:
                a_2[8]=-0.9999999
            if a_3[8]==-1:
                a_3[8]=-0.9999999
            if a_1[8]==1:
                a_1[8]=0.9999999
            if a_2[8]==1:
                a_2[8]=0.9999999
            if a_3[8]==1:
                a_3[8]=0.9999999
            
            w_1=np.array([a_1[2]* math.exp(1)**(1j*(1+a_1[3])*math.pi), a_1[4]* math.exp(1)**(1j*(1+a_1[5])*math.pi), a_1[6]* math.exp(1)**(1j*(1+a_1[7])*math.pi)])
            w_2=np.array([a_2[2]* math.exp(1)**(1j*(1+a_2[3])*math.pi), a_2[4]* math.exp(1)**(1j*(1+a_2[5])*math.pi), a_2[6]* math.exp(1)**(1j*(1+a_2[7])*math.pi)])
            w_3=np.array([a_3[2]* math.exp(1)**(1j*(1+a_3[3])*math.pi), a_3[4]* math.exp(1)**(1j*(1+a_3[5])*math.pi), a_3[6]* math.exp(1)**(1j*(1+a_3[7])*math.pi)])
            theta_1=cosVector([1,0,0],[o2_1[0]-50,o2_1[1]-100, 1-2])
            aLP_1=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_1)), math.exp(1)**(-2*1j*(math.pi*theta_1))])#
            b_1_AP_LOS=math.sqrt(PL_AP[int(o2_1[0]), int(o2_1[1])])
            h_1=b_1_AP_LOS*aLP_1
            interference_1=10**(-9)
            theta_2=cosVector([1,0,0],[o2_2[0]-50,o2_2[1]-100, 1-2])
            aLP_2=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_2)), math.exp(1)**(-2*1j*(math.pi*theta_2))])#
            b_2_AP_LOS=math.sqrt(PL_AP[int(o2_2[0]), int(o2_2[1])])
            h_2=b_2_AP_LOS*aLP_2
            interference_2=10**(-9)
            theta_3=cosVector([1,0,0],[o2_3[0]-50,o2_3[1]-100, 1-2])
            aLP_3=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_3)), math.exp(1)**(-2*1j*(math.pi*theta_3))])#
            b_3_AP_LOS=math.sqrt(PL_AP[int(o2_3[0]), int(o2_3[1])])
            h_3=b_3_AP_LOS*aLP_3
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
            
            order_array=[a_1[9], a_2[9], a_3[9]]
            order_index=[b[0] for b in sorted(enumerate(order_array), key=lambda i:i[1])]
            # action1=action
            # for order_i in order_index:
            #     exec('''if a_{}[8]>0.5:
            #         interference_{}+=(1-(a_{}[8]+1)/2)*(np.linalg.norm(h_{}*w_{}))**2
            #     else:
            #         interference_4+=((a_[8]+1)/2)*(np.linalg.norm(h_4*w_1))**2
            #          ''')
            
            exec('''if a_{}[8]>0:
    interference_{}+=(1-(a_{}[8]+1)/2)*(np.linalg.norm(h_{}*w_{}))**2
else:
    interference_4+=((a_{}[8]+1)/2)*(np.linalg.norm(h_4*w_{}))**2'''.format(order_index[0]+1, order_index[0]+1, order_index[0]+1, order_index[0]+1, order_index[0]+1, order_index[0]+1, order_index[0]+1))
            exec('''if a_{}[8]>0:
    interference_{}+=(1-(a_{}[8]+1)/2)*(np.linalg.norm(h_{}*w_{}))**2
else:
    interference_5+=((a_{}[8]+1)/2)*(np.linalg.norm(h_5*w_{}))**2'''.format(order_index[1]+1, order_index[1]+1, order_index[1]+1, order_index[1]+1, order_index[1]+1, order_index[1]+1, order_index[1]+1))
            exec('''if a_{}[8]>0:
    interference_{}+=(1-(a_{}[8]+1)/2)*(np.linalg.norm(h_{}*w_{}))**2
else:
    interference_6+=((a_{}[8]+1)/2)*(np.linalg.norm(h_6*w_{}))**2'''.format(order_index[2]+1, order_index[2]+1, order_index[2]+1, order_index[2]+1, order_index[2]+1, order_index[2]+1, order_index[2]+1))

            SINR_1=((a_1[8]+1)/2)*(np.linalg.norm(h_1*w_1))**2/interference_1
            SINR_2=((a_2[8]+1)/2)*(np.linalg.norm(h_2*w_2))**2/interference_2
            SINR_3=((a_3[8]+1)/2)*(np.linalg.norm(h_3*w_3))**2/interference_3
            exec('''SINR_4=(1-(a_{}[8]+1)/2)*(np.linalg.norm(h_4*w_{}))**2/interference_4'''.format(order_index[0]+1, order_index[0]+1))
            exec('''SINR_5=(1-(a_{}[8]+1)/2)*(np.linalg.norm(h_5*w_{}))**2/interference_5'''.format(order_index[1]+1, order_index[1]+1))
            exec('''SINR_6=(1-(a_{}[8]+1)/2)*(np.linalg.norm(h_6*w_{}))**2/interference_6'''.format(order_index[2]+1, order_index[2]+1))

            
            
            # o2_= env.step(state,1, a_)
            # o2_ = o2_.astype(np.float32)
            distance_01_2=(o2_1[0]-end_location[0])*(o2_1[0]-end_location[0])/4+(o2_1[1]-end_location[1])*(o2_1[1]-end_location[1])/4
            distance_01 = math.sqrt(distance_01_2)
            exec('''reward = -(distance_01/50)+max(0.08, min(SINR_1, SINR_{})/500)-0.08'''.format(order_index.index(0)+4))
            if distance_01==0:
                done_record1 = True
                #os.system("pause")
                reward=1
            if not done1:
                ep_reward += reward
            # if epsilon_d_2<10**(-14):
            #     epsilon_d_2=10**(-14)
            distance_02_2=(o2_2[0]-end_location2[0])*(o2_2[0]-end_location2[0])/4+(o2_2[1]-end_location2[1])*(o2_2[1]-end_location2[1])/4
            distance_02 = math.sqrt(distance_02_2)
            exec('''reward2 = -(distance_02/50)+max(0.08, min(SINR_2, SINR_{})/500)-0.08'''.format(order_index.index(1)+4))
            if distance_02==0:
                reward2 = 1
                done_record2=True
            if not done2:
                ep_reward2 += reward2
            distance_03_2=(o2_3[0]-end_location3[0])*(o2_3[0]-end_location3[0])/4+(o2_3[1]-end_location3[1])*(o2_3[1]-end_location3[1])/4
            distance_03 = math.sqrt(distance_03_2)
            # if epsilon_d_3<10**(-14):
            #     epsilon_d_3=10**(-14)
            exec('''reward3 = -(distance_03/50)+max(0.08, min(SINR_3, SINR_{})/500)-0.08'''.format(order_index.index(1)+4))
            if distance_03==0:
                reward3 = 1
                done_record3=True
            if not done3:
                ep_reward3 += reward3
    
            
            if not done1:
                td3_1.replay_buffer.store(o_1, a_1, reward, o2_1, done1)
                if episode >= 20 and j % update_every == 0:
                    td3_1.update(batch_size,update_every)
                o_1 = o2_1
            if not done2:
                td3_2.replay_buffer.store(o_2, a_2, reward2, o2_2, done2)
                if episode >= 20 and j % update_every == 0:
                    td3_2.update(batch_size,update_every)
                o_2 = o2_2
            if not done3:
                td3_3.replay_buffer.store(o_3, a_3, reward3, o2_3, done3)
                if episode >= 20 and j % update_every == 0:
                    td3_3.update(batch_size,update_every)
                o_3 = o2_3
            if distance_01==0 and not done1:
                if j<100:
                    for x in range(len(x_k1_array)):
                        filename = 'x_k1'+str(episode)+"_"+str(j)+'.txt'
                        with open(filename,'a') as fileobject: #使用‘a'来提醒python用附加模式的方式打开
                              fileobject.write(str(x_k1_array[x])+'\n')
                    for y in range(len(y_k1_array)):
                        filename = 'y_k1'+str(episode)+"_"+str(j)+'.txt'
                        with open(filename,'a') as fileobject: #使用‘a'来提醒python用附加模式的方式打开
                              fileobject.write(str(y_k1_array[y])+'\n')

                done1=True
            if distance_02==0 and not done2:
                if j<100:
                    for x in range(len(x_k2_array)):
                        filename = 'x_k2'+str(episode)+"_"+str(j)+'.txt'
                        with open(filename,'a') as fileobject: #使用‘a'来提醒python用附加模式的方式打开
                              fileobject.write(str(x_k2_array[x])+'\n')
                    for y in range(len(y_k2_array)):
                        filename = 'y_k2'+str(episode)+"_"+str(j)+'.txt'
                        with open(filename,'a') as fileobject: #使用‘a'来提醒python用附加模式的方式打开
                              fileobject.write(str(y_k2_array[y])+'\n')

                done2=True
            if distance_03==0 and not done3:
                if j<100:
                    for x in range(len(x_k3_array)):
                        filename = 'x_k3'+str(episode)+"_"+str(j)+'.txt'
                        with open(filename,'a') as fileobject: #使用‘a'来提醒python用附加模式的方式打开
                              fileobject.write(str(x_k3_array[x])+'\n')
                    for y in range(len(y_k3_array)):
                        filename = 'y_k3'+str(episode)+"_"+str(j)+'.txt'
                        with open(filename,'a') as fileobject: #使用‘a'来提醒python用附加模式的方式打开
                              fileobject.write(str(y_k3_array[y])+'\n')

                done3=True
                # ep_reward += r
            if done1 and done2 and done3: break
        
        if episode == 0:
            all_episode_reward.append(ep_reward)
            all_episode_reward2.append(ep_reward2)
            all_episode_reward3.append(ep_reward3)
        else:
            all_episode_reward.append(all_episode_reward[-1] * 0.9 + ep_reward * 0.1)
            all_episode_reward2.append(all_episode_reward2[-1] * 0.9 + ep_reward2 * 0.1)
            all_episode_reward3.append(all_episode_reward3[-1] * 0.9 + ep_reward3 * 0.1)
        print(
            'Training  | Episode: {}/{}  | Episode Reward: {:.4f}   {:.4f}   {:.4f} '.format(
                episode + 1, MAX_EPISODE, ep_reward, ep_reward2, ep_reward3
            ))

        # print('Episode:', episode, 'Reward:%i' % int(ep_reward))
        # rewardList.append(ep_reward)

    plt.figure()
    plt.plot(np.arange(len(all_episode_reward)),all_episode_reward)
    plt.plot(np.arange(len(all_episode_reward2)),all_episode_reward2)
    plt.plot(np.arange(len(all_episode_reward3)),all_episode_reward3)
    plt.show()
