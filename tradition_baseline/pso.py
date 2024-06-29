import numpy as np
from scipy.special import erfcinv
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.io import loadmat
import math
import numpy as np
from scipy.special import erfc

m = loadmat("C://Users/Administrator.DESKTOP-NLH290A/Desktop/2302_code/FIGURE_3/mapdata_0717.mat") 
#correct_action=0
MARK= m["MARK_new"]
PL_AP=m["MARK_PL_real"]

def cosVector(x,y):
    result1=0.0;
    result2=0.0;
    result3=0.0;
    for i in range(len(x)):
        result1+=x[i]*y[i]   #sum(X*Y)
        result2+=x[i]**2     #sum(X*X)
        result3+=y[i]**2     #sum(Y*Y)
    return result1/((result2*result3)**0.5)

# 高斯Q函数的逆函数
def Q_inv(x):
    return np.sqrt(2) * erfcinv(2 * x)

# 计算V(γk(t))
def V(gamma):
    # 根据具体公式定义V(γk(t))
    return gamma  # Placeholder, replace with actual formula if necessary

# 目标函数
def objective_function(w, num_points, A1, A2, h_k, sigma2, M, D):
    # 计算gamma_k(t)
    term_tol=0
    db_tol=0
    for i in range(num_points):
        gamma_k_t = np.abs(A1[i] @ w)**2 / (i * np.abs(A1[i] @ w)**2 + sigma2)
        
        # 计算目标函数
        term1 = np.log(2) * np.sqrt(M / (1-(1+gamma_k_t)**(-2)))
        term2 = np.log2(1 + gamma_k_t) - D / M
        term_tol=term1 * term2
        db_tol+=math.log10(max(1-0.5 * erfc(term_tol / np.sqrt(2)),10**(-20)))
    
    return (db_tol/num_points)

import random

class Particle:
    def __init__(self, dimension):
        self.position = np.random.rand(dimension)
        self.velocity = np.random.rand(dimension) - 0.5
        self.best_position = self.position.copy()
        self.best_score = -np.inf

    def update_velocity(self, global_best_position, inertia_weight=0.5, cognitive_coeff=2, social_coeff=2):
        cognitive_component = cognitive_coeff * random.random() * (self.best_position - self.position)
        social_component = social_coeff * random.random() * (global_best_position - self.position)
        self.velocity = inertia_weight * self.velocity + cognitive_component + social_component

    def update_position(self):
        self.position += self.velocity
        self.position = np.clip(self.position, 0, 1)  # Ensure within bounds

class PSO:
    def __init__(self, objective_function, dimension, swarm_size=30, iterations=100):
        self.objective_function = objective_function
        self.dimension = dimension
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.swarm = [Particle(dimension) for _ in range(swarm_size)]
        self.global_best_position = np.random.rand(dimension)
        self.global_best_score = -np.inf

    def optimize(self, *args):
        for iteration in range(self.iterations):
            for particle in self.swarm:
                # print(particle.position)
                score = self.objective_function(particle.position, *args)
                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()

                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle.position.copy()

            for particle in self.swarm:
                particle.update_velocity(self.global_best_position)
                particle.update_position()

            # print(f"Iteration {iteration + 1}/{self.iterations}, Best Score: {self.global_best_score}")

        return self.global_best_position, self.global_best_score

for k in range(3):
    import matplotlib.pyplot as plt
    exec('''path_file = 'C://Users/Administrator.DESKTOP-NLH290A/Desktop/2302_code/response_code/fig3/A算法/robot_path_{}.txt' '''.format(k+1))
    with open(path_file, 'r') as f:
        lines = f.readlines()

    # 解析路径数据
    path = [(int(line.split(',')[0]), int(line.split(',')[1])) for line in lines]

    # 提取 x 和 y 坐标
    exec('''x_coords_{} = [point[0] for point in path]'''.format(k+1))
    exec('''y_coords_{} = [point[1] for point in path]'''.format(k+1))

def calculate_interference(coords, labels, cluster_num):
    interference = 0
    cluster_points = coords[labels == cluster_num]
    num_points = len(cluster_points)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            interference += np.linalg.norm(cluster_points[i] - cluster_points[j])
    return interference

power_array=[math.sqrt(0.02), math.sqrt(0.04), math.sqrt(0.06), math.sqrt(0.08),math.sqrt(0.1)]
for power_j in range(len(power_array)):
    w = np.random.rand(3)
    for t in range(77):
        # print('time+++++++++')
        # for cl in range(1):    
        # for cl in range(2):
        # # 假设三个机器人的坐标
        robot_coords = np.array([[x_coords_1[t], y_coords_1[t]], [x_coords_2[t], y_coords_2[t]]])
        
        # 假设三个固定用户的坐标
        user_coords = np.array([[17, 50], [50, 50], [84, 50]])
        
        # 合并所有坐标
        all_coords = np.vstack((robot_coords, user_coords))
        
        # 定义 KMeans 模型
        kmeans = KMeans(n_clusters=3)
        
        # 使用所有坐标进行聚类
        kmeans.fit(all_coords)
        
        # 获取聚类结果
        labels = kmeans.labels_
        
        # 获取聚类中心
        cluster_centers = kmeans.cluster_centers_
        
        # 打印结果
        # print("Labels:", labels)
        # print("Cluster centers:", cluster_centers)
        
        # 绘制聚类结果
        # plt.scatter(all_coords[:, 0], all_coords[:, 1], c=labels, cmap='viridis')
        # plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='red', marker='x')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('KMeans Clustering of Robots and Users')
        # plt.show()
        
        robot_powers = np.array([10, 20, 30, 10, 20])
        theta_1=cosVector([1,0,0],[all_coords[0][0]-50,all_coords[0][1]-100, 1-2])
        aLP_1=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_1)), math.exp(1)**(-2*1j*(math.pi*theta_1))])#
        b_1_AP_LOS=math.sqrt(PL_AP[int(all_coords[0][0]), int(all_coords[0][1])])
        h_1=b_1_AP_LOS*aLP_1
        interference_1=10**(-9)
        theta_2=cosVector([1,0,0],[all_coords[1][0]-50,all_coords[1][1]-100, 1-2])
        aLP_2=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_2)), math.exp(1)**(-2*1j*(math.pi*theta_2))])#
        b_2_AP_LOS=math.sqrt(PL_AP[int(all_coords[1][0]), int(all_coords[1][1])])
        h_2=b_2_AP_LOS*aLP_2
        interference_2=10**(-9)
        theta_3=cosVector([1,0,0],[all_coords[2][0]-50,all_coords[2][1]-100, 1-2])
        aLP_3=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_3)), math.exp(1)**(-2*1j*(math.pi*theta_3))])#
        b_3_AP_LOS=math.sqrt(PL_AP[int(all_coords[2][0]), int(all_coords[2][1])])
        h_3=b_3_AP_LOS*aLP_3
        interference_3=10**(-9)
        theta_4=cosVector([1,0,0],[all_coords[3][0]-50,all_coords[3][1]-100, 1-2])
        a_4=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_4)), math.exp(1)**(-2*1j*(math.pi*theta_4))])#
        b_4_AP_LOS=math.sqrt(PL_AP[int(all_coords[3][0]), int(all_coords[3][1])])
        h_4=b_4_AP_LOS*a_4
        interference_4=10**(-9)
        theta_5=cosVector([1,0,0],[all_coords[4][0]-50,all_coords[4][1]-100, 1-2])
        a_5=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_5)), math.exp(1)**(-2*1j*(math.pi*theta_5))])#
        b_5_AP_LOS=math.sqrt(PL_AP[int(all_coords[4][0]), int(all_coords[4][1])])
        h_5=b_5_AP_LOS*a_5
        interference_5=10**(-9)
        # theta_6=cosVector([1,0,0],[all_coords[5][0]-50,all_coords[5][1]-100, 1-2])
        # a_6=np.array([1, math.exp(1)**(-1*1j*(math.pi*theta_6)), math.exp(1)**(-2*1j*(math.pi*theta_6))])#
        # b_6_AP_LOS=math.sqrt(PL_AP[int(all_coords[5][0]), int(all_coords[5][1])])
        # h_6=b_6_AP_LOS*a_6
        # interference_6=10**(-9)
        H_array=[]
        H_array.append(h_1)
        H_array.append(h_2)
        H_array.append(h_3)
        H_array.append(h_4)
        H_array.append(h_5)
        # H_array.append(h_6)
        H_array=np.array(H_array)
        
        for r in range(len(robot_powers)):
            robot_powers[r]=np.abs(H_array[r] @ w*power_array[power_j]/math.sqrt(3))**2
        
        #
        num_clusters = 3
        interference_list = []
        
        gamma_avg=0
        gamma_array=np.zeros(3)
        
        for cluster_num in range(num_clusters):
            # interference = calculate_interference(all_coords, labels, cluster_num)
            w = np.random.rand(3)
            previous_A = w
            counter = 0
            for episode in range(500):
                interference = 0
                cluster_points = all_coords[labels == cluster_num]
                cluster_powers = robot_powers[labels == cluster_num]
                
                # 根据功率对簇内的用户进行排序，功率大的优先
                sorted_indices = np.argsort(-cluster_powers)
                sorted_points = cluster_points[sorted_indices]
                # sorted_powers = cluster_powers[sorted_indices]
                cluster_hk = H_array[labels == cluster_num]
                # cluster_wk = wk[labels == cluster_num]
                
                # for i_j in range(len(sorted_points)):
                #     sorted_powers
                
                num_points = len(sorted_points)
                # for i in range(num_points):
                interference = 0
                A1 = cluster_hk * power_array[power_j]/math.sqrt(3)
                A2 = 1
                h_k = H_array  # 示例信道向量
                # K_c = [0, 1, 2, 3]  # 示例其他用户索引
                sigma2 = 10**(-9)
                M = 50
                D = 100
                
                # 粒子群优化
                pso = PSO(objective_function, dimension=3, swarm_size=30, iterations=100)
                w, best_score = pso.optimize(num_points, A1, A2, h_k, sigma2, M, D)
                    
                    
                    # sinr = np.abs(A1 @ w)**2 / (i * np.abs(A1 @ w)**2 + sigma2)
    
                    # print("最佳位置（波束成形向量）:", best_position)
                    # if episode==99:
                if w.all() == previous_A.all():
                    counter += 1
                else:
                    counter = 0 
                    
                previous_A = w

                # 如果A持续十轮不变，则跳出循环
                if counter >= 10:
                    # print(cluster_num, 'sucess!!!!!!!!!!!!')
                    break
            
                
            
            for i in range(num_points):
                gamma_k_t = np.abs(A1[i] @ w)**2 / (i * np.abs(A1[i] @ w)**2 + sigma2)
                gamma_avg+=gamma_k_t
            gamma_k_t=gamma_k_t#/num_points
            
            gamma_array[cluster_num] = best_score
            # print("最佳得分:", best_score)#math.log10(max(1-0.5 * erfc(best_score / np.sqrt(2)),10**(-20))))
        
        filename='DB_NOMA_new_'+str(power_j)+'.txt'
        with open (filename, 'a') as fileobject:
            fileobject.write(str((gamma_array[0]+gamma_array[1]+gamma_array[2])/6)+'\n')

        
                # for j in range(0, i):
                #     # 计算干扰，假设干扰与距离成反比
                #     interference += np.abs(H_array[i] @ w*power_array[power_j]/math.sqrt(3))**2
                
                # sinr += sorted_powers[i] / interference  # 假设干扰公式
            # return interference
            # interference_list.append(interference)
        
        # 打印每个簇的干扰
        # for i, interference in enumerate(interference_list):
        #     print(f"Cluster {i} interference: {interference}")
        
        
            # 示例参数
            