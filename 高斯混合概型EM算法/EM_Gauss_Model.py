# -*- coding:utf-8 -*-
#############################################
#
#
#
#
#
#
#
#
#############################################
from sklearn.datasets import make_blobs
from matplotlib import pyplot
from sklearn.cluster import KMeans
import numpy as np
import pickle
import random
# 导入数据
data_num = 400
features_num = 2
target_num  = 10

Data,Target=make_blobs(n_samples=data_num,n_features=features_num,centers=target_num)

#存储和读取随机参数
pickle.dump(Data,open('.\\Data.txt', 'wb') ) 
pickle.dump(Target,open('.\\Target.txt', 'wb') ) 
Data = pickle.load(open('.\\Data.txt', 'rb') ) 
Target = pickle.load(open('.\\Target.txt', 'rb') ) 



# # 打印数据和标签的维数
# print(Data.shape)
# print(Target.shape)

### EM算法 高斯混合概形
# 初始化模型参数
Y = np.array(Data)
Z = np.ones(Target.shape) # 隐变量标签
P = np.ones(target_num) 
P = P / target_num # 类别概率
U = np.zeros([target_num,features_num])   # 高斯分布均值参数
Target_arr = np.array(Target)
Sigma = np.zeros([target_num,features_num,features_num])   # 高斯分布协方差参数
stop_deta = 0.0001

# # 数据归一化
# y_mean  = np.mean(Y,0)
# y_max = np.max(Y,0)
# y_min = np.min(Y,0)
# for i in range(data_num):
#     yi = Y[i]
#     yi_norm = yi - y_mean
#     yi_norm = yi_norm / (y_max - y_min)
#     Y[i] = yi_norm




# # 在2D图中绘制样本，每个样本颜色不同
# pyplot.subplot(121)
# pyplot.title('skLearn生成高斯混合样本',fontproperties='SimHei')
# pyplot.scatter(Data[:,0],Data[:,1],c=Target);

# pyplot.show()

#########################
# k均值聚类求解初始均值
#########################
# 聚类簇数M
K = target_num
# 初始均值向量选取
u_k = np.zeros([target_num,features_num]) 
Target_k = Target.copy()


# 选取相互最远距离进行初始化
for k in range(target_num):
    d_max = 0
    max_xi = []
    if(k == 0):
        # 第一个中心点随机选取
        u_k[k] = Y[np.random.randint(0,data_num,size=1)]
        continue
    for xi in Y:
        d_sum = 0
        for uj in u_k:
            if np.all(xi==uj):
                d_sum = 0
                break
            d = np.linalg.norm(xi - uj)
            d_sum = d_sum + d
        if d_sum>d_max:
            d_max = d_sum
            max_xi = xi
    u_k[k] = max_xi


# # 随机选取初始点
# u_k = Y[np.random.randint(0,data_num,target_num)]

u_pre = u_k.copy()
C_k = []
while True:
    C_k = []
    for i in range(K):
        C_k.append([])
    i = 0;
    for xi in Y:
        dmin = np.inf
        min_j = 0
        # 选择最近的距离中心
        for j  in range(K):
            u_j = u_k[j]
            dij = np.linalg.norm(xi - u_j)
            if dij <= dmin:
                dmin = dij.copy()
                min_j = j
        # 将xi分给最近的距离中心
        C_k[min_j].append(xi)
        Target_k[i] = min_j
        i = i + 1
    # 计算分配后的中心
    change_count =0
    for i in range(K):
        lenCi= len(C_k[i])
        ui = np.zeros([features_num])
        for xi in C_k[i]:
            ui = ui + xi
        if(lenCi !=0):
            ui = ui/lenCi;
            change_flag = 0
            if np.any(ui != u_k[i]) :
                change_flag = 1
            u_k[i] = ui
            if change_flag == 1:
                change_count = change_count + 1
        else:
            print("Error K-means")
            break
    if change_count == 0:
        break

u_m_kmean  =u_k.copy()

# 调用库进行k均值聚类
kmeans = KMeans(n_clusters=target_num).fit(Y)
y_pred =kmeans.fit_predict(Y)
u_kmean =kmeans.cluster_centers_;


# 随机选取初始点
u_random = Y[np.random.randint(0,data_num,target_num)]

# #选择使用的初始点来初始化
# U = u_m_kmean.copy() # 自行编写的k均值聚类
U = u_kmean.copy() # 使用sklearn的库的k均值聚类
# U = u_random.copy() #  随机生成的初始点

# 初始化协方差
for j in range(target_num):
    Sigma[j] = 2*features_num*np.eye(features_num)


# 计算高斯混合分布的联合密度
def cal_gauss_f(U_j,Sigma_j,Y):
    # 计算高斯密度
    Y_U_j = (Y-U_j).reshape([features_num,1])  # 维数 1xfeatures_num
    f_Y_j = np.dot(Y_U_j.T,np.linalg.inv(Sigma_j)) # 维数 1xfeatures_num
    f_Y_j_up = -np.dot(f_Y_j,Y_U_j)/2 # 维数 1x1
    f_Y_j_full  = np.exp(f_Y_j_up) / ( np.power(2*np.pi,features_num/2)  * np.sqrt(np.abs(np.linalg.det(Sigma_j)))) # 求得高斯密度
    ### 防止密度溢出或者为0
    if np.isnan(f_Y_j_full):
        f_Y_j_full = 100000
    if f_Y_j_full == 0:
        f_Y_j_full = 0.0001
    if f_Y_j_full<0:
        print("Error")
        f_Y_j_full = 0.00000000000001
    return f_Y_j_full

Tji = np.zeros([target_num,data_num])
fji =  np.zeros([target_num,data_num])



# 计算Tji
def cal_T_i_j():
    global Tji
    for j in range(target_num):
        for i in range(data_num):
            fji[j][i] = cal_gauss_f(U[j],Sigma[j],Y[i])
    P_f = np.dot(P.reshape([1,target_num]),fji)
    P_f = P_f.flatten()
    for j in range(target_num):
        for i in range(data_num):
            # 防止微小量的影响
            Tji[j][i] = P[j] * fji[j][i]/P_f[i]
def cal_P_i_j():
    global P
    Tj_sum = np.sum(Tji,axis=1)
    P = Tj_sum/data_num
    # print(P)

def cal_theta_i_j():
    U_new = np.zeros([target_num,features_num]) 
    Sigma_new =np.zeros([target_num,features_num,features_num]) ;
    # 更新均值
    Tj_sum = np.sum(Tji,axis=1)
    T_Q = np.array(Tji);
    # 更新均值
    for j in range(target_num):
        U_j = np.zeros([1,features_num])
        for i in range(data_num):
            U_j = U_j + Tji[j][i]*Y[i]
        U_j = U_j/Tj_sum[j]
        U_new[j] = U_j
    # 更新协方差
    for j in range(target_num):
        Sigma_j = np.zeros([features_num,features_num])
        for i in range(data_num):
            Y_U_j = Y[i] -U_new[j]
            Y_U_j = Y_U_j.reshape([features_num,1])
            Sigma_j = Sigma_j + Tji[j][i]*np.dot(Y_U_j,Y_U_j.T)
        Sigma_j = Sigma_j/Tj_sum[j]
        Sigma_new[j] = Sigma_j
    #print(U_new-U)
    return U_new,Sigma_new

stop_pre = 0
stop_count = 0
max_count = 100


#pyplot.ion()
while(True):
    cal_T_i_j()
    cal_P_i_j()
    U_new,Sigma_new = cal_theta_i_j()
    stop_q = np.sqrt(np.sum(np.square(U-U_new)) + np.sum(np.square(Sigma - Sigma_new)))
    print("参数变化：",end="")
    # print(np.sum(np.square(U-U_new)))
    # print(np.sum(np.square(Sigma - Sigma_new)))
    print(stop_q)

    # 参数不变化退出
    if(stop_q <stop_deta):
        print("Normal Break")
        break
    # 迭代超时退出
    if(stop_count > max_count):
        print("Un Normal Break")
        break 
    U = U_new.copy()
    Sigma = Sigma_new.copy()
    Target_predict  = np.argmax(Tji,0)
    stop_pre = stop_q
    stop_count = stop_count + 1

    # 显示聚类迭代过程
    pyplot.clf()
    pyplot.title('EM算法迭代过程',fontproperties='SimHei')
    pyplot.scatter(Y[:,0],Y[:,1],c= Target_predict,marker='.')
    pyplot.scatter(U[:,0],U[:,1],c='r',marker='o');
    pyplot.pause(0.01)

cal_T_i_j()
cal_P_i_j()
Target_predict  = np.argmax(Tji,0)


# k_trans = []
# for k in range(target_num):
#     index =np.argwhere(Target_arr ==  k)
#     index = index.reshape([index.size])
#     Y_k = Y[index]
#     u = np.mean(Y_k,0)
#     min_j = 0
#     min_du = 9999999999
#     for j in range(target_num):
#         du = u - U[j]
#         du = np.sum(np.square(du))
#         du = np.sqrt(du)
#         if du <min_du:
#             if np.any(np.array(k_trans) == j):
#                 pass
#             else:
#                 min_du = du
#                 min_j = j
#     k_trans.append(min_j)
# k_trans =np.array(k_trans)
# Target_predict = k_trans[Target_predict]

# 在2D图中绘制样本，每个样本颜色不同
pyplot.subplot(221)
pyplot.title('skLearn生成高斯混合样本',fontproperties='SimHei')
pyplot.scatter(Data[:,0],Data[:,1],c=Target,marker='.')


# 在2D图中绘制样本，每个样本颜色不同
pyplot.subplot(222)
pyplot.title('EM算法估计GMM分布结果',fontproperties='SimHei')
pyplot.scatter(Y[:,0],Y[:,1],c= Target_predict,marker='.')
pyplot.scatter(U[:,0],U[:,1],c='r',marker='o');


# K-均值聚类
pyplot.subplot(223)
pyplot.title('自编k均值聚类结果',fontproperties='SimHei')
pyplot.scatter(Y[:,0],Y[:,1],c=Target_k,marker='.')
pyplot.scatter(u_pre[:,0],u_pre[:,1],c='b',marker='o');
pyplot.scatter(u_m_kmean[:,0],u_m_kmean[:,1],c='r',marker='o');

pyplot.subplot(224)
pyplot.title('SkLearn运行k均值聚类结果',fontproperties='SimHei')
kmeans = KMeans(n_clusters=target_num).fit(Y)
y_pred =kmeans.fit_predict(Y)
u_kmean =kmeans.cluster_centers_;
pyplot.scatter(Y[:,0], Y[:,1],c=y_pred,marker='.')
pyplot.scatter(u_kmean[:,0],u_kmean[:,1],c= 'r')
pyplot.pause(7)



