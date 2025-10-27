from collections import Counter

import math
import random

from matplotlib import pyplot as plt
from sklearn import neighbors, svm,naive_bayes
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import dcor
from torch.distributions import MultivariateNormal, kl_divergence


import torch
import torch.distributed as dist

from read_mat import read_mat, np_mat
from sklearn.model_selection import KFold

def entropy(data):
    """计算信息熵"""
    counts = Counter(data)
    probs = [count / len(data) for count in counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)

def conditional_entropy(x, y):
    """计算条件熵 H(Y|X)"""
    df = pd.DataFrame({'x': x, 'y': y})
    joint_counts = df.groupby(['x', 'y']).size().reset_index(name='count')
    x_counts = df['x'].value_counts(normalize=True).to_dict()
    total_entropy = 0
    for _, row in joint_counts.iterrows():
        p_xy = row['count'] / len(df)
        p_x = x_counts[row['x']]
        total_entropy -= p_xy * np.log2(p_xy / p_x)
    return total_entropy

def mutual_information(x, y):
    """计算互信息 I(X;Y)"""
    return entropy(y) - conditional_entropy(x, y)

def symmetrical_uncertainty(f, c):
    """计算对称不确定性 SU(f, c)"""
    mi = mutual_information(f, c)
    h_f = entropy(f)
    h_c = entropy(c)
    if h_f + h_c == 0:
        return 0
    return 2 * (mi / (h_f + h_c))



def encode(y_i):
    # print(y_i)
    # print('cao')
    # print(1 / (1 + math.exp(-y_i)))
    #1 / (1 + math.exp(-y_i)) > 0.6
    if y_i > 0.5 :
        return 1
    else:
        return 0


def jFitnessFunction(feat, label, X):
    # Default of [alpha; beta]
    ws = [1, 0]  # [1, 0]


    # Check if any feature exist
    if np.sum(X > 0) == 0:
        return 1
    else:
        # Error rate
        error = jwrapper_KNN(feat[:, X > 0], label)
        # Number of selected features
        num_feat = np.sum(X > 0)
        # Total number of features
        max_feat = len(X)
        # Set alpha & beta
        alpha = ws[0]
        beta = ws[1]
        # Cost function
        cost = alpha * error + beta * (num_feat / max_feat)
        # cost = error
        return cost, error, (num_feat / max_feat)

def jwrapper_KNN(sFeat, label):
    """测试集准确率计算"""
    k = 1
    #移除 KFold
    kf = KFold(n_splits=5)
    Acc = []
    for train_index, test_index in kf.split(range(len(sFeat))):
        xtrain, xvalid = sFeat[train_index], sFeat[test_index]
        ytrain, yvalid = label[train_index], label[test_index]
        #Training model
        My_Model = KNeighborsClassifier(n_neighbors=k)
        My_Model.fit(xtrain, ytrain)
        # Prediction
        pred = My_Model.predict(xvalid)
        # Accuracy
        Acc.append(np.sum(pred == yvalid) / len(yvalid))

        # Error rate


    error = 1 - np.mean(Acc)
    return error

def cal_dis(x0, y0, p1, p2):
    return abs(p1 * x0 - y0 + p2) / np.sqrt(p1 ** 2 + 1)


def sort_array_by_scores(array, scores):
  """
  根据分数数组对二维 NumPy 数组的行进行排序。

  Args:
    array: 要排序的形状为 (n, m) 的 NumPy 数组。
    scores: 用于排序的形状为 (n,) 的分数数组。

  Returns:
    一个新的 NumPy 数组，其行已根据分数数组排序。
  """

  # 1. 获取排序后的索引
  sorted_indices_f = np.argsort(scores)

  # 2. 使用排序后的索引对数组进行排序
  sorted_array = array[sorted_indices_f]

  return sorted_array


#-------SU--------
X1, y = read_mat()
print(X1)
print(y)
y1 = y.copy()
print(y1.shape)
np_X, np_y = np_mat()
print(np_X.shape)
print(np_y.shape)
ca_labels = X1.columns
knn_1 = neighbors.KNeighborsClassifier(n_neighbors=1)
sc_X = StandardScaler(copy=True)
print("dhjsidfkhs")
print(X1)
print(y)
# 离散化
# discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
# X_discrete = discretizer.fit_transform(X1)
# X_discrete = pd.DataFrame(X_discrete, columns=X1.columns)
X_discrete = X1.copy()

# --- 计算 SU ---
su_values = []
for feature in X_discrete.columns:
    su = symmetrical_uncertainty(X_discrete[feature], np_y.copy().flatten())
    su_values.append(su)


su_values = np.array(su_values)  # 转换为 NumPy 数组
# 获取排序后的索引
sorted_indices = np.argsort(su_values)[::-1]  # 降序排列

y1 = np.arange(0, X1.shape[1])
# 对X进行排序，同时按照X的改变对y进行改变
sorted_X = su_values[sorted_indices] #值
sorted_y = y1[sorted_indices] #特征
print(sorted_y)
def Knee_Point_Selection_Strategy(W):
    numFeature = W.shape[0]
    maxPoint = np.array([1, W[0]])
    minPoint = np.array([numFeature, W[numFeature - 1]])

    # Fit a line between the max and min points
    slope, intercept = np.polyfit([maxPoint[0], minPoint[0]], [maxPoint[1], minPoint[1]], 1)

    max_Dis = 0
    kneePoint = 0

    # Iterate through each feature to find the knee point
    for i in range(1, numFeature + 1):
        dis = cal_dis(i, W[i - 1], slope, intercept)
        if dis > max_Dis:
            max_Dis = dis
            kneePoint = i
    return kneePoint


keen_Point = Knee_Point_Selection_Strategy(sorted_X)
pro_features = []
p = 0
rem_features = []
r = 0
for i in range(0, max(sorted_y)):
    if i <= keen_Point:
        pro_features.append(sorted_y[i])
        p = p + sorted_X[i]
    else:
        rem_features.append(sorted_y[i])
        r = r + sorted_X[i]
print(pro_features)
print(rem_features)
W_p = p/(r+p)
print(W_p)

p1 = np.zeros(sorted_X.shape[0])
for i in range(0, len(pro_features)):
    if W_p > np.random.rand():
        p1[pro_features[i]] = 1

for i in range(0, len(rem_features)):
    if (1-W_p) > np.random.rand():
        p1[rem_features[i]] = 1


def create_populations(n): #构造子任务
    population = np.zeros((n, sorted_X.shape[0]))
    score_pop = np.zeros(n)
    score_pop[-1] = 0
    rev = int(len(sorted_X)*(1)/8)
    #构造两个互补的辅助任务
    for j in range(len(pro_features)):
        if W_p >= np.random.random():
            if sum(population[0]) - rev == 0:
                break
            population[0][pro_features[j]] = 1
            score_pop[0] += su_values[pro_features[j]]

    for j in range(len(rem_features)):
        if (1 -W_p) >= np.random.random():
            if sum(population[0]) - rev == 0:
                break
            population[0][rem_features[j]] = 1
            score_pop[0] += su_values[rem_features[j]]
    rev1 = rev - sum(population[0])
    for i in range(int(rev1)):
        if population[0][sorted_y[i]] == 0:
            population[0][sorted_y[i]] = 1
            score_pop[0] += su_values[sorted_y[i]]

    for j in range(len(rem_features)):
        if (1 -W_p) >= np.random.random():
            if sum(population[1]) - rev == 0:
                break
            population[1][rem_features[j]] = 1
            score_pop[1] += su_values[rem_features[j]]
    for j in range(len(pro_features)):
        if W_p >= np.random.random():
            if sum(population[1]) - rev == 0:
                break
            population[1][pro_features[j]] = 1
            score_pop[1] += su_values[pro_features[j]]
    rev2 = rev - sum(population[1])
    for i in range(int(rev2)):
        if population[1][sorted_y[i]] == 0:
            population[1][sorted_y[i]] = 1
            score_pop[1] += su_values[sorted_y[i]]



    for pro in range(0, int(len(sorted_X)*(1)/8)):
        population[-1][sorted_y[pro]] = 1
    score_pop[-1] = sum(sorted_X[:int(len(sorted_X)*(1)/8)])
    return population, score_pop


def survive(X1,X2):
    p = X1[-1]
    print(p)
    count = 0
    for i in range(len(X2)):
        if X2[i] <= p:
            count += 1
        else:
            break

    return count/len(X2)
k = 3
all_plt = []
all_ac = []
all_DR = []

for num in range(5):
    pop, SU = create_populations(k)
    N = X1.shape[1]
    stopeval = 2000
    counteval = 0
    # print(pop)
    mean_plot = np.zeros(20)
    #-------- parameter setting -------#
    sigma = np.zeros(k).astype(np.float64)
    alpha = np.zeros(k).astype(np.float64)
    adjusteGen = np.zeros(k)
    dim = np.zeros(k)
    for i in range(k):
        dim[i] = sum(pop[i])
    max_dim = len(sorted_y)
    mean0 = np.zeros((k, max_dim)).astype(np.float64)
    mean = np.zeros((k, max_dim)).astype(np.float64)
    lambda_list = []
    lambda_standard = min((N // 10), 200)
    print(lambda_standard)
    for i in range(k):
        lambda_list.append((lambda_standard * (SU[i] / sum(SU))).astype(int))
    print(lambda_list)
    mu_list = [lambda_val // 2 for lambda_val in lambda_list]
    M_ij = np.zeros((k, k)).astype(np.float64)
    M_ij_temp = np.zeros((k, k)).astype(np.float64)
    d0 = np.zeros(k).astype(np.float64)
    D = np.zeros((k, k)).astype(np.float64)
    D0 = np.zeros((k, k)).astype(np.float64)
    p = np.zeros((k, max_dim))
    mul = []
    sigmal = []

    for i in range(k):
        sigma[i] = 0.1
        alpha[i] = 0.1
        adjusteGen[i] = 1000
        matrix = np.random.uniform(low=0, high=1, size=(max_dim, lambda_list[i]))
        mean[i] = np.mean(matrix, axis=1)
        for ii in range(max_dim):
            if pop[i][ii] == 0:
                mean[i][ii] = 0
        mu1 = torch.Tensor(mean[i])
        sigma1 = torch.eye(max_dim)
        mul.append(mu1)
        sigmal.append(sigma1)


    for i in range(k):
        for j in range(i+1,k):
            dist1 = MultivariateNormal(mul[j], sigmal[j])
            dist2 = MultivariateNormal(mul[i], sigmal[i])
            D[i][j] = kl_divergence(dist1, dist2)
            D[j][i] = D[i][j]
        d0[i] = sum(D[i])/(k-1)

    print(D)
    fit_plot = []
    num_plt = []
    DR_plot = []
    acc_plot = []
    geng = 0
    n_gen = 1
    while counteval < stopeval:
        normalize_fit_list = []
        x2 = []
        num_evolutions = 0
        mean0 = mean.copy()
        part1 = np.zeros((k, max_dim))
        part2 = np.zeros((k, max_dim))
        geng_fit = np.zeros(k)
        FT_fit = np.zeros(k)
        DR_fit = np.zeros(k)
        M = np.zeros((k, k))
        mean_fit = np.zeros(k)
        sort_glob_samples = []
        sort_glob_fit = []
        sort_glob_samples0 = []
        # -----Sample and Evaluation --------#
        for i in range(k):
            normalize_fit = np.zeros(lambda_list[i]).astype(np.float64)
            kid_f = np.zeros(lambda_list[i])
            kidf_fit = np.zeros(lambda_list[i])
            FT = np.zeros(lambda_list[i])
            DR = np.zeros(lambda_list[i])
            kid_feature = []
            kid_u = np.zeros(lambda_list[i])
            dim_i = max_dim
            samples0 = np.random.multivariate_normal(np.zeros(dim_i), np.eye(dim_i), lambda_list[i]) # sample
            samples = mean[i][:dim_i] + sigma[i] * samples0

            for ii in range(max_dim):
                if pop[i][ii] == 0:
                    mean[i][ii] = 0
            if i == 0:
                for j in range(lambda_list[i]):
                    counteval += 1
                    gen = np.zeros(max_dim)
                    for ii in range(max_dim):
                        if pop[i][ii] == 1:
                            a = encode(samples[j][ii])
                            gen[ii] = a
                        else:
                            samples0[j][ii] = 0
                            samples[j][ii] = 0
                    kid_feature.append(gen.copy())
                    raw_data_copy = np_X.copy()
                    kid_f[j], FT[j], DR[j] = jFitnessFunction(raw_data_copy,np_y.copy().ravel(),gen.copy())


            else:
                for j in range(lambda_list[i]):
                    counteval += 1
                    gen = np.zeros(max_dim)
                    for ii in range(max_dim):
                        if pop[i][ii] == 1:
                            a = encode(samples[j][ii])
                            gen[ii] = a
                        else:
                            samples0[j][ii] = 0
                            samples[j][ii] = 0

                    kid_feature.append(gen.copy())
                    raw_data_copy = np_X.copy()
                    kid_f[j], FT[j], DR[j] = jFitnessFunction(raw_data_copy,np_y.copy().ravel(),gen.copy())
            mu1 = torch.Tensor(mean[i])
            sigma1 = torch.eye(dim_i)
            mul[i] = mu1
            sigmal[i] = (sigma1)
            min_index = np.argmin(kid_f)
            geng_fit[i] = 1 - min(kid_f)
            FT_fit[i] = 1 - FT[min_index]
            DR_fit[i] = 1 - DR[min_index]
            print(f"第{i}个任务的最好点是{FT_fit[i]}")
            print(f"{geng_fit[i]}")
            print(f"DR Rate {DR_fit[i]}")
            best_gen = kid_feature[min_index]
            sort_glob_samples0.append(sort_array_by_scores(samples0, kid_f))
            sort_glob_samples.append(sort_array_by_scores(samples, kid_f))
            sort_glob_fit.append(np.sort(kid_f))
            mean_fit[i] = np.mean(kid_f)
            num_evolutions += 100
            x2.append(num_evolutions)
            if np.std(kid_f) <= 1e-6:
                std = 1e-6
                continue
            else:
                std = np.std(kid_f)

            for j in range(lambda_list[i]):
                normalize_fit[j] = (kid_f[j] - np.mean(kid_f))/std
                nor_2 = normalize_fit[j]*samples0[j]
                part2[i] += nor_2
        M1 = np.zeros((k, k, max_dim, max_dim)).astype(np.float64)

        #---------- Knowledge Transfer----------#
        if geng%n_gen==0 and geng > 0:
            for i in range(k):
                m = np.zeros(k).astype(np.float64) + 1e13
                for t in range(k):
                    if i==t:
                        continue
                    dist1 = MultivariateNormal(mul[t], sigmal[t])
                    dist2 = MultivariateNormal(mul[i], sigmal[i])
                    kl = kl_divergence(dist1, dist2)
                    m[t] = ((mean_fit[t]) + np.sqrt(kl)/(2*N))
                    print(m[t])
                m_uti = np.zeros(k).astype(np.float64)
                index_max = np.argmin(m) # Finding the best match for a secondary task
                print(f"pipei{index_max}")
            #DAE Mapping
                map_size = min(mu_list[i], mu_list[index_max])
                map_fit_s = np.zeros((k, map_size))
                samp_q = sort_glob_samples[i][:map_size].T
                samp_p = sort_glob_samples[index_max][:map_size].T
                q_pt = samp_q @ samp_p.T
                p_pt = samp_p @ samp_p.T
                Mnn= q_pt @ np.linalg.pinv(p_pt)
                map_mean = Mnn@(mean[index_max].T)
                mean_tempt = mean.copy()
                mean_tempt[index_max] = map_mean
                m_uti[index_max] = min(survive(sort_glob_fit[i][:map_size], sort_glob_fit[index_max][:map_size]),0.1)#存活率
                m_uti[i] = 1 - m_uti[index_max]
                x_tempt = mean0[i].copy()
                for j in range(k):
                    part1[i] += m_uti[j] * mean_tempt[j]
                if geng % 100 == 0 and geng != 0:
                    sigma[i] = min(np.mean(abs(mean0[i] - x_tempt)) + 0.001, 1)
                    print("sigma")
                    print(sigma[i])
                    alpha[i] = sigma[i] ** 2

                mean[i] = part1[i] - alpha[i]*part2[i]/(lambda_list[i]*sigma[i])

                print(m_uti)
        else:
            for i in range(k):
                x_tempt = mean0[i].copy()
                mean[i] = mean[i] - alpha[i]*part2[i]/(lambda_list[i]*sigma[i])

                if geng%100==0 and geng!=0:
                    # print(mean0[i])
                    # print(x_tempt)
                    sigma[i] = min(np.mean(abs(mean0[i] - x_tempt))+0.001,1)
                    print("sigma")
                    print(sigma[i])
                    alpha[i] = sigma[i]**2
        #mean[i] = min(1,max(0,sum(mean[i])))

        print(max(geng_fit))
        cost_index = np.argmax(geng_fit)
        DR_plot.append(DR_fit[cost_index])
        acc_plot.append(FT_fit[cost_index])

        #print(mul)

        fit_plot.append(max(geng_fit))
        #mean_plot[geng] += min(geng_fit)/11
        geng += 1
        num_plt.append(counteval)
        D0 = D.copy()

    # print(fit_plot)
    # geng_list = np.arange(0, 20)
    # plt.plot(geng_list, fit_plot, marker='o', linestyle='--', color='r', label='NMTES')
    # plt.title('Spambase')
    # plt.xlabel('num_evaluation')
    # plt.ylabel('accuracy')
    # # 添加图例
    # plt.legend()
    # plt.grid()  # 可选: 添加网格
    # plt.show()\
    print(acc_plot)
    print(DR_plot)
    all_plt.append(fit_plot)
    all_ac.append(acc_plot)
    all_DR.append(DR_plot)

# print(1 - mean_plot)
print(all_plt)
print(all_ac)
print(all_DR)