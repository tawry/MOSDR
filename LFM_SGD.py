from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings("ignore")
import time
from functools import wraps

class LFM():
    def __init__(self, data, X_masked, K=40, lamda=0.01, alpha=0.000007, max_iter=250):
        '''
        Arguments
        - data: complete dataset
        - X: masked dataset
        - R (ndarray)   : sample-feature matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        '''
        self.data = data
        self.X_masked = X_masked
        self.lamda = lamda
        self.K = K
        self.alpha = alpha
        self.max_iter = max_iter

    def SGD_new(self):

        X_masked_imputzero = np.copy(self.X_masked)
        X_masked_imputzero[np.isnan(X_masked_imputzero)] = 0
        R_coo = coo_matrix(X_masked_imputzero)
        M, N = self.X_masked.shape
        self.P = np.random.rand(M, self.K)
        self.Q = np.random.rand(self.K, N)

        self.P = np.array(self.P, dtype=np.longdouble)
        self.Q = np.array(self.Q, dtype=np.longdouble)

        rmse1 = np.inf
        flag = 1

        for step in range(self.max_iter + 1):
            for ui in range(len(R_coo.data)):
                rui = R_coo.data[ui]
                u = R_coo.row[ui]
                i = R_coo.col[ui]
                if rui:
                    eui = (rui - np.dot(self.P[u, :], self.Q[:, i]))
                    self.P[u, :] = self.P[u, :] + self.alpha * 2 * (eui * self.Q[:, i] - self.lamda * self.P[u, :])
                    self.Q[:, i] = self.Q[:, i] + self.alpha * 2 * (eui * self.P[u, :] - self.lamda * self.Q[:, i])

            if not step % 5:
                rmse = self.error()

                if rmse > rmse1:
                    print("  times:\t" + str(step) + '\t\t' + str(rmse1))
                    flag = 0
                    break
                rmse1 = rmse
                self.alpha = 0.9 * self.alpha
        if flag:print("  times:\t" + str(step) + '\t\t' + str(rmse1))
        return


    def error(self):
        # ratings = R.data
        # rows = R.row
        # cols = R.col
        # t0 = time.time()

        e = 0
        times = 0
        abss = 0
        preR = self.P.dot(self.Q)
        #self.index = np.argwhere(np.isnan(self.X_masked))
        self.index = np.argwhere(self.X_masked == 0)
        for i, j in self.index:
            e = e + pow(self.data[i, j] - preR[i, j], 2)
            # abss = abss + np.abs(data.at[i, j] - preR[i, j])
            times += 1
        rmse = np.sqrt(e / times)
        #print(" this time RMSE: " + str(rmse))
        # t1 = time.time()
        # print("times: ",t1-t0)
        return rmse

    def replace_nan(self):
        """
        Replace np.nan of X with the corresponding value of X_hat
        """

        X_hat = self.P.dot(self.Q)
        X = np.copy(self.X_masked)
        for i, j in self.index:
            X[i, j] = X_hat[i, j]
        return X

def cal_indice_value(X, L, c_index):
    temp1 = np.sqrt(np.sum([(X[i, j] - L[i, j]) ** 2 for i, j in zip(c_index[0], c_index[1])]))
    temp2 = np.sqrt(np.sum([(X[i, j]) ** 2 for i, j in zip(c_index[0], c_index[1])]))
    return round(temp1 / temp2, 4)


def cal_indices(raw_data, L, data_index,indices):
    result = {}
    # 随机缺失部分数据
    missing_index = np.unravel_index(indices, raw_data.shape)       #将之前生成的 indices 数组（用于表示要在原始数据 raw_data 中设置为零的元素位置）转换成多维索引，以便找到这些位置在原始数据中的行和列位置。
    result['ERM'] = cal_indice_value(raw_data, L, missing_index)

    # 随机列部分数据
    missing_cols_index = ([], [])
    for i in range(raw_data.shape[0]):   #遍历了原始数据 raw_data 的行，后面可以改这里遍历缺失行索引missing_rows
        for j in sorted(missing_cols):  #遍历了已排序的缺失列索引 missing_cols
            missing_cols_index[0].append(i)
            missing_cols_index[1].append(j)
    result['ERC'] = cal_indice_value(raw_data, L, missing_cols_index)

    # 未缺失部分数据
    result['ERN'] = cal_indice_value(raw_data, L, data_index)

    # 未缺失部分数据
    nums = raw_data.shape[0] * raw_data.shape[1]
    result['RMSE'] = np.sqrt(np.sum((raw_data - L) ** 2) / nums)
    result['MAE'] = np.sum(np.abs(raw_data - L)) / nums
    return result

def data_recovery(init_data, to_cmp):
    '''第一步①先删除列缺失部分，对剩下的具有随机缺失的矩阵，进行中间(1)(2)(3)的数据恢复'''
    # 数据恢复
    K = 2  # 可以调整这些参数
    lamda = 0.01
    alpha = 0.0007
    max_iter = 250
    L = LFM(data.values, to_cmp, K, lamda, alpha, max_iter)

    return L

def main(init_data, ratio=0.2, missing_row=0, missing_col=0):
    # 设置随机矩阵
    X = (init_data.values).copy()           #将传入的 init_data 转换为 NumPy 数组，存在X变量中
    indices = np.random.choice(X.shape[1] * X.shape[0], replace=False, size=int(X.shape[1] * X.shape[0] * ratio))       #生成一个随机索引数组(indices)，用于表示要在矩阵X中随机置为零的元素位置。
    X[np.unravel_index(indices, X.shape)] = 0       #将矩阵 X 中的特定位置的元素设置为零，这些位置由之前生成的 indices 数组表示

    # 设置随机缺失的行列
    missing_rows = np.random.choice(X.shape[0], replace=False, size=missing_row)  #缺失行索引
    missing_cols = np.random.choice(X.shape[1], replace=False, size=missing_col)  #缺失列索引

    #创建一个名为 to_cmp 的矩阵，该矩阵是原始数据矩阵 X 的一个副本，并且在指定的行和列位置上将其元素设置为零。这个矩阵通常用于后续的数据比较或验证操作，以检查数据恢复的效果。
    to_cmp = X.copy()
    to_cmp[:,missing_cols] = 0      #模拟了缺失列的操作，将这些列的元素都置为零。
    to_cmp[missing_rows,:] = 0      #模拟了缺失行的操作，将这些行的元素都置为零。

    #X = np.delete(X, missing_rows, 0)  #删除指定缺失行
    #X = np.delete(X, missing_cols, 1)  #删除指定缺失列


    # 未缺失部分数据 index
    data_index = np.where(to_cmp != 0)

    '''数据恢复'''
    lfm= data_recovery(init_data, to_cmp)
    lfm.SGD_new()
    L = lfm.replace_nan()
    '''计算指标'''
    raw_data = init_data.values
    result = cal_indices(raw_data, L, data_index,indices)

    return result,L,to_cmp




if __name__ == '__main__':
    import shutil
    import os
    if os.path.exists('./data1/L_Matrix'):   #检查当前目录下是否存在名为 'L_Matrix' 的文件夹
        shutil.rmtree('./data1/L_Matrix/')   #如果存在会删除名为 'L_Matrix' 的文件夹及其内容
    os.mkdir('./data1/L_Matrix')             #创建一个新的名为 'L_Matrix' 的文件夹
    if os.path.exists('./data1/X_Matrix'):
        shutil.rmtree('./data1/X_Matrix/')
    os.mkdir('./data1/X_Matrix')             #创建一个新的名为 'X_Matrix' 的文件夹
    data = pd.read_excel('../data_source/signal_normalized.xlsx')         #读取数据，并将其存储在变量 data 中
    init_data = data.iloc[:, 1:]

    # 设置随机种子
    np.random.seed(10)
    # 设置随机比例
    # ratio = 0.2
    ratios = [0.2, 0.4, 0.6, 0.8]
    # 随机缺失的行列数
    # missing_row = 0
    missing_rows = [0]
    # missing_col = 4
    missing_cols = [2, 4, 6, 8, 10]
    result = pd.DataFrame([], columns=['随机比例', '缺失行数', '缺失列数', 'ERM', 'ERC', 'ERN', 'RMSE',
                                       'MAE'])  # result DataFrame现在是一个空白的表格有上述列名，用于将来存储关于某些数据的结果
    # 统计所有情况
    all_args = product(ratios, missing_rows, missing_cols)  # 创建了一个包含所有参数组合的迭代器或列表。
    result_index = 0  # 初始化一个结果的索引变量
    for ratio, missing_row, missing_col in all_args:
        res, L, to_cmp = main(init_data, ratio, missing_row, missing_col)  # 获得恢复后的res指标，L恢复矩阵，to_cmp初始缺失矩阵
        result.loc[result_index] = [ratio, missing_row, missing_col] + list(
            res.values())  # 创建一个包含多个值的列表，包括 ratio、missing_row、missing_col 以及 res 中的所有值
        print(f'ep:{result_index}, ratio: {ratio}, missing_row: {missing_row}, missing_col: {missing_col}, {res}')
        res_L = pd.DataFrame(L, columns=init_data.columns)
        res_L.to_csv(f'./data1/L_Matrix/L{result_index}.csv', index=False)

        to_cmp = pd.DataFrame(to_cmp, columns=init_data.columns)
        to_cmp.to_csv(f'./data1/X_Matrix/X{result_index}.csv', index=False)
        result_index += 1

    curr_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

    sns.catplot(x="随机比例", y="ERM", data=result, kind="point", hue="缺失行数", col="缺失列数")
    plt.tight_layout()
    plt.savefig(f'./data1/fig/fig1_{curr_time}.png', dpi=200)

    sns.catplot(x="缺失列数", y="ERC", data=result, kind="point", hue="缺失行数", col="随机比例")
    plt.tight_layout()
    plt.savefig(f'./data1/fig/fig2_{curr_time}.png', dpi=200)

    result.to_csv(f'./data1/result/result_{curr_time}.csv', encoding='utf-8', index=False)
    '''
    configure --prefix=/usr --disable-profile --enable-add-ons --with-headers=/usr/include --with-binutils=/usr/bin --disable-sanity-checks --disable-werror
    '''

































#
# class MF():
#
#     def __init__(self, data,X, k, alpha, beta, iterations):
#         """
#         Perform matrix factorization to predict np.nan entries in a matrix.
#         Arguments
#         - data: complete dataset
#         - X (ndarray)   : sample-feature matrix
#         - k (int)       : number of latent dimensions
#         - alpha (float) : learning rate
#         - beta (float)  : regularization parameter
#         """
#         self.data = data
#         self.X = X
#         self.num_samples, self.num_features = X.shape
#         self.k = k
#         self.alpha = alpha
#         self.beta = beta
#         self.iterations = iterations
#         # True if not nan
#         self.not_nan_index = (np.isnan(self.X) == False)
#         self.nan_index = (np.isnan(self.X) == True)
#
#     def train(self):
#         # Initialize factorization matrix U and V
#         self.U = np.random.normal(scale=1. / self.k, size=(self.num_samples, self.k))
#         self.V = np.random.normal(scale=1. / self.k, size=(self.num_features, self.k))
#
#         # Initialize the biases
#         self.b_u = np.zeros(self.num_samples)
#         self.b_v = np.zeros(self.num_features)
#         self.b = np.mean(self.X[np.where(self.not_nan_index)])
#         # Create a list of training samples
#         self.samples = [
#             (i, j, self.X[i, j])
#             for i in range(self.num_samples)
#             for j in range(self.num_features)
#             if not np.isnan(self.X[i, j])
#         ]
#
#         # Perform stochastic gradient descent for number of iterations
#         training_process = []
#         for i in range(self.iterations):
#             np.random.shuffle(self.samples)
#             self.sgd()
#             # total square error
#             se = self.square_error()
#             training_process.append((i, se))
#             if (i + 1) % 10 == 0:
#                 print("Iteration: %d ; error = %.4f" % (i + 1, se))
#
#         return training_process
#
#
#     def square_error(self):
#         """
#         A function to compute the total square error
#         """
#         predicted = self.full_matrix()
#         error = 0
#         number = 0
#         for i in range(self.num_samples):
#             for j in range(self.num_features):
#                 if self.not_nan_index[i, j]:
#                     error += pow(self.X[i, j] - predicted[i, j], 2)
#                     number += 1
#         return error/number
#
#     @fn_timer
#     def sgd(self):
#         """
#         Perform stochastic graident descent
#         """
#         for i, j, x in self.samples:
#             # Computer prediction and error
#             prediction = self.get_x(i, j)
#             e = (x - prediction)
#
#             # Update biases
#             self.b_u[i] += self.alpha * (2 * e - self.beta * self.b_u[i])
#             self.b_v[j] += self.alpha * (2 * e - self.beta * self.b_v[j])
#
#             # Update factorization matrix U and V
#             """
#             If RuntimeWarning: overflow encountered in multiply,
#             then turn down the learning rate alpha.
#             """
#             self.U[i, :] += self.alpha * (2 * e * self.V[j, :] - self.beta * self.U[i, :])
#             self.V[j, :] += self.alpha * (2 * e * self.U[i, :] - self.beta * self.V[j, :])
#
#     def get_x(self, i, j):
#         """
#         Get the predicted x of sample i and feature j
#         """
#         prediction = self.b + self.b_u[i] + self.b_v[j] + self.U[i, :].dot(self.V[j, :].T)
#         return prediction
#
#     def full_matrix(self):
#         """
#         Computer the full matrix using the resultant biases, U and V
#         """
#         return self.b + self.b_u[:, np.newaxis] + self.b_v[np.newaxis, :] + self.U.dot(self.V.T)
#
#     def replace_nan(self, X_hat):
#         """
#         Replace np.nan of X with the corresponding value of X_hat
#         """
#         X = np.copy(self.X)
#         for i in range(self.num_samples):
#             for j in range(self.num_features):
#                 if np.isnan(X[i, j]):
#                     X[i, j] = X_hat[i, j]
#         return X
#
#
#
#
#
#





# #
# data = sio.loadmat(r"E:\pythonproject\fuzzy_feature_selection\data\lungcancer.mat")
# data2 = data['lungcancer']
# data = pd.DataFrame(data2)
# #
# #
# # rui = data[4][5]
# #
# X = data.iloc[:,:-1]
#
#
# # Y = data.iloc[:,[-1]]
# #
# #
# #
# X_masked = mask_types(X,0.5,1)   #随机缺失数据
# R = X_masked.copy()
# R[np.isnan(X_masked)] = 0
# R = pd.DataFrame(R)
# # # new_data = pd.concat((R,Y),axis=1)
# # #
# R = coo_matrix(R)
# # # y = len(R.data)
# P,Q=SGD(data,R,X_masked,K=20,gamma=0.000007,lamda=0.01, steps=40)
# # # # # preR = pd.DataFrame(P.dot(Q))
