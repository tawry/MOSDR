# -*- coding: utf-8 -*-
# Created on Wed Jul 9 10:01:33 2014
# Implemented in Python 3.4.0
# Author: Yun-Jhong Wu
# E-mail: yjwu@umich.edu

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import time
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def RobustPCA(X, lbd=.1, nu=1, rho=1.5, tol=1e-4, maxiters=1000):
    """
    Robust PCA by augmented Lagrange multiplier
    (Lin, Z., Chen, M., & Ma, Y. (2010). The augmented lagrange multiplier
     method for exact recovery of corrupted low-rank matrices.
     arXiv preprint arXiv:1009.5055.)

    lbd: weight of sparse error (relative to low-rank term)
    nu: initial step size (nu = 1 / $mu$ for $mu$ in the above paper)
    rho: step size adjustment
    """

    niters = 0
    rank_new = 100
    L = np.zeros(X.shape)
    n = np.min(X.shape)
    Res = np.array([[np.inf]])
    X_Fnorm = np.linalg.norm(X, 'fro')
    tol *= X_Fnorm
    nu *= X_Fnorm
    Y = 1 / max(X_Fnorm, np.max(np.abs(X)) / lbd) * X

    while np.linalg.norm(Res, 'fro') > tol and niters < maxiters:
        niters += 1
        X_plus_Y = X + nu * Y
        S = X_plus_Y - L
        S = np.maximum(S - lbd * nu, 0) + np.minimum(S + lbd * nu, 0)
        U, D, V = np.linalg.svd(X_plus_Y - S, full_matrices=False)
        D -= nu
        D = D[D > 0]
        rank_new = min(D.size + 1 + (D.size < rank_new) * int(0.05 * n), n)
        L = np.dot(U[:, :D.size] * D, V[:D.size, :])
        Res = X - L - S
        Y += (1 / nu) * Res
        nu /= rho

    return L, S

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


def data_recovery(X, missing_cols, missing_rows, miu=0.01, lamda=0.01, beta=0.01, maxiter=100):
    # '''第一步①先删除列缺失部分，对剩下的具有随机缺失的矩阵，进行中间(1)(2)(3)的数据恢复'''
    # # 数据恢复
    # L, S = RobustPCA(X, lbd=.1, nu=1, rho=1.5, tol=1e-4, maxiters=1000)
    # '''第二步②把缺失的列用新的值插进去，获得的新矩阵再次用(1)(2)(3)进行数据恢复'''
    # # 缺失列插入新值
    # for i in sorted(missing_cols):          #遍历了 missing_cols 中的列索引
    #     L = np.insert(L, i, 0, axis=1)      #在矩阵 L 的指定列位置 i 插入零列
    # c_shape = L.shape[1]        #计算了矩阵 L 的列数
    # for i in sorted(missing_cols):
    #     if i == 0:
    #         L[:, i] = L[:,i + 1]            #将第一列的数据设置为与第二列相同的数据，以模拟缺失数据的填充。因为第一列没有左侧的相邻列，所以选择使用右侧相邻列的数据来填充。
    #     elif i == c_shape - 1:
    #         L[:, i] = L[:, i - 1]           #如果列索引是最后一列，将最后一列的数据设置为与倒数第二列相同的数据
    #     else:
    #         L[:,i] = np.mean(L[:,[i-1,i+1]],axis=1)   #如果列索引既不是0也不是最后一列，将当前列的数据设置为与其左侧和右侧相邻列的数据的平均值
    # # 缺失行插入新值
    # for j in sorted(missing_rows):
    #     L = np.insert(L, j, 0, axis=0)
    # r_shape = L.shape[0]
    # for i in sorted(missing_rows):
    #     if i == 0:
    #         L[i, :] = L[i + 1,:]
    #     elif i == r_shape - 1:
    #         L[i, :] = L[i - 1,:]
    #     else:
    #         L[i, :] = np.mean(L[[i-1,i+1],:],axis=0)
    # 数据恢复
    L, S = RobustPCA(X, lbd=.1, nu=1, rho=1.5, tol=1e-4, maxiters=1000)
    return L, S



def main(init_data, ratio=0.2, missing_row=0, missing_col=0):
        start_time = time.time()
    # 设置随机矩阵
    X = (init_data.values).copy()           #将传入的 init_data 转换为 NumPy 数组，存在X变量中
    indices = np.random.choice(X.shape[1] * X.shape[0], replace=False, size=int(X.shape[1] * X.shape[0] * ratio))       #生成一个随机索引数组(indices)，用于表示要在矩阵X中随机置为零的元素位置。
    X[np.unravel_index(indices, X.shape)] = 0       #将矩阵 X 中的特定位置的元素设置为零，这些位置由之前生成的 indices 数组表示

    # 初始化参数
    miu = 1
    lamda = 0.0001
    beta = 0.0001
    maxiter = 100
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
    L, S= data_recovery(to_cmp, missing_cols, missing_rows, miu, lamda, beta, maxiter)
    '''计算指标'''
    raw_data = init_data.values
    result = cal_indices(raw_data, L, data_index,indices)
    # 记录结束时间
    end_time = time.time()
    # 计算并打印运行时间
    elapsed_time = end_time - start_time
    return result,L,to_cmp,elapsed_time

if __name__ == '__main__':
    import shutil
    import os

    # if os.path.exists('./data1/L_Matrix'):  # 检查当前目录下是否存在名为 'L_Matrix' 的文件夹
    #     shutil.rmtree('./data1/L_Matrix/')  # 如果存在会删除名为 'L_Matrix' 的文件夹及其内容
    # os.mkdir('./data1/L_Matrix')  # 创建一个新的名为 'L_Matrix' 的文件夹
    # if os.path.exists('./data1/X_Matrix'):
    #     shutil.rmtree('./data1/X_Matrix/')
    # os.mkdir('./data1/X_Matrix')  # 创建一个新的名为 'X_Matrix' 的文件夹
    #data = pd.read_excel('data_source/juming.xlsx')  # 读取数据，并将其存储在变量 data 中
    data = pd.read_excel('../data_source/signal_normalized.xlsx')
    init_data = data.iloc[:, 1:]  # 选择了所有行（":" 表示选择所有行）和从第二列（索引为1）到最后一列的所有列，排除了 DataFrame 中的第一列
    # 设置随机种子
    np.random.seed(10)
    # 设置随机比例
    # ratio = 0.2
    ratios = [0.2,0.4,0.6,0.8]
    # 随机缺失的行列数
    # missing_row = 0
    missing_rows = [0]
    # missing_col = 4
    missing_cols = [2,4,6,8,10]
    result = pd.DataFrame([], columns=['随机比例', '缺失行数', '缺失列数', 'ERM', 'ERC', 'ERN',
                                       'RMSE','MAE'])  # result DataFrame现在是一个空白的表格有上述列名，用于将来存储关于某些数据的结果
    # 统计所有情况
    all_args = product(ratios, missing_rows, missing_cols)  # 创建了一个包含所有参数组合的迭代器或列表。
    result_index = 0  # 初始化一个结果的索引变量
    for ratio, missing_row, missing_col in all_args:
        res, L, to_cmp,elapsed_time = main(init_data, ratio, missing_row, missing_col)  # 获得恢复后的res指标，L恢复矩阵，to_cmp初始缺失矩阵
        result.loc[result_index] = [ratio, missing_row, missing_col] + list(
            res.values())  # 创建一个包含多个值的列表，包括 ratio、missing_row、missing_col 以及 res 中的所有值
        print(f'ep:{result_index}, ratio: {ratio}, missing_row: {missing_row}, missing_col: {missing_col}, {res}')
        res_L = pd.DataFrame(L, columns=init_data.columns)
        res_L.to_csv(f'./data1/L_Matrix/L{result_index}.csv', index=False)

        to_cmp = pd.DataFrame(to_cmp, columns=init_data.columns)
        to_cmp.to_csv(f'./data1/X_Matrix/X{result_index}.csv', index=False)
        result_index += 1
    print(f"该代码段运行了 {elapsed_time} 秒。")
    curr_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

    sns.catplot(x="随机比例", y="ERM", data=result, kind="point", hue="缺失行数", col="缺失列数")  #用于可视化result中的数据，同时根据不同的缺失行数和缺失列数对数据进行分组和着色
    plt.tight_layout()
    plt.savefig(f'data1/fig/fig1_{curr_time}.png', dpi=200)

    sns.catplot(x="缺失列数", y="ERC", data=result, kind="point", hue="缺失行数", col="随机比例")
    plt.tight_layout()
    plt.savefig(f'data1/fig/fig2_{curr_time}.png', dpi=200)

    #result.to_csv(f'data1/result/result_{curr_time}.csv', index=False)
    '''
    configure --prefix=/usr --disable-profile --enable-add-ons --with-headers=/usr/include --with-binutils=/usr/bin --disable-sanity-checks --disable-werror
    '''
