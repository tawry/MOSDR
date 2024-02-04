import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import time
import matplotlib as mpl
import scipy.sparse as sp
import time

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def test(sparse_mat, miu, lamda, beta, maxiter):
    pos_train = np.where(sparse_mat != 0)
    X = sparse_mat.copy()
    L = sparse_mat.copy()
    S = np.zeros(sparse_mat.shape)
    G = np.zeros(sparse_mat.shape)
    Y = np.zeros(sparse_mat.shape)

    for it in range(maxiter):
        # 更新L
        Cl = L + Y / miu
        u, s, v = np.linalg.svd(Cl, full_matrices=0)
        vec = s - 1/ miu
        vec[np.where(vec < 0)] = 0   #奇异值阈值算子
        L = np.matmul(np.matmul(u, np.diag(vec)), v)
        L[pos_train] = X[pos_train]   #将X矩阵原不为0的地方放入L对应位置，不修改有值位置的值
        # 更新S
        Cs = S + Y / miu
        tau = lamda / miu
        tmp_b = np.where(Cs > tau)    #获取对应索引，软阈值
        tmp_s = np.where(Cs < -tau)
        tmp_o = np.where((Cs <= tau) & (Cs >= -tau))
        S[tmp_b] = (Cs - tau)[tmp_b]
        S[tmp_s] = (Cs + tau)[tmp_s]
        S[tmp_o] = 0
        # 更新G
        G = miu / (2 * beta + miu) * (G + Y / miu)
        # 更新Y
        Y = Y + miu * (X - L - S - G)
    # ll3 = np.where(L == 0)
    return L, S, G

def test2(sparse_mat, miu, lamda, beta, yita,maxiter):
    pos_train = np.where(sparse_mat != 0)
    X = sparse_mat.copy()
    L = sparse_mat.copy()
    K = sparse_mat.copy()
    S = np.zeros(sparse_mat.shape)
    G = np.zeros(sparse_mat.shape)
    Y1 = np.zeros(sparse_mat.shape)
    Y2 = np.zeros(sparse_mat.shape)
    n = sparse_mat.shape[1]
    D = np.zeros((n, n - 1))
    for i in range(n - 1):   #时间微分矩阵一定要动态调整
        D[i, i] = -1
        D[i + 1, i] = 1
    DT = np.transpose(D)
    for it in range(maxiter):
        # 更新L
        Cl = 0.5*(L + Y1 / miu + K - Y2 / miu)
        u, s, v = np.linalg.svd(Cl, full_matrices=0)
        vec = s - 1 / (2*miu)
        vec[np.where(vec < 0)] = 0   #奇异值阈值算子
        L = np.matmul(np.matmul(u, np.diag(vec)), v)
        #L[pos_train] = X[pos_train]   #将X矩阵原不为0的地方放入L对应位置，不修改有值位置的值
        # 更新S
        Cs = S + Y1 / miu
        tau = lamda / miu
        tmp_b = np.where(Cs > tau)    #获取对应索引，软阈值
        tmp_s = np.where(Cs < -tau)
        tmp_o = np.where((Cs <= tau) & (Cs >= -tau))
        S[tmp_b] = (Cs - tau)[tmp_b]
        S[tmp_s] = (Cs + tau)[tmp_s]
        S[tmp_o] = 0
        # 更新G
        G = miu / (2 * beta + miu) * (G + Y1 / miu)
        # 更新K
        K = (Y2 + miu * L) @ np.linalg.inv(2 * yita * np.dot(D, DT) + miu * np.eye(n))
        # 更新Y
        Y1 = Y1 + miu * (X - L - S - G)
        Y2 = Y2 + miu * (L - K)
    # ll3 = np.where(L == 0)
    return L, S, G


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


def data_recovery(X, missing_cols, missing_rows, miu, lamda, beta, miu2, lamda2, beta2, yita2,maxiter=100):
    '''第一步①先删除列缺失部分，对剩下的具有随机缺失的矩阵，进行中间(1)(2)(3)的数据恢复'''
    # 数据恢复
    #yita = 0.0001
    L, S, G = test(X, miu, lamda, beta,maxiter)
    '''第二步②把缺失的列用新的值插进去，获得的新矩阵再次用(1)(2)(3)进行数据恢复'''
    # 缺失列插入新值
    for i in sorted(missing_cols):          #遍历了 missing_cols 中的列索引
        L = np.insert(L, i, 0, axis=1)      #在矩阵 L 的指定列位置 i 插入零列
    c_shape = L.shape[1]        #计算了矩阵 L 的列数
    for i in sorted(missing_cols):
        if i == 0:
            L[:, i] = L[:,i + 1]            #将第一列的数据设置为与第二列相同的数据，以模拟缺失数据的填充。因为第一列没有左侧的相邻列，所以选择使用右侧相邻列的数据来填充。
        elif i == c_shape - 1:
            L[:, i] = L[:, i - 1]           #如果列索引是最后一列，将最后一列的数据设置为与倒数第二列相同的数据
        else:
            L[:,i] = np.mean(L[:,[i-1,i+1]],axis=1)   #如果列索引既不是0也不是最后一列，将当前列的数据设置为与其左侧和右侧相邻列的数据的平均值
    # 缺失行插入新值
    for j in sorted(missing_rows):
        L = np.insert(L, j, 0, axis=0)
    r_shape = L.shape[0]
    for i in sorted(missing_rows):
        if i == 0:
            L[i, :] = L[i + 1,:]
        elif i == r_shape - 1:
            L[i, :] = L[i - 1,:]
        else:
            L[i, :] = np.mean(L[[i-1,i+1],:],axis=0)
    # 数据恢复
    #L, S, G = test(L, miu, lamda, beta, maxiter)
    L, S, G = test2(L, miu2, lamda2, beta2, yita2,maxiter)
    return L, S, G


def main(init_data, ratio=0.2, missing_row=0, missing_col=0):
    start_time = time.time()
    # 设置随机矩阵
    X = (init_data.values).copy()           #将传入的 init_data 转换为 NumPy 数组，存在X变量中
    indices = np.random.choice(X.shape[1] * X.shape[0], replace=False, size=int(X.shape[1] * X.shape[0] * ratio))       #生成一个随机索引数组(indices)，用于表示要在矩阵X中随机置为零的元素位置。
    X[np.unravel_index(indices, X.shape)] = 0       #将矩阵 X 中的特定位置的元素设置为零，这些位置由之前生成的 indices 数组表示

    # 初始化参数
    # miu = 0.002
    # lamda = 0.0001
    # beta = 0.0001
    # maxiter = 100
    #
    # yita2 = 100
    # miu2 = 1
    # lamda2 = 20
    # beta2 = 200

    miu = 1
    lamda = 0.0001
    beta = 0.0001
    maxiter = 100

    yita2 = 20
    miu2 = 200
    lamda2 = 10
    beta2 = 0.0001

    # 设置随机缺失的行列
    missing_rows = np.random.choice(X.shape[0], replace=False, size=missing_row)  #缺失行索引
    missing_cols = np.random.choice(X.shape[1], replace=False, size=missing_col)  #缺失列索引

    #创建一个名为 to_cmp 的矩阵，该矩阵是原始数据矩阵 X 的一个副本，并且在指定的行和列位置上将其元素设置为零。这个矩阵通常用于后续的数据比较或验证操作，以检查数据恢复的效果。
    to_cmp = X.copy()
    to_cmp[:,missing_cols] = 0      #模拟了缺失列的操作，将这些列的元素都置为零。
    to_cmp[missing_rows,:] = 0      #模拟了缺失行的操作，将这些行的元素都置为零。

    X = np.delete(X, missing_rows, 0)  #删除指定缺失行
    X = np.delete(X, missing_cols, 1)  #删除指定缺失列


    # 未缺失部分数据 index
    data_index = np.where(X != 0)

    '''数据恢复'''
    L, S, G = data_recovery(X, missing_cols, missing_rows, miu, lamda, beta, miu2, lamda2, beta2, yita2,maxiter)
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
    # if os.path.exists('./twomodel_data4/twomodel_data4_L_Matrix'):   #检查当前目录下是否存在名为 'L_Matrix' 的文件夹
    #     shutil.rmtree('./twomodel_data4/twomodel_data4_L_Matrix/')   #如果存在会删除名为 'L_Matrix' 的文件夹及其内容
    # os.mkdir('./twomodel_data4/twomodel_data4_L_Matrix')             #创建一个新的名为 'L_Matrix' 的文件夹
    # if os.path.exists('./twomodel_data4/twomodel_data4_X_Matrix'):
    #     shutil.rmtree('./twomodel_data4/twomodel_data4_X_Matrix/')
    # os.mkdir('./twomodel_data4/twomodel_data4_X_Matrix')             #创建一个新的名为 'X_Matrix' 的文件夹
    data = pd.read_csv('../data_source/Tetuan City power consumption2.csv')         #读取数据，并将其存储在变量 data 中
    init_data = data.iloc[:1000, :]        #选择了所有行（":" 表示选择所有行）和从第二列（索引为1）到最后一列的所有列，排除了 DataFrame 中的第一列
    normalized_data = (init_data - init_data.min()) / (init_data.max() - init_data.min())
    init_data = normalized_data.transpose()

    # 设置随机种子
    np.random.seed(10)
    # 设置随机比例
    # ratio = 0.2
    ratios =[0.2,0.4,0.6,0.8]
    # 随机缺失的行列数
    # missing_row = 0
    missing_rows = [0]
    # missing_col = 4
    missing_cols = [2,4,6,8,10]
    result = pd.DataFrame([],columns=['随机比例','缺失行数','缺失列数','ERM','ERC','ERN','RMSE','MAE'])        #result DataFrame现在是一个空白的表格有上述列名，用于将来存储关于某些数据的结果
    # 统计所有情况
    all_args = product(ratios,missing_rows,missing_cols)            #创建了一个包含所有参数组合的迭代器或列表。
    result_index = 0                            #初始化一个结果的索引变量
    for ratio,missing_row,missing_col in all_args:
        res,L,to_cmp,elapsed_time = main(init_data, ratio, missing_row, missing_col)      #获得恢复后的res指标，L恢复矩阵，to_cmp初始缺失矩阵
        result.loc[result_index] = [ratio, missing_row, missing_col] + list(res.values())           #创建一个包含多个值的列表，包括 ratio、missing_row、missing_col 以及 res 中的所有值
        print(f'ep:{result_index}, ratio: {ratio}, missing_row: {missing_row}, missing_col: {missing_col}, {res}')
        res_L = pd.DataFrame(L,columns=init_data.columns)
        res_L.to_csv(f'./twomodel_data4/twomodel_data4_L_Matrix/L{result_index}.csv',index=False)

        to_cmp = pd.DataFrame(to_cmp, columns=init_data.columns)
        to_cmp.to_csv(f'./twomodel_data4/twomodel_data4_X_Matrix/X{result_index}.csv', index=False)
        result_index += 1
    print(f"该代码段运行了 {elapsed_time} 秒。")
    curr_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

    sns.catplot(x="随机比例",y="ERM",data=result,kind="point",hue="缺失行数", col="缺失列数")
    plt.tight_layout()
    plt.savefig(f'./twomodel_data3/twomodel_data3_fig/fig1_{curr_time}.png', dpi=200)

    sns.catplot(x="缺失列数", y="ERC", data=result, kind="point", hue="缺失行数", col="随机比例")
    plt.tight_layout()
    plt.savefig(f'./twomodel_data4/twomodel_data4_fig/fig2_{curr_time}.png', dpi=200)

    # result.to_csv(f'./twomodel_data4/twomodel_data4_result/result_{curr_time}.csv',index=False)
    '''
    configure --prefix=/usr --disable-profile --enable-add-ons --with-headers=/usr/include --with-binutils=/usr/bin --disable-sanity-checks --disable-werror
    '''
