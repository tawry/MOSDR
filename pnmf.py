"""
Probabilistic NMF:
Fekade, Berihun, et al. "Probabilistic recovery of incomplete sensed data in IoT." IEEE Internet of Things Journal 5.4 (2017): 2282-2292.
"""
from nmfbase import NMFBase
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import matplotlib as mpl
import time

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

class PNMF(NMFBase):
    """
    Attributes
    ----------
    W : matrix of basis vectors
    H : matrix of coefficients
    frob_error : frobenius norm
    """
    def compute_factors(self, max_iter=100, alpha= 0.2, beta= 0.2):
        # if self.check_non_negativity():
        #     pass
        # else:
        #     print("The given matrix contains negative values")
        #     exit()
        if not hasattr(self,'W'):
            self.initialize_w()

        if not hasattr(self,'H'):
            self.initialize_h()

        self.frob_error = np.zeros(max_iter)

        for i in range(max_iter):
            self.update_h(alpha)
            self.update_w(beta)
            self.frob_error[i] = self.frobenius_norm()

    def update_h(self, beta):

        XtW = np.dot(self.W.T, self.X)
        HWtW = np.dot(self.W.T.dot(self.W), self.H ) + beta+ 2**-8
        self.H *= XtW
        self.H /= HWtW

    def update_w(self, alpha):

        XH = self.X.dot(self.H.T)
        WHtH = self.W.dot(self.H.dot(self.H.T)) + alpha+ 2**-8
        self.W *= XH
        self.W /= WHtH

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
    to_cmp[missing_rows,:] = 0     #模拟了缺失行的操作，将这些行的元素都置为零。

    #X = np.delete(X, missing_rows, 0)  #删除指定缺失行
    #X = np.delete(X, missing_cols, 1)  #删除指定缺失列


    # 未缺失部分数据 index
    data_index = np.where(to_cmp != 0)

    '''数据恢复'''
    pnmf = PNMF(to_cmp, rank=1)
    pnmf.compute_factors(500, 0.002, 0.002)
    L = np.dot(pnmf.W, pnmf.H)
    '''计算指标'''
    raw_data = init_data.values
    result = cal_indices(raw_data, L, data_index,indices)

    return result,L,to_cmp

if __name__ == '__main__':
    import shutil
    import os
    if os.path.exists('./data1/L_Matrix'):  # 检查当前目录下是否存在名为 'L_Matrix' 的文件夹
        shutil.rmtree('./data1/L_Matrix/')  # 如果存在会删除名为 'L_Matrix' 的文件夹及其内容
    os.mkdir('./data1/L_Matrix')  # 创建一个新的名为 'L_Matrix' 的文件夹
    if os.path.exists('./data1/X_Matrix'):
        shutil.rmtree('./data1/X_Matrix/')
    os.mkdir('./data1/X_Matrix')  # 创建一个新的名为 'X_Matrix' 的文件夹
    data = pd.read_excel('../data_source/signal_normalized.xlsx')  # 读取数据，并将其存储在变量 data 中
    init_data = data.iloc[:, 1:]

    # 设置随机种子
    np.random.seed(10)
    # 设置随机比例
    # ratio = 0.2
    ratios = [0.2,0.4,0.6,0.8]
    # 随机缺失的行列数
    # missing_row = 0
    missing_rows = [0]
    # missing_col =0
    missing_cols = [2,4,6,8,10]
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


