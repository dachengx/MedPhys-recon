import numpy as np
from tqdm import tqdm
import numba


# @numba.njit
def sensitivity_map(t, pixels_wall, x_num):
    """
    使用 Siddon 算法计算传输矩阵
    Parameters
    ----------
    t : array
        投影
    pixels_wall : list
        像素在各个维度的边界坐标
    x_num : list
        传输矩阵的 index

    Returns
    -------
    idx : array
        传输矩阵中具有非 0 相交长度的 index
    line : array
        相交长度，与 idx 长度相同
    """
    dim = len(t) // 2
    d = 0
    for j in range(dim):
        d += (t[j + dim] - t[j]) ** 2
    d = np.sqrt(d)
    d_frac = np.empty(0)
    dig_list = []
    dig_l = []
    for j, wall in enumerate(pixels_wall):
        x = [t[j], t[j+dim]]
        dig_list.append(np.digitize(x, wall))
        if dig_list[j][0] > dig_list[j][-1]:
            tl = np.arange(dig_list[j][-1], dig_list[j][0], 1)[::-1]
        else:
            tl = np.arange(dig_list[j][0], dig_list[j][-1], 1)
        dig_l.append(tl)
        x_list = pixels_wall[j][dig_l[j]]
        d_l = (x_list - t[j]) / (t[j+dim] - t[j]) * d
        d_frac = np.append(d_frac, d_l)
    x_init = np.array([[dig_list[j][0] - 1 for j in range(dim)]]).T
    ll = np.array([0])
    for j in range(dim):
        ll = np.append(ll, len(dig_l[j]))
    ll = np.cumsum(ll)
    int_list = np.zeros((dim, ll[-1])).astype(np.int16)
    for j in range(dim):
        if dig_list[j][0] > dig_list[j][-1]:
            int_list[j, ll[j]:ll[j+1]] = -1
        elif dig_list[j][0] < dig_list[j][-1]:
            int_list[j, ll[j]:ll[j+1]] = 1
    x_delta = int_list[:, d_frac.argsort()][:, :-1]
    x_cumdelta = np.zeros((dim, ll[-1] - 1)).astype(np.int16)
    for j in range(dim):
        x_cumdelta[j] = np.cumsum(x_delta[j])
    x_dig = x_init + x_cumdelta
    pixels_dig_xyz = x_dig * x_num
    pixels_dig = np.sum(pixels_dig_xyz, axis=0)
    vali = np.full(x_dig.shape[1], True)
    for j, wall in enumerate(pixels_wall):
        vali &= ((x_dig[j] >= 0) & (x_dig[j] < len(wall) - 1))
    idx = pixels_dig[vali]
    line = np.diff(np.sort(d_frac))[vali]
    return idx, line


def osem(p, m, iteration, L):
    """
    使用 OS-EM 算法进行迭代重建
    Parameters
    ----------
    p : array
        投影
    m : array
        传输矩阵
    iteration : int
        迭代次数
    L : int
        分割投影的份数

    Returns
    -------
    f_list : list
        重建结果列表
    d_list : list
        前后两次迭代的重建结果按像素的最大差异
    """
    f = np.full(m.shape[1], p.sum() / m.sum())
    f_list = [f]
    d_list = []
    D = m.shape[0] // L
    tiny = np.finfo(np.float32).tiny

    for i in tqdm(range(iteration)):
        for j in range(L):
            f_t = f_list[-1] / (m[D*j:D*(j+1), :].sum(axis=0) + tiny) * (m[D*j:D*(j+1), :].T @ (p[D*j:D*(j+1)] / (m[D*j:D*(j+1), :] @ f_list[-1] + tiny)))
            f_list.append(f_t)
            d_list.append(np.abs(f_list[-2] - f_list[-1]).max())
            # if d_list[-1] < 1e-3:
            #     break
    return f_list, d_list


def mlem(p, m, iteration):
    """
    使用 ML-EM 算法进行迭代重建
    Parameters
    ----------
    p : array
        投影
    m : array
        传输矩阵
    iteration : int
        迭代次数

    Returns
    -------
    f_list : list
        重建结果列表
    d_list : list
        前后两次迭代的重建结果按像素的最大差异
    """
    f = np.full(m.shape[1], p.sum() / m.sum()).astype(np.float32)
    f_list = [f]
    d_list = []

    for i in tqdm(range(iteration)):
        f_t = f_list[-1] / m.sum(axis=0) * (m.T @ (p / (m @ f_list[-1])))
        f_list.append(f_t)
        d_list.append(np.abs(f_list[-2] - f_list[-1]).max())
        if d_list[-1] < 1e-3:
            break
    return f_list, d_list
