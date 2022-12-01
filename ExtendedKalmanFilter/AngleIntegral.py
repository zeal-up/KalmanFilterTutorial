import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

# 状态方程：定义一个非线性方程

# -----------------------------------------

# 生成没有噪声的观测信号
N_steps = 1000  # 1000 steps to estimate
Delta_t = 0.1  # 0.1s
timeline = np.arange(N_steps) * Delta_t
x0_gt = 0.5
angular_rate = 0.1  # constant angular rate, rad/s
angular_rate_axis = np.array([1., 0, 0])
angular_rate_theta = angular_rate
x0 = R.from_euler('XYZ', [0., 0., 0])

# 生成带噪声的信号——角速度观测值
zs_gt = np.ones(N_steps) * angular_rate_theta
zs_noise_var = 0.1
zs_noises = np.random.randn(N_steps) * zs_noise_var
zs_w_noise = zs_gt + zs_noises

xs_angle_gt = []
xs_angle_noise = []
xt_gt = x0
xt_noise = x0
for i in range(N_steps):
    delta_q_gt = R.from_rotvec(angular_rate_axis*zs_gt[i]*Delta_t)
    delta_q_noise = R.from_rotvec(angular_rate_axis*zs_w_noise[i]*Delta_t)
    xt_gt = delta_q_gt * xt_gt
    xt_noise = delta_q_noise * xt_noise

    x_angle_gt = xt_gt.as_euler('XYZ', degrees=False)[0]
    x_angle_noise = xt_noise.as_euler('XYZ', degrees=False)[0]
    xs_angle_gt.append(x_angle_gt)
    xs_angle_noise.append(x_angle_noise)

xs_angle_gt = np.array(xs_angle_gt).reshape(-1)
xs_angle_noise = np.array(xs_angle_noise).reshape(-1)


#---------------------------------
# vis result 
VIS_DATA = True
if VIS_DATA:
    plt.plot(timeline, xs_angle_gt, 'r', linewidth=2)
    plt.plot(timeline, xs_angle_noise, c='b')
    plt.title("zs-gt(red) / zs-w-noise(blue)")
    plt.show()
# ==========================================================================================

def derivative_f(delta_angle):
    delta_R = R.from_euler('XYZ', [delta_angle])
    delta_q = delta_R.as_quat()  # q1, q2, q3, w
    q1, q2, q3, w = delta_q
    return np.array(
        [w, -q1, -q2, -q3],
        [q1, w, q3, -q2],
        [q2, -q3, w, q1],
        [q3, q2, -q1, w]
        
    )

def derivtive_h(delta_angle):
    return np.array([
        [0, -delta_angle, 0, 0],
        [delta_angle, 0, 0, 0],
        [0, 0, 0, delta_angle],
        [0, 0, -delta_angle, 0]
    ])

def fun_c(delta_angle, xt):
    xt = R.from_euler('XYZ', xt, degrees=False)
    delta_R = R.from_euler('XYZ', delta_angle, degrees=False)
    ret_R = delta_R * xt
    ret_angle = ret_R.as_euler('XYZ', degrees=False)[0]
    return ret_angle

# ------------------------------------------------------------------------------------------
# 设置初始值
x0 = 0.1  # 系统状态向量初始值，假设未知，随便设为0
P0 = 0.1  # 系统状态初始协方差，由于x0是不准确的，因此P0不能为0
Q = 0.001  # 过程噪声的协方差，需要调整的参数
R = 0.001  # 观测噪声的协方差，需要调整的参数

x_t_ = None  # predicted system state vector
x_t = None  # corrected system state vector
P_t_ = None  # covariance matrix of predicted state vector
P_t = None  # covariance matrix of corrected state vector
K = None

kf_result = []
x_t = x0
P_t = P0
for i in range(N_steps):

    x_t_ = f(x_t)  # 预测方程
    P_t_ = A(x_t)**2*P0 + W(x_t)**2*Q  # 预测状态向量的协方差矩阵
    

    zt = zs_w_noise[i]  # 当前时刻的观测值
    zth = h(x_t_)  # 当前时刻测量值的预测值
    H_t = H(zt)  # 当前时刻H矩阵的值
    K = P_t_*H_t/(H_t**2*P_t_+V(zt)**2*R)  # 卡尔曼增益
    x_t = x_t_ + K*(zt-zth)  # 更新方程
    P_t = P_t_ - K*H_t*P_t_  # 更新状态向量协方差矩阵

    kf_result.append(x_t)
kf_result = np.array(kf_result)

# vis result 
VIS_DATA = True
if VIS_DATA:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(timeline, zs_gt, 'r', linewidth=2)
    ax.scatter(timeline, zs_w_noise, c='b', marker='x', s=0.5)
    # ax.plot(timeline, kf_result, 'g', linewidth=1)
    ax.set_title("zs-gt(red) / zs-w-noise(blue) / KF result(green)")

    # plt.show()
