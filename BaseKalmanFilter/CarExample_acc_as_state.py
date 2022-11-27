import numpy as np
from matplotlib import pyplot as plt


C = 100  # velocity of light, just a simulation value
# generate perfect acceleration and corresponding velocity and distance
# 生成没有噪声的加速度信号，并用积分计算出对应的速度和距离 ---------------------------------
N_steps = 1000  # 1000 steps to estimate
Delta_t = 0.05  # time interval is 0.05s
timeline = np.arange(N_steps) * Delta_t  # \Delta t = 0.05s
accs_gt = np.sin(timeline)  # set accelerator as a log shape
vels_gt = np.cumsum(accs_gt*Delta_t)  # get the ground truth velocity;速度是加速度的积分
dists_gt = np.cumsum(vels_gt*Delta_t)  # get the ground truth distance; 距离是速度的积分
zs_gt = dists_gt/C
#----------------------------------

# add noise to acceleration signals and calculate the velocity and distance with simple integral
# 往加速度信号添加噪声模拟真实传感器信号，并用简单的积分计算速度和距离
accs_noise_var = 0.005  # set acceleration noise variation as 0.5
accs_noise = np.random.rand(N_steps) * accs_noise_var  # 注意，这里使用均匀分布模拟噪声，以模拟不知道真实噪声分布情况
accs_w_noise = accs_gt + accs_noise
vels_w_noise = np.cumsum(accs_w_noise*Delta_t)
# vels_w_noise = np.cumsum(accs_noise * Delta_t)
dists_w_noise = np.cumsum(vels_w_noise*Delta_t)
zs_noise_var = 0.001
zs_noise = np.random.rand(N_steps) * zs_noise_var
zs_w_noise = zs_gt + zs_noise
#---------------------------------
VIS_DATA = True
if VIS_DATA:
    fig = plt.figure()
    ax11 = fig.add_subplot(421)
    ax11.plot(timeline, accs_gt)
    ax11.set_title("Acceleration Ground Truth")

    ax12 = fig.add_subplot(422)
    ax12.plot(timeline, accs_w_noise)
    ax12.set_title("Acceleration With Noise")



    ax21 = fig.add_subplot(423)
    ax21.plot(timeline, vels_gt)
    ax21.set_title("Velocity Ground Truth")

    ax22 = fig.add_subplot(424)
    ax22.plot(timeline, vels_w_noise)
    ax22.set_title("Velocity With Noise")


    ax31 = fig.add_subplot(425)
    ax31.plot(timeline, dists_gt)
    ax31.set_title("Distance Ground Truth")

    ax32 = fig.add_subplot(426)
    ax32.plot(timeline, dists_w_noise)
    ax32.set_title("Distance With Noise")

    ax41 = fig.add_subplot(427)
    ax41.plot(timeline, zs_gt)
    ax41.set_title("Measurment Ground Truth")

    ax42 = fig.add_subplot(428)
    ax42.plot(timeline, zs_w_noise)
    ax42.set_title("Measurment With Noise")
    # plt.show()
    plt.subplots_adjust(wspace =0, hspace =0.5)#调整子图间距

# ------------------------------------------------------------------------------------------

A = np.array([
    [1, Delta_t, Delta_t**2*0.5],
    [0, 1, Delta_t],
    [0, 0, 1]
])  # 状态转移矩阵

Q_var = 0.005
Q = np.array([
    [Q_var, 0, 0],
    [0, Q_var, 0],
    [0, 0, Q_var]
])  # 预测噪声协方差矩阵
P_var = 1e-5
P0 = np.array([
    [P_var, 0, 0],
    [0.0, P_var, 0],
    [0, 0, P_var]
])  # 状态向量协方差初始值

H = np.array([
    [1/C, 0, 0],
    [0, 0, 1]
])  # 测量转换矩阵
R_var = 0.0001
R = np.array([
    [R_var, 0],
    [0, R_var*5],
])  # 测量协方差矩阵，此时只有一个测量变量，因此只有一个值

x0 = np.array([
    [0],
    [0],
    [0]
])  # 系统状态向量初始化，速度和距离均初始化为0

x_t_ = None  # predicted system state vector
x_t = None  # corrected system state vector
P_t_ = None  # covariance matrix of predicted state vector
P_t = None  # covariance matrix of corrected state vector
K = None

est_vel = [0]
est_dist = [0]
est_acc = [0]
for i in range(N_steps):
    if i == 0:
        x_t = x0
        P_t = P0
        continue

    x_t_ = A@x_t  # 预测方程
    P_t_ = A@P_t@(A.T) + Q  # 预测状态向量的协方差矩阵
    

    K = P_t_@H.T @ np.linalg.inv((H@P_t_@H.T + R))  # 卡尔曼增益
    zt = np.array([
        [zs_w_noise[i]],
        [accs_w_noise[i]]
    ])
    x_t = x_t_ + K@(zt - H@x_t_)  # 更新方程
    P_t = P_t_ - K@H@P_t_  # 更新状态向量协方差矩阵

    est_vel.append(x_t[1][0])
    est_dist.append(x_t[0][0])

est_vel = np.array(est_vel).reshape(-1)
est_dist = np.array(est_dist).reshape(-1)
diff_vel_est = est_vel - vels_gt
diff_dist_est = est_dist - dists_gt
diff_vel_sum = vels_w_noise - vels_gt
diff_dist_sum = dists_w_noise - dists_gt


# print(est_vel)
fig2 = plt.figure()
ax2_11 = fig2.add_subplot(231)
ax2_11.plot(timeline, est_vel)
ax2_11.set_title("Estimated velocity")

ax2_12 = fig2.add_subplot(232)
ax2_12.plot(timeline, diff_vel_est)
ax2_12.set_title("Error - Vel - KalmanFilter")

ax2_13 = fig2.add_subplot(233)
ax2_13.plot(timeline, diff_vel_sum)
ax2_13.set_title("Error - Vel - Simple integration")

ax2_21 = fig2.add_subplot(234)
ax2_21.plot(timeline, est_dist)
ax2_21.set_title("Estimated distance")

ax2_22 = fig2.add_subplot(235)
ax2_22.plot(timeline, diff_dist_est)
ax2_22.set_title("Error - Dist - KalmanFilter")

ax2_23 = fig2.add_subplot(236)
ax2_23.plot(timeline, diff_dist_sum)
ax2_23.set_title("Error - Dist - Simple integration")


plt.show()
