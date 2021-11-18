# %%
import numpy as np
from tools.kine_UAV import KineUAV
from tools.kine_UAV import RefPos
from tools.rotation_matrix import RotationMatrix

from control import lqr

import matplotlib.animation
from   mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# %%
# instantiation
kine_UAV = KineUAV()
ref_pos = RefPos()
rm = RotationMatrix()
# %%
# simulation parameters
tf = 50
dt = 0.01
# LQR parameters
Q = np.diag([0,0,0,0,0,0,0,0,1,1,1])
R = np.diag([0.01,0.01,0.01])
T_trim = 9.81
# initialization
state_now = np.array([0,0,1,0,0,0,0.1,0.1])
state_integral = np.array([0,0,0])
# save data
states = []
controls = []
refs = []
# %%
A_aug, B_aug = kine_UAV.augsys_linear()
K, S, E = lqr(A_aug,B_aug,Q,R)

for time in np.arange(0,tf,dt):
    ref_now = ref_pos.circle(time)
    state_integral = state_integral + (state_now[0:3]-ref_now)*dt
    state_aug_now = np.concatenate((state_now,state_integral))
    u_linear = -np.matmul(K, state_aug_now)
    u_linear[0] += T_trim
    d_state = kine_UAV.kine_nl_all(state_now, u_linear)
    state_next = state_now + d_state*dt

    u_linear[1:] = u_linear[1:]*180/np.pi
    states.append(state_now)
    controls.append(u_linear)
    refs.append(ref_now)
    state_now = state_next
# %%
ref_plot = list(zip(*refs))
state_plot = list(zip(*states))
control_plot = list(zip(*controls))
# %%
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(ref_plot[0], ref_plot[1], ref_plot[2], 'gray')
ax.plot3D(state_plot[0], state_plot[1], state_plot[2], 'blue')
# %%
steps = np.arange(0,tf,dt)
font_size = 14
"""
tracking performance
"""
fig = plt.figure()
##
plt.subplot(3,1,1)
plt.plot(steps, state_plot[0])
plt.plot(steps, ref_plot[0], linestyle='--')
# plt.xlabel('Time [s]',fontsize = font_size)
plt.ylabel(r'$x~\mathrm{[m]}$',fontsize = font_size)
plt.xticks(fontsize = font_size)
plt.yticks(fontsize = font_size)
plt.xlim(0, tf+dt)
plt.grid(color='gray', linestyle=':')
plt.legend(["Real trajectory", "Reference"], loc='upper center',bbox_to_anchor=(0.5, 2), ncol= 2, fontsize = font_size)
##
plt.subplot(3,1,2)
plt.plot(steps, state_plot[1])
plt.plot(steps, ref_plot[1], linestyle='--')
# plt.xlabel('Time [s]',fontsize = font_size)
plt.ylabel(r'$y~\mathrm{[m]}$',fontsize = font_size)
plt.xticks(fontsize = font_size)
plt.yticks(fontsize = font_size)
plt.xlim(0, tf+dt)
plt.grid(axis='both',color='gray', linestyle=':')
##
plt.subplot(3,1,3)
plt.plot(steps, state_plot[2])
plt.plot(steps, ref_plot[2], linestyle='--')
plt.xlabel('Time [s]',fontsize = font_size)
plt.ylabel(r'$z~\mathrm{[m]}$',fontsize = font_size)
plt.xticks(fontsize = font_size)
plt.yticks(fontsize = font_size)
plt.xlim(0, tf+dt)
plt.grid(axis='both',color='gray', linestyle=':')

plt.tight_layout()
fig.align_ylabels()
# plt.savefig('./figures/state_1_nl.pdf', format='pdf', dpi=1000, bbox_inches = 'tight')
# %%
"""
controls
"""
fig = plt.figure()
##
plt.subplot(3,1,1)
plt.plot(steps, control_plot[0])
# plt.xlabel('Time [s]',fontsize = font_size)
plt.ylabel(r'$T~\mathrm{[m \cdot s ^{-2}]}$',fontsize = font_size)
plt.xticks(fontsize = font_size)
plt.yticks(fontsize = font_size)
plt.xlim(0, tf+dt)
plt.grid(color='gray', linestyle=':')
##
plt.subplot(3,1,2)
plt.plot(steps, control_plot[1])
# plt.xlabel('Time [s]',fontsize = font_size)
plt.ylabel(r'$\phi_\mathrm{ref}~\mathrm{[deg]}$',fontsize = font_size)
plt.xticks(fontsize = font_size)
plt.yticks(fontsize = font_size)
plt.xlim(0, tf+dt)
plt.grid(axis='both',color='gray', linestyle=':')
##
plt.subplot(3,1,3)
plt.plot(steps, control_plot[2])
plt.xlabel('Time [s]',fontsize = font_size)
plt.ylabel(r'$\theta_\mathrm{ref}~\mathrm{[deg]}$',fontsize = font_size)
plt.xticks(fontsize = font_size)
plt.yticks(fontsize = font_size)
plt.xlim(0, tf+dt)
plt.grid(axis='both',color='gray', linestyle=':')

plt.tight_layout()
fig.align_ylabels()

plt.show()

