import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse
from scipy.sparse.linalg import spsolve
from matplotlib import animation

r = 0.1  # risk free rate
sig = 0.2  # volatility
S0 = 100  # initial stock price
X0 = np.log(S0)  # initial log stock price
K = 100  # strike price
T_expire = 1  # time to maturity

Nspace = 2000  # M space steps
Ntime = 3000  # N time steps

S_max = 3 * float(K)  # upper bound of the stock price
S_min = float(K) / 3  # lower bound of the stock price

x_max = np.log(S_max)  # A2
x_min = np.log(S_min)  # A1

x, dx = np.linspace(x_min, x_max, Nspace, retstep=True)  # space discretization
T, dt = np.linspace(0, T_expire, Ntime, retstep=True)  # time discretization

Payoff = np.maximum(np.exp(x) - K, 0)  # Call payoff
V = np.zeros((Nspace, Ntime))  # grid initialization
V[:, -1] = Payoff  # terminal conditions
V[-1, :] = np.exp(x_max) - K * np.exp(-r * T[::-1])  # boundary condition
V[0, :] = 0  # boundary condition

# construction of the tri-diagonal matrix D
sig2 = sig * sig
dxx = dx * dx

offset = np.zeros(Nspace - 2)  # vector to be used for the boundary terms
# a = ((dt / 2) * ((r - 0.5 * sig2) / dx - sig2 / dxx))
# b = (1 + dt * (sig2 / dxx + r))
# c = (-(dt / 2) * ((r - 0.5 * sig2) / dx + sig2 / dxx))

a = -0.5 * sig2 * dt / dxx
b = 1 + dt * (r + (r * dx + sig2) / dxx)
c = (-dx * r - 0.5 * sig2) * dt / dxx


def create_tri_diag():
    n = Nspace - 2
    D = np.zeros((n, n))
    D[0, 0] = 1
    D[-1, -1] = 1
    for i in range(1, n - 1):
        D[i, i - 1] = a
        D[i, i] = b
        D[i, i + 1] = c
    return D


# Backward iteration for implicit scheme
A = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()
for i in range(Ntime - 2, -1, -1):
    offset[0] = a * V[0, i]  # boundary terms
    offset[-1] = c * V[-1, i]  # boundary terms
    # print(offset)
    V[1:-1, i] = spsolve(A, (V[1:-1, i + 1] - offset))  # solve the linear system
    # print(f"V[:, {i}] = i{V[:, i]}")

# Plotting

S = np.exp(x)
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

ax1.plot(S, Payoff, color='blue', label="Payoff")
ax1.plot(S, V[:, 0], color='red', label="BS curve")
ax1.set_xlim(60, 170);
ax1.set_ylim(0, 50)
ax1.set_xlabel("S");
ax1.set_ylabel("price")
ax1.legend(loc='upper left');
ax1.set_title("BS price at t=0")

X, Y = np.meshgrid(T, S)
print(V)
print(X)
print(Y)
ax2.plot_surface(Y, X, V, cmap=cm.ocean)
ax2.set_title("BS price surface")
ax2.set_xlabel("S");
ax2.set_ylabel("t");
ax2.set_zlabel("V")
ax2.view_init(30, -100)  # this function rotates the 3d plot
plt.show()

# animation
fig = plt.figure(figsize=(15, 6))
ax = fig.add_subplot(111)
ax.set_xlim(60, 170);
ax.set_ylim(0, 50)
ax.set_xlabel("S");
ax.set_ylabel("price")
ax.set_title("BS price at t=0")
line, = ax.plot([], [], color='red', label="BS curve")
line2, = ax.plot([], [], color='blue', label="Payoff")
ax.legend(loc='upper left');


def init():
    line.set_data([], [])
    line2.set_data([], [])

    return line, line2


def animate(i):
    line.set_data(S, V[:, i * 100])
    line2.set_data(S, Payoff)
    return line, line2


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=Ntime, interval=20, blit=True)

#save animation to gif file
anim.save('BS_animation.gif', writer='imagemagick', fps=60)

plt.show()
