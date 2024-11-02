import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import kerr_circular_orbits
from kerr_parameters import *

# ------------------------------------------------------------- PARÂMETROS DE TESTE -------------------------------------------------------------

# Definição de constantes
mu = -1
M = 1.0  # Massa do buraco negro
a = 0.5  # Fator de Kerr
Rs = 2 * M  # Raio de Schwarzschild

# Parâmetros de integração
lambda_ = 0.0
h = 0.1  # Passo de integração
n_steps = 20000

# ------------------------------------------------------------- CONDIÇÕES INICIAIS -------------------------------------------------------------

r0 = 4 * Rs
theta0 = np.pi / 2
phi0 = 0.0
t0 = 0.0

E_base = kerr_circular_orbits.equatorial_energy(r0, M, a, 1)
Lz_base = kerr_circular_orbits.equatorial_angular_momentum(r0, M, a, 1)
Q = 0

# Definir as perturbações
perturbations = [0.001, 0.01, 0.0]  # 0.1%, 1%, 0% de perturbação
perturbation_labels = {
    0.0: 'Sem Perturbação',
    0.001: 'Perturbação de 0.1%',
    0.01: 'Perturbação de 1%',
}
colors = {0.0: 'b', 0.001: 'r', 0.01: 'g'}
linestyles = {0.0: '-', 0.001: '--', 0.01: '-.'}
alphas = {0.0: 1.0, 0.001: 0.7, 0.01: 0.7}

# ------------------------------------------------------------- FUNÇÕES NECESSÁRIAS -------------------------------------------------------------

# Sistema de equações diferenciais
def derivatives(lambda_, state, E, Lz):
    r, theta, phi, t, ur, utheta = state

    d_r = Delta(r, M, a) / Sigma(r, theta, a) * ur
    d_theta = 1 / Sigma(r, theta, a) * utheta
    d_phi = (Sigma(r, theta, a) * (E * np.sin(theta)**2 * Rs * a * r +
               Lz * np.cos(theta)**2 * a**2 - Lz * Rs * r + Lz * r**2)) / (
               np.sin(theta)**2 * (Rs**2 * a**2 * r**2 +
               A(r, theta, M, a) * np.cos(theta)**2 * a**2 -
               A(r, theta, M, a) * Rs * r + A(r, theta, M, a) * r**2))
    d_t = (Sigma(r, theta, a) * (-Lz * Rs * a * r + A(r, theta, M, a) * E)) / (
           (Rs**2 * a**2 * r**2 * np.sin(theta)**2 +
          A(r, theta, M, a) * np.cos(theta)**2 * a**2 -
          A(r, theta, M, a) * Rs * r + A(r, theta, M, a) * r**2))

    d_ur = (mu * r / Sigma(r, theta, a) -
            (2 * r - Rs) * ur**2 / (2 * Sigma(r, theta, a)) -
            (1 / (2 * Delta(r, M, a)**2 * Sigma(r, theta, a))) *
            ((2 * E * Lz * Rs * a - E**2 * Rs * a**2 -
              2 * E ** 2 * a ** 2 * r - 4 * E**2 * r**3) * Delta(r, M, a) -
             (2 * E * Lz * Rs * a * r - Lz**2 * a**2 -
              E**2 * Rs * a**2 * r - E ** 2 * a ** 2 * r ** 2 - E**2 * r**4) * (2 * r - Rs)))

    d_utheta = (np.cos(theta) * np.sin(theta) / Sigma(r, theta, a)) * (
                (Lz**2 / np.sin(theta)**4) - a**2 * (E**2 + mu))

    return np.array([d_r, d_theta, d_phi, d_t, d_ur, d_utheta])

# Método de Runge-Kutta de quarta ordem
def runge_kutta_step(state, lambda_, h, E, Lz):
    k1 = derivatives(lambda_, state, E, Lz)
    k2 = derivatives(lambda_ + 0.5 * h, state + 0.5 * h * k1, E, Lz)
    k3 = derivatives(lambda_ + 0.5 * h, state + 0.5 * h * k2, E, Lz)
    k4 = derivatives(lambda_ + h, state + h * k3, E, Lz)

    return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Conversão para coordenadas "cartesianas" (Boyer-Lindquist)
def to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# Conversão para coordenadas cartesianas tradicionais
# def to_cartesian(r, theta, phi):
#     x = (r**2 + a**2) * np.sin(theta) * np.cos(phi)
#     y = (r**2 + a**2) * np.sin(theta) * np.sin(phi)
#     z = r * np.cos(theta)
#     return x, y, z

# ------------------------------------------------------------- INTEGRAÇÃO -------------------------------------------------------------

results = {}

for pert in perturbations:
    # Ajustar E e Lz com base na perturbação
    E = E_base * (1 + pert)
    Lz = Lz_base * (1 + pert)
    Q = 0  # Mantemos Q constante

    # Cálculo das velocidades iniciais
    u_r0 = u_r(Lz, E, a, mu, Q, r0, Rs)[1]  # escolher sinal da velocidade inicial
    u_theta0 = u_theta(Lz, E, a, mu, Q, theta0)[0]

    ur0 = u_r0 * Sigma(r0, theta0, a) / Delta(r0, M, a)
    utheta0 = u_theta0 * Sigma(r0, theta0, a)
    print(f'Perturbação: {pert*100}%, Energia: {E}')
    print(f'Velocidade radial inicial: {ur0}, Velocidade polar inicial: {utheta0}')

    state = np.array([r0, theta0, phi0, t0, u_r0, u_theta0])

    # Loop de integração com controle adaptativo do passo h
    trajectory = []
    lambdas = []
    lambda_local = lambda_
    for i in range(n_steps):
        trajectory.append(state.copy())
        lambdas.append(lambda_local)

        # Armazene o estado anterior para comparar
        state_anterior = state.copy()

        # Realiza o passo de integração
        state = runge_kutta_step(state, lambda_local, h, E, Lz)

        lambda_local += h

    trajectory = np.array(trajectory)

    # Armazenar resultados
    results[pert] = {
        'trajectory': trajectory,
        'lambdas': lambdas,
    }

# ------------------------ Gráficos de t, r, theta, phi vs lambda ------------------------

fig2, axs = plt.subplots(3, 1, figsize=(12, 12))

for pert in perturbations:
    trajectory = results[pert]['trajectory']
    lambdas = results[pert]['lambdas']

    r_vals = trajectory[:, 0]
    theta_vals = trajectory[:, 1]
    phi_vals = trajectory[:, 2]
    t_vals = trajectory[:, 3]

    label = perturbation_labels[pert]
    color = colors[pert]
    linestyle = linestyles[pert]

    axs[0].plot(lambdas, r_vals, label=f'r($\\lambda$) - {label}', color=color, linestyle=linestyle)
    axs[1].plot(lambdas, phi_vals, label=f'ϕ($\\lambda$) - {label}', color=color, linestyle=linestyle)
    axs[2].plot(lambdas, t_vals, label=f't($\\lambda$) - {label}', color=color, linestyle=linestyle)

# Ajustar eixos e legendas
for ax in axs:
    ax.set_xlabel(r'$\lambda /R_s$')
    ax.legend()
    ax.grid()

axs[0].set_ylabel(r'$r/R_s$')
axs[1].set_ylabel('ϕ (rad)')
axs[2].set_ylabel(r'$t/R_s$')

plt.tight_layout()
plt.show()

# ------------------------ Gráficos de r(t), theta(t), phi(t) ------------------------

import matplotlib.gridspec as gridspec

# Criando um GridSpec para controlar as proporções dos subplots
fig3 = plt.figure(figsize=(12, 12))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])  # Proporções iguais

# Subplots individuais
axs0 = fig3.add_subplot(gs[0])
axs1 = fig3.add_subplot(gs[1])
axs2 = fig3.add_subplot(gs[2])

for pert in perturbations:
    trajectory = results[pert]['trajectory']
    t_vals = trajectory[:, 3]
    r_vals = trajectory[:, 0]
    theta_vals = trajectory[:, 1]
    phi_vals = trajectory[:, 2]

    label = perturbation_labels[pert]
    color = colors[pert]
    linestyle = linestyles[pert]

    # Gráfico de r(t)
    axs0.plot(t_vals, r_vals, label=f'r(t) - {label}', color=color, linestyle=linestyle)
    axs0.set_xlabel(r'$t/R_s$')
    axs0.set_ylabel(r'$r/R_s$')
    axs0.legend()
    axs0.grid()

    # Gráfico de theta(t)
    axs1.plot(t_vals, theta_vals, label=f'θ(t) - {label}', color=color, linestyle=linestyle)
    axs1.set_xlabel(r'$t/R_s$')
    axs1.set_ylabel('θ (rad)')
    axs1.legend()
    axs1.grid()

    # Gráfico de phi(t)
    axs2.plot(t_vals, phi_vals, label=f'ϕ(t) - {label}', color=color, linestyle=linestyle)
    axs2.set_xlabel(r'$t/R_s$')
    axs2.set_ylabel('ϕ (rad)')
    axs2.legend()
    axs2.grid()

# Ajustar espaçamento entre os subplots
plt.tight_layout(pad=3.0)
plt.show()

# ------------------------ Gráfico de r(t) ------------------------

fig3, axs = plt.subplots(1, 1, figsize=(12, 4))

for pert in perturbations:
    trajectory = results[pert]['trajectory']
    t_vals = trajectory[:, 3]
    r_vals = trajectory[:, 0]

    label = perturbation_labels[pert]
    color = colors[pert]
    linestyle = linestyles[pert]

    # Gráfico de r(t)
    axs.plot(t_vals, r_vals, label=f'r(t) - {label}', color=color, linestyle=linestyle)
    axs.set_xlabel(r'$t/R_s$')
    axs.set_ylabel(r'$r/R_s$')
    axs.legend()
    axs.grid()

# Ajustar o layout
plt.tight_layout()
plt.show()

# ------------------------ Gráfico 3D das Trajetórias ------------------------

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])  # Define a proporção dos eixos como iguais
ax.set_xlim([-(r0*2), (r0*2)])
ax.set_ylim([-(r0*2), (r0*2)])
ax.set_zlim([-(r0*2), (r0*2)])

# Ajustar a visualização para uma vista de cima
ax.view_init(elev=90, azim=0)

# Desenhar horizontes de eventos e ergosfera
theta_vals_grid = np.linspace(0, np.pi, 100)
phi_vals_grid = np.linspace(0, 2 * np.pi, 100)
theta_grid, phi_grid = np.meshgrid(theta_vals_grid, phi_vals_grid)

# Horizonte de eventos externo
r_Hp = r_H_plus(M, a)
x_Hp = r_Hp * np.sin(theta_grid) * np.cos(phi_grid)
y_Hp = r_Hp * np.sin(theta_grid) * np.sin(phi_grid)
z_Hp = r_Hp * np.cos(theta_grid)
ax.plot_surface(x_Hp, y_Hp, z_Hp, color='#000000', alpha=0.6, label='Horizonte Externo')
print(f'Horizonte de Eventos Externo: {r_Hp}')

# Horizonte de eventos interno
r_Hm = r_H_minus(M, a)
x_Hm = r_Hm * np.sin(theta_grid) * np.cos(phi_grid)
y_Hm = r_Hm * np.sin(theta_grid) * np.sin(phi_grid)
z_Hm = r_Hm * np.cos(theta_grid)
ax.plot_surface(x_Hm, y_Hm, z_Hm, color='#000000', alpha=1.0, label='Horizonte Interno')
print(f'Horizonte de Eventos Interno: {r_Hm}')

# Ergosfera externa
r_Ep = r_E_plus(M, a, theta_grid)
x_Ep = r_Ep * np.sin(theta_grid) * np.cos(phi_grid)
y_Ep = r_Ep * np.sin(theta_grid) * np.sin(phi_grid)
z_Ep = r_Ep * np.cos(theta_grid)
ax.plot_surface(x_Ep, y_Ep, z_Ep, color='#FFFF00', alpha=0.3, label='Ergosfera')

# Plotar as trajetórias
# Primeiro, plotar as órbitas perturbadas
for pert in [0.001, 0.01]:
    trajectory = results[pert]['trajectory']
    r_vals = trajectory[:, 0]
    theta_vals = trajectory[:, 1]
    phi_vals = trajectory[:, 2]

    x_vals, y_vals, z_vals = [], [], []
    for i in range(len(r_vals)):
        r = r_vals[i]
        theta = theta_vals[i]
        phi = phi_vals[i]
        x, y, z = to_cartesian(r, theta, phi)
        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(z)

    label = perturbation_labels[pert]
    color = colors[pert]
    linestyle = linestyles[pert]
    alpha = alphas[pert]

    ax.plot(x_vals, y_vals, z_vals, label=f'Trajetória - {label}', color=color, linestyle=linestyle, alpha=alpha)

# Em seguida, plotar a órbita sem perturbação para que ela fique por cima
pert = 0.0
trajectory = results[pert]['trajectory']
r_vals = trajectory[:, 0]
theta_vals = trajectory[:, 1]
phi_vals = trajectory[:, 2]

x_vals, y_vals, z_vals = [], [], []
for i in range(len(r_vals)):
    r = r_vals[i]
    theta = theta_vals[i]
    phi = phi_vals[i]
    x, y, z = to_cartesian(r, theta, phi)
    x_vals.append(x)
    y_vals.append(y)
    z_vals.append(z)

label = perturbation_labels[pert]
color = 'k'  # Cor preta para destaque
linestyle = '-'
linewidth = 2  # Linha mais grossa
alpha = 1.0

ax.plot(x_vals, y_vals, z_vals, label=f'Trajetória - {label}', color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)

# Ajustar rótulos dos eixos
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.legend()
plt.show()
