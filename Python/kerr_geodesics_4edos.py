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
# E = 0.935179  # Energia por unidade de massa
# Lz = 2.37176  # Momento angular por unidade de massa
Rs = 2 * M  # Raio de Schwarzschild

# Condições iniciais
r0 = 2 * Rs
theta0 = np.pi / 2
phi0 = 0.0
t0 = 0.0
u_r0 = 0.0
u_theta0 = 0.0

# Parâmetros de integração
lambda_ = 0.0
h = 0.01  # Passo de integração
n_steps = 20000

# Órbita circular para mu = -1 no equador
# mu = -1
# E = kerr_circular_orbits.equatorial_energy(r0, M, a, 1)
# Lz = kerr_circular_orbits.equatorial_angular_momentum(r0, M, a, 1)

# Órbita circular para mu = 0 no equador
# mu = 0
# possible_light_radii = kerr_circular_orbits.equatorial_light_radii(M, a)
# r0 = possible_light_radii[0]
# print(f"Raio da órbita circular do fóton: {r0}")
# Lz = kerr_circular_orbits.equatorial_photon_angular_momentum(r0, M, a, E)
# print(f"Momento angular da órbita circular do fóton {Lz} com energia {E}")

# Perturbações
# delta_r = 1.01  # Perturbação no raio inicial
# delta_ur = 1.01  # Perturbação na velocidade radial inicial

# Aplicar perturbações nas condições iniciais
# r0 = r0 * delta_r
# u_r0 = u_r0 * delta_ur

# Órbita polar de raio constante para mu = -1
# mu = -1
# u_theta0 = 1.0
# E = kerr_circular_orbits.polar_energy(r0, M, a) * 1.1
# Lz = 0.0

# Queda livre
# Lz = 0

# Trajetória não plana
# theta0 = np.pi / 4

# Teste de captura de partícula
# r0 = 8.0
# theta0 = np.pi / 2
# phi0 = 0.0
# t0 = 0.0
# u_r0 = -0.4
# u_theta0 = 1.0
# mu = -1
# E = kerr_circular_orbits.polar_energy(r0, M, a) * 1.1
# Lz = 0.0

# r0 = r_E_plus(M, a, theta0) # Fóton estático instável (L_z < 0)

delta_ur = 1.0

# ------------------------------------------------------------- ESCOLHA DE TESTE PELO USUÁRIO -------------------------------------------------------------

while(True):
    texto_input = """Escolha o teste a ser simulado inserindo um dos números abaixo:
    1. Órbita circular de uma partícula com massa no equador;
    2. Órbita circular de uma partícula sem massa no equador;
    3. Órbita polar com raio constante de uma partícula com massa;
    4. Queda livre de uma partícula com massa;
    5. Captura de uma partícula com massa;
    6. Trajetória interna ao horizonte externo.
    """
    teste_id = input(texto_input)

    if teste_id == '1':
        r0 = 2 * Rs
        theta0 = np.pi / 2
        phi0 = 0.0
        t0 = 0.0
        u_r0 = 0.0
        u_theta0 = 0.0
        mu = -1
        E = kerr_circular_orbits.equatorial_energy(r0, M, a, 1)
        Lz = kerr_circular_orbits.equatorial_angular_momentum(r0, M, a, 1)

        while(True):
            texto_input2 = """Incluir perturbação? Insira um dos números abaixo
            1. Sim;
            2. Não.
            """

            teste_id2 = input(texto_input2)

            if teste_id2 == '1':
                # Perturbações
                delta_r = 1.01  # Perturbação no raio inicial
                delta_ur = 1.01  # Perturbação na velocidade radial inicial

                # Aplicar perturbações nas condições iniciais
                r0 = r0 * delta_r
                u_r0 = u_r0 * delta_ur
                break
            elif teste_id2 == '2':
                break
            else:
                pass
        break
    elif teste_id == '2':
        theta0 = np.pi / 2
        phi0 = 0.0
        t0 = 0.0
        u_r0 = 0.0
        u_theta0 = 0.0
        mu = 0
        possible_light_radii = kerr_circular_orbits.equatorial_light_radii(M, a)
        r0 = possible_light_radii[0]
        E = 0.935179
        print(f"Raio da órbita circular do fóton: {r0}")
        Lz = kerr_circular_orbits.equatorial_photon_angular_momentum(r0, M, a, E)
        print(f"Momento angular da órbita circular do fóton {Lz} com energia {E}")
        
        while(True):
            texto_input2 = """Incluir perturbação? Insira um dos números abaixo
            1. Sim;
            2. Não.
            """

            teste_id2 = input(texto_input2)

            if teste_id2 == '1':
                # Perturbações
                delta_r = 1.01  # Perturbação no raio inicial
                delta_ur = 1.01  # Perturbação na velocidade radial inicial

                # Aplicar perturbações nas condições iniciais
                r0 = r0 * delta_r
                u_r0 = u_r0 * delta_ur
                break
            elif teste_id2 == '2':
                break
            else:
                pass
        break

    elif teste_id == '3':
        r0 = 2 * Rs
        theta0 = np.pi / 2
        phi0 = 0.0
        t0 = 0.0
        u_r0 = 0.0
        mu = -1
        u_theta0 = 1.0
        E = kerr_circular_orbits.polar_energy(r0, M, a)
        Lz = 0.0
        break

    elif teste_id == '4':
        r0 = 2 * Rs
        theta0 = np.pi / 2
        phi0 = 0.0
        t0 = 0.0
        u_r0 = 0.0
        mu = -1
        u_theta0 = 0.0
        E = kerr_circular_orbits.equatorial_energy(r0, M, a, 1)
        Lz = 0.0
        break
    
    elif teste_id == '5':
        h = 0.01
        n_steps = 800
        r0 = 2 * Rs
        theta0 = np.pi / 4
        phi0 = 0.0
        t0 = 0.0
        # u_r0 = 0.0
        # u_theta0 = 1.0
        mu = -1
        E = kerr_circular_orbits.polar_energy(r0, M, a)
        Lz = -0.1
        Q = kerr_circular_orbits.polar_carter_constant_Q(r0, M, a)
        break

    elif teste_id == '6':
        h = 0.01
        r0 = 0.5
        theta0 = np.pi / 2
        phi0 = 0.0
        t0 = 0.0
        u_r0 = 0.9
        u_theta0 = 0.0
        mu = -1
        E = -0.95
        Lz = 0.0
        break

    else:
        pass


# ------------------------------------------------------------- INTEGRAÇÃO -------------------------------------------------------------

u_r0 = u_r(Lz, E, a, mu, Q, r0, Rs)[1] * delta_ur # escolher sinal da velocidade inicial
# print(u_r0)
u_theta0 = u_theta(Lz, E, a, mu, Q, theta0)[0]
# print(u_theta0)

ur0 = u_r0 * Sigma(r0, theta0, a) / Delta(r0, M, a)
utheta0 = u_theta0 * Sigma(r0, theta0, a)
print(f'Velocidade radial inicial: {ur0}, Velocidade polar inicial: {utheta0}')

state = np.array([r0, theta0, phi0, t0])
print(state)

# Sistema de equações diferenciais
def derivatives(lambda_, state):
    r, theta, phi, t = state

    # Equação para d_theta
    d_theta = +1 * (1 / Sigma(r, theta, a)) * np.sqrt(
        Q - np.cos(theta)**2 * (-a**2 * (mu + E**2) + (Lz**2 / np.sin(theta)**2))
    )
    
    # Equação para d_r
    d_r = -1 * (1 / Sigma(r, theta, a)) * np.sqrt(
        E**2 * r**2 * (r**2 + a**2 + (Rs * a**2 / r)) - 2 * Rs * a * E * Lz * r - 
        Lz**2 * r**2 * (1 - Rs / r) - Delta(r, M, a) * (Q - mu * r**2)
    )

    d_phi = (Sigma(r, theta, a) * (E * np.sin(theta)**2 * Rs * a * r + 
               Lz * np.cos(theta)**2 * a**2 - Lz * Rs * r + Lz * r**2)) / (
               np.sin(theta)**2 * (Rs**2 * a**2 * r**2 + 
               A(r, theta, M, a) * np.cos(theta)**2 * a**2 - 
               A(r, theta, M, a) * Rs * r + A(r, theta, M, a) * r**2))
    d_t = (Sigma(r, theta, a) * (-Lz * Rs * a * r + A(r, theta, M, a) * E)) / (
           (Rs**2 * a**2 * r**2 * np.sin(theta)**2 + 
          A(r, theta, M, a) * np.cos(theta)**2 * a**2 - 
          A(r, theta, M, a) * Rs * r + A(r, theta, M, a) * r**2))

    return np.array([d_r, d_theta, d_phi, d_t])


# Método de Runge-Kutta de quarta ordem
def runge_kutta_step(state, lambda_, h):
    k1 = derivatives(lambda_, state)
    k2 = derivatives(lambda_ + 0.5 * h, state + 0.5 * h * k1)
    k3 = derivatives(lambda_ + 0.5 * h, state + 0.5 * h * k2)
    k4 = derivatives(lambda_ + h, state + h * k3)

    return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Conversão para coordenadas cartesianas
def to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# Definir uma tolerância para a variação de t
t_tolerancia = 0.05  # Ajuste conforme necessário

# Função para calcular a taxa de variação de t e ajustar o passo de integração h
def ajustar_passo(state_atual, state_anterior, h_atual, limite):
    # print(h)
    t_atual = state_atual[3]
    t_anterior = state_anterior[3]
    variacao_t = abs(t_atual - t_anterior)
    
    # Se a variação de t ultrapassar o limite, reduz o passo
    if variacao_t > limite:
        h_atual *= 0.9  # Diminui o passo
    
    return h_atual

# Loop de integração com controle adaptativo do passo h
trajectory = []
lambdas = []
for i in range(n_steps):
    trajectory.append(state)
    lambdas.append(lambda_)
    
    # Armazene o estado anterior para comparar
    state_anterior = state.copy()
    
    # Realiza o passo de integração
    state = runge_kutta_step(state, lambda_, h)
    
    # Ajusta o passo se necessário
    h = ajustar_passo(state, state_anterior, h, t_tolerancia)
    
    lambda_ += h

trajectory = np.array(trajectory)

# Converter para coordenadas cartesianas para visualização
x_vals, y_vals, z_vals = [], [], []
for state in trajectory:
    r, theta, phi = state[0], state[1], state[2]
    x, y, z = to_cartesian(r, theta, phi)
    x_vals.append(x)
    y_vals.append(y)
    z_vals.append(z)

# Separar variáveis para plotar
r_vals = trajectory[:, 0]
theta_vals = trajectory[:, 1]
phi_vals = trajectory[:, 2]
t_vals = trajectory[:, 3]

# ------------------------ Gráficos de t, r, theta, phi vs lambda ------------------------

# Plotar t, r, theta, phi em função de lambda
fig2, axs = plt.subplots(4, 1, figsize=(12, 16))

axs[0].plot(lambdas, r_vals, label='r(λ)', color='b')
axs[0].set_xlabel('λ')
axs[0].set_ylabel('r')
axs[0].legend()
axs[0].grid()

axs[1].plot(lambdas, theta_vals, label='θ(λ)', color='g')
axs[1].set_xlabel('λ')
axs[1].set_ylabel('θ (rad)')
axs[1].legend()
axs[1].grid()

axs[2].plot(lambdas, phi_vals, label='ϕ(λ)', color='r')
axs[2].set_xlabel('λ')
axs[2].set_ylabel('ϕ (rad)')
axs[2].legend()
axs[2].grid()

axs[3].plot(lambdas, t_vals, label='t(λ)', color='purple')
axs[3].set_xlabel('λ')
axs[3].set_ylabel('t')
axs[3].legend()
axs[3].grid()

plt.tight_layout()
plt.show()

# ------------------------ Gráficos de r(t), theta(t), phi(t) ------------------------

fig3, axs = plt.subplots(3, 1, figsize=(12, 12))

# Gráfico de r(t)
axs[0].plot(t_vals, r_vals, label='r(t)', color='b')
axs[0].set_xlabel('t')
axs[0].set_ylabel('r')
axs[0].legend()
axs[0].grid()

# Gráfico de theta(t)
axs[1].plot(t_vals, theta_vals, label='θ(t)', color='g')
axs[1].set_xlabel('t')
axs[1].set_ylabel('θ (rad)')
axs[1].legend()
axs[1].grid()

# Gráfico de phi(t)
axs[2].plot(t_vals, phi_vals, label='ϕ(t)', color='r')
axs[2].set_xlabel('t')
axs[2].set_ylabel('ϕ (rad)')
axs[2].legend()
axs[2].grid()

# Ajustar o layout
plt.tight_layout()
plt.show()

# ------------------------ Animação 3D ------------------------

# Animação com Matplotlib
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])  # Define a proporção dos eixos como iguais
ax.set_xlim([-(r0+1), (r0+1)])
ax.set_ylim([-(r0+1), (r0+1)])
ax.set_zlim([-(r0+1), (r0+1)])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Trajetória da Partícula na Métrica de Kerr')

# Desenhar horizontes de eventos e ergosfera
theta_vals_grid = np.linspace(0, np.pi, 100)
phi_vals_grid = np.linspace(0, 2 * np.pi, 100)
theta_grid, phi_grid = np.meshgrid(theta_vals_grid, phi_vals_grid)

# Horizonte de eventos externo
r_Hp = r_H_plus(M, a)
x_Hp = r_Hp * np.sin(theta_grid) * np.cos(phi_grid)
y_Hp = r_Hp * np.sin(theta_grid) * np.sin(phi_grid)
z_Hp = r_Hp * np.cos(theta_grid)
ax.plot_surface(x_Hp, y_Hp, z_Hp, color='#000000', alpha=0.8, label='Horizonte Externo')
print(f'Horizonte de Eventos Externo: {r_Hp}')

# Horizonte de eventos interno
r_Hm = r_H_minus(M, a)
x_Hm = r_Hm * np.sin(theta_grid) * np.cos(phi_grid)
y_Hm = r_Hm * np.sin(theta_grid) * np.sin(phi_grid)
z_Hm = r_Hm * np.cos(theta_grid)
ax.plot_surface(x_Hm, y_Hm, z_Hm, color='#000000', alpha=0.9, label='Horizonte Interno')
print(f'Horizonte de Eventos Interno: {r_Hm}')

# Ergosfera externa
r_Ep = r_E_plus(M, a, theta_grid)
x_Ep = r_Ep * np.sin(theta_grid) * np.cos(phi_grid)
y_Ep = r_Ep * np.sin(theta_grid) * np.sin(phi_grid)
z_Ep = r_Ep * np.cos(theta_grid)
ax.plot_surface(x_Ep, y_Ep, z_Ep, color='#FFFF00', alpha=0.3, label='Ergosfera Externa')

# Linha da trajetória da partícula
line, = ax.plot([], [], [], lw=2, color='b', label='Trajetória')

# Adicionar ponto da posição atual da partícula
current_point, = ax.plot([], [], [], 'ro', markersize=8, label='Posição Atual')  # 'ro' define um ponto vermelho

# Adicionar uma legenda manualmente
ax.legend(['Horizonte Externo', 'Ergosfera Externa', 'Trajetória'])

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    current_point.set_data([], [])
    current_point.set_3d_properties([])
    return line, current_point

def update(num):
    line.set_data(x_vals[:num], y_vals[:num])
    line.set_3d_properties(z_vals[:num])

    # Atualizar a posição atual da partícula (marcada por um ponto)
    current_point.set_data([x_vals[num]], [y_vals[num]])  # Ponto atual
    current_point.set_3d_properties([z_vals[num]])        # Ponto atual em Z

    return line, current_point

ani = FuncAnimation(fig, update, frames=n_steps, init_func=init, blit=True, interval=1, repeat=False)

# print(x_vals)
# print(t_vals)

plt.show()
