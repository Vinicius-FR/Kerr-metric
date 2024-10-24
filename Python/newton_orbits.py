import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constantes adimensionais
G = 1.0  # Constante gravitacional
M = 1.0  # Massa do corpo central

# Parâmetros da partícula
E = -0.1  # Energia da partícula (definida para uma órbita elíptica)
L = 0.975   # Momento angular da partícula

# Funções auxiliares
def radial_acceleration(r):
    """Aceleração radial devido à gravidade"""
    return -G * M / r**2

def vr_initial(E, L, r):
    """Calcula a velocidade radial inicial a partir da energia e momento angular"""
    return np.sqrt(2 * (E + 1/r) - (L/r)**2)

def rk4_step(r, vr, theta, dt):
    """Executa um passo do método de Runge-Kutta de quarta ordem para r e theta"""
    # Aceleração radial
    k1_r = vr
    k1_vr = radial_acceleration(r) + L**2 / r**3

    k2_r = vr + 0.5 * dt * k1_vr
    k2_vr = radial_acceleration(r + 0.5 * dt * k1_r) + L**2 / (r + 0.5 * dt * k1_r)**3

    k3_r = vr + 0.5 * dt * k2_vr
    k3_vr = radial_acceleration(r + 0.5 * dt * k2_r) + L**2 / (r + 0.5 * dt * k2_r)**3

    k4_r = vr + dt * k3_vr
    k4_vr = radial_acceleration(r + dt * k3_r) + L**2 / (r + dt * k3_r)**3

    r_new = r + dt * (k1_r + 2 * k2_r + 2 * k3_r + k4_r) / 6
    vr_new = vr + dt * (k1_vr + 2 * k2_vr + 2 * k3_vr + k4_vr) / 6

    # Atualizando o ângulo theta
    theta_new = theta + dt * L / r**2

    return r_new, vr_new, theta_new

# Condições iniciais
r0 = 1.0  # Raio inicial
theta0 = 0.0  # Ângulo inicial
vr0 = vr_initial(E, L, r0)  # Velocidade radial inicial

# Parâmetros da simulação
dt = 0.1  # Intervalo de tempo
total_time = 100.0  # Tempo total de simulação

# Listas para armazenar os resultados
r_vals = []
theta_vals = []

# Inicializando valores de r, vr e theta
r = r0
vr = vr0
theta = theta0

# Loop de simulação
for t in np.arange(0, total_time, dt):
    r_vals.append(r)
    theta_vals.append(theta)
    r, vr, theta = rk4_step(r, vr, theta, dt)

# Convertendo para coordenadas cartesianas para plotar
x_vals = np.array(r_vals) * np.cos(theta_vals)
y_vals = np.array(r_vals) * np.sin(theta_vals)

# Configuração da figura e dos eixos
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.set_xlim(-6, 6)
ax.set_ylim(-2, 10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Trajetória Newtoniana')
ax.grid(True)

# Desenhar o corpo central
central_body, = ax.plot(0, 0, 'ko', label='Corpo fixo')
# Linha da trajetória
trajetoria, = ax.plot([], [], lw=2, color='b', label='Trajetória')
# Ponto representando o corpo teste
ponto, = ax.plot([], [], 'ro', label='Corpo teste')

# Função de inicialização para a animação
def init():
    trajetoria.set_data([], [])
    ponto.set_data([], [])
    return trajetoria, ponto

# Função de atualização para cada quadro da animação
def update(frame):
    trajetoria.set_data(x_vals[:frame], y_vals[:frame])  # Trajetória completa até o quadro atual
    ponto.set_data([x_vals[frame]], [y_vals[frame]])  # Garantindo que os valores são sequências
    return trajetoria, ponto

# Criando a animação
ani = FuncAnimation(fig, update, frames=len(x_vals), init_func=init, blit=True, interval=30)

# Adicionar a legenda com os parâmetros em LaTeX
# legend_text = r'$M = {:.1f}, E = {:.1f}, L = {:.3f}, (r_0, \theta_0) = ({:.1f}, {:.1f})$'.format(M, E, L, r0, theta0)
# plt.legend(title=legend_text)

plt.legend()

plt.show()
