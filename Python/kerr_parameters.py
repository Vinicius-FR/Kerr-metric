import numpy as np

# Funções para Delta, Sigma, A
def Delta(r, M, a):
    return r**2 - 2 * M * r + a**2

def Sigma(r, theta, a):
    return r**2 + a**2 * np.cos(theta)**2

def A(r, theta, M, a):
    return (r**2 + a**2)**2 - a**2 * Delta(r, M, a) * np.sin(theta)**2

# Horizontes de eventos e ergosfera
def r_H_plus(M, a):
    return M + np.sqrt(M**2 - a**2)

def r_H_minus(M, a):
    return M - np.sqrt(M**2 - a**2)

def r_E_plus(M, a, theta):
    return M + np.sqrt(M**2 - a**2 * np.cos(theta)**2)

def r_E_minus(M, a, theta):
    return M - np.sqrt(M**2 - a**2 * np.cos(theta)**2)

def carter_constant(utheta, theta, a, Lz, E, m0):
    term1 = utheta**2
    term2 = np.cos(theta)**2 * (-a**2 * (m0 + E**2) + (Lz / np.sin(theta))**2)

    Q = term1 + term2
    return Q

def u_theta(Lz, E, a, m_bar, Q, theta):
    # Constante de Carter
    kappa = Q + Lz**2

    # Termos da equação
    term1 = (Lz**2) / (np.sin(theta)**2)
    term2 = (E**2 + m_bar) * a**2 * np.cos(theta)**2
    # print(f'termo 1: {term1}')
    # print(f'termo 2: {term2}')
    
    # Resolver para u_theta^2
    u_theta_squared = kappa - term1 + term2
    u_theta_squared = np.round(u_theta_squared, 8)
    # print(u_theta_squared)
    
    # Retornar u_theta (positivo ou negativo)
    return np.sqrt(u_theta_squared), -np.sqrt(u_theta_squared)

def u_r(Lz, E, a, m_bar, Q, r, Rs):
    # Função Delta
    Delta = r**2 - Rs * r + a**2

    # Constante de Carter
    kappa = Q + Lz**2
    
    # Termo dentro da equação de u_r^2
    term1 = (2 * E * Lz * Rs * a * r - Lz**2 * a**2 - E**2 * Rs * a**2 * r - E**2 * a**2 * r**2 - E**2 * r**4) / Delta
    term2 = m_bar * r**2
    
    # Resolver para u_r^2
    u_r_squared = (-kappa - term1 + term2) / Delta
    u_r_squared = np.round(u_r_squared, 8)
    # print(f'u_r^2: {u_r_squared}')
    
    # Retornar u_r (positivo ou negativo)
    return np.sqrt(u_r_squared), -np.sqrt(u_r_squared)