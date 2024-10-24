import numpy as np

def equatorial_energy(r, M, a, sign):
    """
    Calcula a energia específica (E) de uma partícula com massa em órbita circular na métrica de Kerr.
    
    Args:
    r (float): Coordenada radial.
    Rs (float): Raio de Schwarzschild (2 * M, onde M é a massa do buraco negro).
    a (float): Parâmetro de rotação do buraco negro.
    sign (int): +1 para órbitas prógradas, -1 para órbitas retrógradas.
    
    Returns:
    float: Energia específica (E).
    """
    Rs = 2 * M
    sqrt_term = np.sqrt(Rs * r / 2)  # \sqrt{R_s r / 2}
    
    numerator = r**2 - Rs * r + sign * a * sqrt_term
    denominator = r * np.sqrt(r**2 - (3/2) * Rs * r + sign * 2 * a * sqrt_term)
    
    return numerator / denominator

def equatorial_angular_momentum(r, M, a, sign):
    """
    Calcula o momento angular específico (L_z) de uma partícula com massa em órbita circular na métrica de Kerr.
    
    Args:
    r (float): Coordenada radial.
    Rs (float): Raio de Schwarzschild (2 * M, onde M é a massa do buraco negro).
    a (float): Parâmetro de rotação do buraco negro.
    sign (int): +1 para órbitas prógradas, -1 para órbitas retrógradas.
    
    Returns:
    float: Momento angular específico (L_z).
    """
    Rs = 2 * M
    sqrt_term = np.sqrt(Rs * r / 2)  # \sqrt{R_s r / 2}
    
    numerator = sign * sqrt_term * (r**2 - sign * 2 * a * sqrt_term + a**2)
    denominator = r * np.sqrt(r**2 - (3/2) * Rs * r + sign * 2 * a * sqrt_term)
    
    return numerator / denominator

def equatorial_photon_angular_momentum(r, m, a, E):
    """
    Calcula o momento angular específico (L) de um fóton em órbita circular na métrica de Kerr.
    
    Args:
    r (float): Coordenada radial.
    m (float): Massa do buraco negro.
    a (float): Parâmetro de rotação do buraco negro.
    E (float): Energia do fóton.
    sign (int): +1 para órbitas prógradas, -1 para órbitas retrógradas.
    
    Returns:
    float: Momento angular específico (L).
    """

    numerator = - a * (r + 3 * m) * E
    denominator = r - 3 * m
    return numerator / denominator


# Testando as funções
# r = 10.0  # Exemplo de valor para r
# m = 1.0   # Massa do buraco negro
# a = 0.5   # Parâmetro de rotação do buraco negro

# # Energia e momento angular para órbitas prógradas (sign = +1) e retrógradas (sign = -1)
# E_prograde = energy(r, m, a, 1)
# L_prograde = angular_momentum(r, m, a, 1)

# E_retrograde = energy(r, m, a, -1)
# L_retrograde = angular_momentum(r, m, a, -1)

# print("Energia prógrada (E):", E_prograde)
# print("Momento angular prógrado (L):", L_prograde)
# print("Energia retrógrada (E):", E_retrograde)
# print("Momento angular retrógrado (L):", L_retrograde)

# m = 1.0  # Parâmetro m
# a = 1.0  # Parâmetro a

def equatorial_light_radii(m, a):

    # Coeficientes da equação cúbica: r^3 - 6*m*r^2 + 9*m^2*r - 4*m*a^2 = 0
    coefficients = [1, -6*m, 9*m**2, -4*m*a**2]

    # Usar numpy para encontrar todas as raízes (reais e complexas)
    all_roots = np.roots(coefficients)

    # Filtrar apenas as raízes reais
    real_roots = all_roots[np.isreal(all_roots)].real

    print("Raízes reais encontradas:", real_roots)

    return real_roots

# Exibir as raízes reais encontradas
# real_roots = equatorial_light_radii(m, a)
# print("Raízes reais encontradas:", real_roots)

def polar_energy(r, m, a):
    """
    Calcula a energia específica (E) de uma partícula em órbita circular na métrica de Kerr.
    
    Args:
    r (float): Coordenada radial.
    m (float): Massa do buraco negro.
    a (float): Parâmetro de rotação do buraco negro.
    
    Returns:
    float: Energia específica (E).
    """
    # Numerador da equação
    numerator = r * (r**2 - 2*m*r + a**2)**2

    # Denominador da equação
    denominator = (r**2 + a**2) * (r**3 - 3*m*r**2 + a**2*r + a**2*m)

    # Calcular E^2
    E_squared = numerator / denominator

    # Retornar E (a raiz quadrada de E^2)
    return np.sqrt(E_squared)

def polar_carter_constant_Q(r, m, a):
    """
    Calcula a constante de Carter Q para uma órbita polar.
    
    Args:
    r (float): Coordenada radial.
    m (float): Massa do buraco negro.
    a (float): Parâmetro de rotação do buraco negro.
    
    Returns:
    float: Constante de Carter (Q).
    """
    # Numerador da equação
    numerator = m * r**2 * (r**4 + 2 * a**2 * r**2 - 4 * m * a**2 * r + a**4)
    
    # Denominador da equação
    denominator = (r**2 + a**2) * (r**3 - 3 * m * r**2 + a**2 * r + a**2 * m)
    
    # Calcular Q
    Q = numerator / denominator
    
    return Q

# def equatorial_static_photon_angular_momentum(r, m, a, E):
#     """
#     Calcula o momento angular específico (L) de um fóton estático no plano equatorial na métrica de Kerr.
    
#     Args:
#     r (float): Coordenada radial.
#     m (float): Massa do buraco negro.
#     a (float): Parâmetro de rotação do buraco negro.
#     E (float): Energia do fóton.
#     sign (int): +1 para órbitas prógradas, -1 para órbitas retrógradas.
    
#     Returns:
#     float: Momento angular específico (L).
#     """

#     numerator = - 2 * m * a * r
#     denominator = 
#     return numerator / denominator