import numpy as np
import cv2


def h(size):
    """
    Cria um filtro de média (passa-baixas) genérico de tamanho size×size.
    
    Para h3: h(3) retorna (1/9) * [1 1 1]
                                   [1 1 1]
                                   [1 1 1]
    
    Para h7: h(7) retorna (1/49) * matriz 7×7 de uns
    
    Args:
        size (int): Tamanho do filtro (deve ser ímpar, ex: 3, 7, etc.)
    
    Returns:
        numpy.ndarray: Kernel do filtro normalizado (soma = 1)
    """
    if size % 2 == 0:
        raise ValueError("O tamanho do filtro deve ser ímpar")
    
    # Cria uma matriz de uns
    kernel = np.ones((size, size), dtype=np.float32)
    
    # Normaliza dividindo pelo número total de elementos
    kernel = kernel / (size * size)
    
    return kernel


def apply_filter(image, size, filter_type='low_pass'):
    """
    Aplica um filtro passa-baixas ou passa-altas à imagem usando o tamanho especificado.
    
    Para passa-altas: h'[n1, n2] = δ[n1, n2] – h_k[n1, n2]
    Onde h_k é o filtro passa-baixas e δ é o impulso unitário.
    
    Args:
        image (numpy.ndarray): Imagem de entrada (pode ser colorida ou em escala de cinza)
        size (int): Tamanho do filtro (ex: 3 para h3, 7 para h7)
        filter_type (str): Tipo de filtro - 'low_pass' ou 'high_pass' (padrão: 'low_pass')
    
    Returns:
        numpy.ndarray: Imagem filtrada
    """
    if filter_type == 'low_pass':
        kernel = h(size)
        filtered = cv2.filter2D(image, -1, kernel)
        return filtered
    elif filter_type == 'high_pass':
        # Cria o filtro passa-baixas
        low_pass_kernel = h(size)
        
        # Cria o impulso unitário (δ[n1, n2])
        impulse = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        impulse[center, center] = 1.0
        
        # Filtro passa-altas: h'[n1, n2] = δ[n1, n2] – h_k[n1, n2]
        high_pass_kernel = impulse - low_pass_kernel
        
        # Aplica o filtro
        filtered = cv2.filter2D(image, -1, high_pass_kernel)
        
        # Adiciona offset de 128 para imagens de 8 bits/pixel (zero = cinza médio)
        # Isso permite representar valores negativos resultantes do filtro
        filtered = filtered.astype(np.float32) + 128
        
        # Garante que os valores estejam no range válido [0, 255]
        filtered = np.clip(filtered, 0, 255)
        
        return filtered.astype(image.dtype)
    else:
        raise ValueError("filter_type deve ser 'low_pass' ou 'high_pass'")

