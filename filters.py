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
    
    # Cria o filtro passa-baixas
    low_pass_kernel = h(size)
    
    if filter_type == 'low_pass':
        filtered = cv2.filter2D(image, -1, low_pass_kernel)
        return filtered
    elif filter_type == 'high_pass':
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


def apply_filter_DFT(image, wc=np.pi/2, direction='both', filter_type='low_pass'):
    """
    Aplica um filtro passa-baixas ou passa-altas ideal no domínio da DFT com corte em ωc.
    
    Filtro passa-baixas ideal:
    - Em ambas as direções: H(u,v) = 1 para |u| ≤ ωc e |v| ≤ ωc
    - Apenas horizontal: H(u,v) = 1 para |u| ≤ ωc
    - Apenas vertical: H(u,v) = 1 para |v| ≤ ωc
    
    Filtro passa-altas ideal:
    - Em ambas as direções: H(u,v) = 1 para |u| > ωc ou |v| > ωc
    - Apenas horizontal: H(u,v) = 1 para |u| > ωc
    - Apenas vertical: H(u,v) = 1 para |v| > ωc
    
    Args:
        image (numpy.ndarray): Imagem de entrada (pode ser colorida ou em escala de cinza)
        wc (float): Frequência de corte ωc em radianos (padrão: π/2)
        direction (str): Direção do filtro - 'both', 'horizontal' ou 'vertical' (padrão: 'both')
        filter_type (str): Tipo de filtro - 'low_pass' ou 'high_pass' (padrão: 'low_pass')
    
    Returns:
        numpy.ndarray: Imagem filtrada no domínio espacial
    """
    if direction not in ['both', 'horizontal', 'vertical']:
        raise ValueError("direction deve ser 'both', 'horizontal' ou 'vertical'")
    
    if filter_type not in ['low_pass', 'high_pass']:
        raise ValueError("filter_type deve ser 'low_pass' ou 'high_pass'")
    
    # Converter para float para evitar problemas de precisão
    img_float = image.astype(np.float32)
    
    # Se a imagem for colorida, processar cada canal separadamente
    if len(img_float.shape) == 3:
        channels = []
        for i in range(img_float.shape[2]):
            channel = img_float[:, :, i]
            filtered_channel = _apply_filter_DFT_single_channel(channel, wc, direction, filter_type)
            channels.append(filtered_channel)
        filtered = np.stack(channels, axis=2)
    else:
        filtered = _apply_filter_DFT_single_channel(img_float, wc, direction, filter_type)
    
    # Para filtros passa-altas, adicionar offset de 128 (zero = cinza médio)
    # Isso permite representar valores negativos resultantes do filtro
    if filter_type == 'high_pass':
        filtered = filtered.astype(np.float32) + 128
    
    # Garantir que os valores estejam no range válido
    filtered = np.clip(filtered, 0, 255)
    
    return filtered.astype(image.dtype)


def _apply_filter_DFT_single_channel(image, wc, direction='both', filter_type='low_pass'):
    """
    Aplica filtro passa-baixas ou passa-altas DFT em um único canal (função auxiliar).
    
    Args:
        image (numpy.ndarray): Canal da imagem (2D)
        wc (float): Frequência de corte ωc
        direction (str): Direção do filtro - 'both', 'horizontal' ou 'vertical'
        filter_type (str): Tipo de filtro - 'low_pass' ou 'high_pass'
    
    Returns:
        numpy.ndarray: Canal filtrado
    """
    M, N = image.shape
    
    # Calcular a DFT 2D
    F = np.fft.fft2(image)
    
    # Fazer fftshift para colocar a frequência zero no centro
    F_shifted = np.fft.fftshift(F)
    
    # Criar grades de frequências normalizadas
    # Frequências vão de -π a π após fftshift
    u = np.arange(-M//2, M//2) * (2 * np.pi / M)
    v = np.arange(-N//2, N//2) * (2 * np.pi / N)
    
    # Criar matrizes de frequências 2D
    U, V = np.meshgrid(v, u)  # U: horizontal, V: vertical
    
    # Criar máscara do filtro conforme a direção e tipo
    if filter_type == 'low_pass':
        # Filtro passa-baixas ideal
        if direction == 'both':
            # H(u,v) = 1 para |u| ≤ ωc e |v| ≤ ωc, 0 caso contrário
            H = ((np.abs(U) <= wc) & (np.abs(V) <= wc)).astype(np.float32)
        elif direction == 'horizontal':
            # H(u,v) = 1 para |u| ≤ ωc (apenas filtro horizontal)
            H = (np.abs(U) <= wc).astype(np.float32)
        elif direction == 'vertical':
            # H(u,v) = 1 para |v| ≤ ωc (apenas filtro vertical)
            H = (np.abs(V) <= wc).astype(np.float32)
    else:  # filter_type == 'high_pass'
        # Filtro passa-altas ideal (inverso do passa-baixas)
        if direction == 'both':
            # H(u,v) = 1 para |u| > ωc ou |v| > ωc, 0 caso contrário
            H = ((np.abs(U) > wc) | (np.abs(V) > wc)).astype(np.float32)
        elif direction == 'horizontal':
            # H(u,v) = 1 para |u| > ωc (apenas filtro horizontal)
            H = (np.abs(U) > wc).astype(np.float32)
        elif direction == 'vertical':
            # H(u,v) = 1 para |v| > ωc (apenas filtro vertical)
            H = (np.abs(V) > wc).astype(np.float32)
    
    # Aplicar o filtro multiplicando no domínio da frequência
    G_shifted = F_shifted * H
    
    # Fazer ifftshift para voltar à ordem original
    G = np.fft.ifftshift(G_shifted)
    
    # Calcular a IDFT para voltar ao domínio espacial
    g = np.fft.ifft2(G)
    
    # Pegar apenas a parte real (a parte imaginária deve ser ~0)
    filtered = np.real(g)
    
    return filtered

