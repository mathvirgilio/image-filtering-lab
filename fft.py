"""
Implementação de funções FFT para processamento de imagens.
Compatível com numpy.fft para ifftshift e ifft2.
"""

import numpy as np
from typing import Union, Optional


def ifft(x: np.ndarray, n: Optional[int] = None, axis: int = -1, norm: Optional[str] = None) -> np.ndarray:
    """
    Calcula a IFFT (Inverse Fast Fourier Transform) de um array.
    
    Compatível com numpy.fft.ifft
    
    Parâmetros:
    -----------
    x : array_like
        Array complexo de entrada
    n : int, opcional
        Tamanho da IFFT
    axis : int, opcional
        Eixo ao longo do qual calcular a IFFT
    norm : str, opcional
        Normalização: 'ortho' para normalização ortogonal, None para padrão
        
    Retorna:
    --------
    out : ndarray
        Array complexo com a IFFT do sinal de entrada
    """
    x = np.asarray(x)
    
    if n is None:
        n = x.shape[axis] if axis >= 0 else x.shape[x.ndim + axis]
    
    # IFFT é calculada como: IFFT(x) = conj(FFT(conj(x))) / n
    x_conj = np.conj(x)
    
    # Prepara o array similar à FFT
    if axis == -1 or axis == x.ndim - 1:
        if x_conj.shape[-1] < n:
            pad_shape = list(x_conj.shape)
            pad_shape[-1] = n - x_conj.shape[-1]
            x_conj = np.concatenate([x_conj, np.zeros(pad_shape, dtype=x_conj.dtype)], axis=-1)
        elif x_conj.shape[-1] > n:
            x_conj = x_conj[..., :n]
    else:
        x_conj = np.moveaxis(x_conj, axis, -1)
        if x_conj.shape[-1] < n:
            pad_shape = list(x_conj.shape)
            pad_shape[-1] = n - x_conj.shape[-1]
            x_conj = np.concatenate([x_conj, np.zeros(pad_shape, dtype=x_conj.dtype)], axis=-1)
        elif x_conj.shape[-1] > n:
            x_conj = x_conj[..., :n]
        x_conj = np.moveaxis(x_conj, -1, axis)
    
    if not np.iscomplexobj(x_conj):
        x_conj = x_conj.astype(np.complex128)
    
    # Calcula FFT do conjugado usando numpy (ou pode usar implementação própria)
    if axis == -1 or axis == x_conj.ndim - 1:
        result = np.fft.fft(x_conj, axis=axis)
    else:
        x_moved = np.moveaxis(x_conj, axis, -1)
        result_moved = np.fft.fft(x_moved, axis=-1)
        result = np.moveaxis(result_moved, -1, axis)
    
    # Aplica normalização
    if norm == 'ortho':
        result = np.conj(result) / np.sqrt(n)
    else:
        result = np.conj(result) / n
    
    return result


def ifftshift(x: np.ndarray, axes: Optional[Union[int, tuple]] = None) -> np.ndarray:
    """
    Inverte o deslocamento do zero de frequência aplicado por fftshift.
    
    Compatível com numpy.fft.ifftshift
    
    Parâmetros:
    -----------
    x : array_like
        Array de entrada
    axes : int ou tuple de ints, opcional
        Eixos ao longo dos quais deslocar. Se None, desloca todos os eixos.
        
    Retorna:
    --------
    y : ndarray
        Array com o zero de frequência deslocado de volta
    """
    x = np.asarray(x)
    
    if axes is None:
        axes = tuple(range(x.ndim))
    elif isinstance(axes, int):
        axes = (axes,)
    
    result = x.copy()
    
    for axis in axes:
        n = result.shape[axis]
        # ifftshift desloca de volta: move ceil(n/2) elementos para o início
        shift = (n + 1) // 2
        result = np.roll(result, shift, axis=axis)
    
    return result


def ifft2(a: np.ndarray, s: Optional[tuple] = None, axes: tuple = (-2, -1), norm: Optional[str] = None) -> np.ndarray:
    """
    Calcula a IFFT 2D (Inverse Fast Fourier Transform 2D) de um array.
    
    Compatível com numpy.fft.ifft2
    
    Parâmetros:
    -----------
    a : array_like
        Array de entrada (pode ser 2D ou multi-dimensional)
    s : tuple de ints, opcional
        Forma (shape) da saída. Se None, usa a forma de a ao longo dos eixos especificados
    axes : tuple de ints, opcional
        Eixos ao longo dos quais calcular a IFFT 2D. Padrão é (-2, -1)
    norm : str, opcional
        Normalização: 'ortho' para normalização ortogonal, None para padrão
        
    Retorna:
    --------
    out : ndarray
        Array complexo com a IFFT 2D
    """
    a = np.asarray(a)
    
    if s is None:
        s = (a.shape[axes[0]], a.shape[axes[1]])
    
    # Aplica IFFT ao longo do primeiro eixo, depois ao longo do segundo
    result = ifft(a, n=s[0], axis=axes[0], norm=norm)
    result = ifft(result, n=s[1], axis=axes[1], norm=norm)
    
    return result

