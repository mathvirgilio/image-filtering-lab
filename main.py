import cv2
import os
import numpy as np
from filters import apply_filter

# Caminho para a imagem
image_path = os.path.join('images', 'cameraman.jpg')

# Carregar a imagem
img = cv2.imread(image_path)

# Verificar se a imagem foi carregada corretamente
if img is None:
    print(f"Erro: Não foi possível carregar a imagem de {image_path}")
else:
    print(f"Imagem carregada com sucesso!")
    print(f"Dimensões: {img.shape}")
    
    # Converter para escala de cinza para melhor visualização dos filtros
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print("\n=== Testes de Filtros ===\n")
    
    # Teste 1: Filtro passa-baixas h3 (tamanho 3)
    print("1. Aplicando filtro passa-baixas h3...")
    low_pass_3 = apply_filter(img_gray, size=3, filter_type='low_pass')
    print(f"   Dimensões da imagem filtrada: {low_pass_3.shape}")
    
    # Teste 2: Filtro passa-baixas h7 (tamanho 7)
    print("2. Aplicando filtro passa-baixas h7...")
    low_pass_7 = apply_filter(img_gray, size=7, filter_type='low_pass')
    print(f"   Dimensões da imagem filtrada: {low_pass_7.shape}")
    
    # Teste 3: Filtro passa-altas h'3 (tamanho 3)
    print("3. Aplicando filtro passa-altas h'3...")
    high_pass_3 = apply_filter(img_gray, size=3, filter_type='high_pass')
    print(f"   Dimensões da imagem filtrada: {high_pass_3.shape}")
    
    # Teste 4: Filtro passa-altas h'7 (tamanho 7)
    print("4. Aplicando filtro passa-altas h'7...")
    high_pass_7 = apply_filter(img_gray, size=7, filter_type='high_pass')
    print(f"   Dimensões da imagem filtrada: {high_pass_7.shape}")
    
    print("\n=== Exibindo resultados ===\n")
    print("Pressione qualquer tecla para avançar entre as imagens...")
    
    # Exibir imagem original
    cv2.imshow('1. Imagem Original', img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Exibir filtros passa-baixas
    cv2.imshow('2. Passa-Baixas h3 (tamanho 3x3)', low_pass_3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('3. Passa-Baixas h7 (tamanho 7x7)', low_pass_7)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Exibir filtros passa-altas
    cv2.imshow('4. Passa-Altas h\'3 (tamanho 3x3)', high_pass_3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('5. Passa-Altas h\'7 (tamanho 7x7)', high_pass_7)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Opcional: Criar uma imagem comparativa com todas as versões
    print("\nCriando imagem comparativa...")
    
    # Redimensionar imagens para o mesmo tamanho (se necessário) e criar montagem
    h, w = img_gray.shape
    scale = 0.5  # Redimensionar para caber na tela
    
    # Redimensionar todas as imagens
    img_small = cv2.resize(img_gray, (int(w*scale), int(h*scale)))
    lp3_small = cv2.resize(low_pass_3, (int(w*scale), int(h*scale)))
    lp7_small = cv2.resize(low_pass_7, (int(w*scale), int(h*scale)))
    hp3_small = cv2.resize(high_pass_3, (int(w*scale), int(h*scale)))
    hp7_small = cv2.resize(high_pass_7, (int(w*scale), int(h*scale)))
    
    # Criar montagem: 2 linhas x 3 colunas
    top_row = np.hstack([img_small, lp3_small, lp7_small])
    bottom_row = np.hstack([img_small, hp3_small, hp7_small])
    comparison = np.vstack([top_row, bottom_row])
    
    # Adicionar textos (opcional, requer cv2.putText)
    cv2.imshow('Comparacao: Original | Passa-Baixas h3 | Passa-Baixas h7 | Original | Passa-Altas h\'3 | Passa-Altas h\'7', comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nTestes concluídos!")

