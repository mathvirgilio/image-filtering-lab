import cv2
import os

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
    
    # Exibir a imagem
    cv2.imshow('Imagem', img)
    
    # Aguardar até que uma tecla seja pressionada
    cv2.waitKey(0)
    
    # Fechar todas as janelas
    cv2.destroyAllWindows()

