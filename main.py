import cv2
import os
import numpy as np
from filters import apply_filter, apply_filter_DFT

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
    
    # Criar diretório para imagens processadas se não existir
    output_dir = 'processed_images'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== Aplicando Filtros ===\n")
    
    # Lista de operações a serem executadas
    operations = [
        {
            'name': 'low_pass_size3',
            'title': 'Passa-Baixas (size=3)',
            'func': lambda: apply_filter(img_gray, size=3, filter_type='low_pass')
        },
        {
            'name': 'low_pass_size7',
            'title': 'Passa-Baixas (size=7)',
            'func': lambda: apply_filter(img_gray, size=7, filter_type='low_pass')
        },
        {
            'name': 'high_pass_size3',
            'title': 'Passa-Altas (size=3)',
            'func': lambda: apply_filter(img_gray, size=3, filter_type='high_pass')
        },
        {
            'name': 'high_pass_size7',
            'title': 'Passa-Altas (size=7)',
            'func': lambda: apply_filter(img_gray, size=7, filter_type='high_pass')
        },
        {
            'name': 'dft_low_pass_wc_pi2_both',
            'title': 'DFT Passa-Baixas (wc=pi/2, both)',
            'func': lambda: apply_filter_DFT(img_gray, wc=np.pi/2, direction='both', filter_type='low_pass')
        },
        {
            'name': 'dft_low_pass_wc_pi4_both',
            'title': 'DFT Passa-Baixas (wc=pi/4, both)',
            'func': lambda: apply_filter_DFT(img_gray, wc=np.pi/4, direction='both', filter_type='low_pass')
        },
        {
            'name': 'dft_high_pass_wc_pi2_both',
            'title': 'DFT Passa-Altas (wc=pi/2, both)',
            'func': lambda: apply_filter_DFT(img_gray, wc=np.pi/2, direction='both', filter_type='high_pass')
        },
        {
            'name': 'dft_high_pass_wc_pi4_both',
            'title': 'DFT Passa-Altas (wc=pi/4, both)',
            'func': lambda: apply_filter_DFT(img_gray, wc=np.pi/4, direction='both', filter_type='high_pass')
        },
        {
            'name': 'dft_low_pass_wc_pi8_horizontal',
            'title': 'DFT Passa-Baixas (wc=pi/8, horizontal)',
            'func': lambda: apply_filter_DFT(img_gray, wc=np.pi/8, direction='horizontal', filter_type='low_pass')
        },
        {
            'name': 'dft_high_pass_wc_pi8_horizontal',
            'title': 'DFT Passa-Altas (wc=pi/8, horizontal)',
            'func': lambda: apply_filter_DFT(img_gray, wc=np.pi/8, direction='horizontal', filter_type='high_pass')
        }
    ]
    
    # Processar cada operação
    for i, op in enumerate(operations, 1):
        print(f"{i}. Aplicando {op['title']}...")
        filtered_img = op['func']()
        
        # Criar imagem lado a lado (original à esquerda | filtrada à direita)
        comparison = np.hstack([img_gray, filtered_img])
        
        # Salvar imagem com original à esquerda e filtrada à direita
        output_path = os.path.join(output_dir, f"{op['name']}.jpg")
        cv2.imwrite(output_path, comparison)
        print(f"   Salvo: {output_path}")
        
        # Exibir comparação
        cv2.imshow(f"{i}. {op['title']} - Original | Filtrada", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\n=== Processamento concluído! ===")
    print(f"Todas as imagens foram salvas em '{output_dir}'")

