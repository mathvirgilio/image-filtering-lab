# Image Filtering Lab

Laboratório de processamento de imagens que implementa filtros passa-baixas e passa-altas tanto no domínio espacial quanto no domínio da frequência (DFT).

## 📋 Descrição

Este projeto demonstra diferentes técnicas de filtragem de imagens:

- **Filtros Espaciais**: Filtros passa-baixas e passa-altas aplicados via convolução 2D
- **Filtros DFT**: Filtros ideais no domínio da frequência com diferentes frequências de corte e direções

## 🚀 Requisitos

- Python 3.8+
- OpenCV (opencv-python)
- NumPy

## 📦 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/mathvirgilio/image-filtering-lab.git
cd image-filtering-lab
```

2. Crie um ambiente virtual (recomendado):
```bash
python -m venv image_filtering_lab
```

3. Ative o ambiente virtual:
   - Windows:
     ```bash
     image_filtering_lab\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source image_filtering_lab/bin/activate
     ```

4. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 🎯 Uso

Execute o script principal:

```bash
python main.py
```

O script irá:
1. Carregar a imagem de `images/cameraman.jpg`
2. Aplicar diversos filtros (passa-baixas e passa-altas com diferentes parâmetros)
3. Exibir comparações lado a lado (original | filtrada)
4. Salvar todas as imagens processadas em `processed_images/`

### Filtros Aplicados

O script aplica os seguintes filtros:

**Filtros Espaciais:**
- Passa-baixas (size=3)
- Passa-baixas (size=7)
- Passa-altas (size=3)
- Passa-altas (size=7)

**Filtros DFT:**
- Passa-baixas (wc=π/2, ambas direções)
- Passa-baixas (wc=π/4, ambas direções)
- Passa-altas (wc=π/2, ambas direções)
- Passa-altas (wc=π/4, ambas direções)
- Passa-baixas (wc=π/8, horizontal)
- Passa-altas (wc=π/8, horizontal)

## 📁 Estrutura do Projeto

```
image-filtering-lab/
├── filters.py              # Implementação dos filtros
├── main.py                 # Script principal
├── requirements.txt        # Dependências do projeto
├── images/                 # Imagens de entrada
│   ├── cameraman.jpg
│   ├── lena.jpg
│   ├── lena.gif
│   └── fabio.png
└── processed_images/       # Imagens processadas (geradas automaticamente)
```

## 🔧 Funções Principais

### `filters.py`

- **`h(size)`**: Cria um kernel de filtro de média (passa-baixas) de tamanho `size×size`
- **`apply_filter(image, size, filter_type)`**: Aplica filtro espacial passa-baixas ou passa-altas
- **`apply_filter_DFT(image, wc, direction, filter_type)`**: Aplica filtro ideal no domínio da frequência

### Parâmetros dos Filtros DFT

- **`wc`**: Frequência de corte em radianos (ex: π/2, π/4, π/8)
- **`direction`**: Direção do filtro
  - `'both'`: Aplica em ambas as direções (horizontal e vertical)
  - `'horizontal'`: Aplica apenas na direção horizontal
  - `'vertical'`: Aplica apenas na direção vertical
- **`filter_type`**: Tipo de filtro
  - `'low_pass'`: Filtro passa-baixas (remove altas frequências)
  - `'high_pass'`: Filtro passa-altas (remove baixas frequências)

## 📝 Notas Técnicas

- Os filtros passa-altas adicionam um offset de 128 para representar valores negativos (zero = cinza médio)
- Os valores são normalizados para o range [0, 255] antes de salvar
- As imagens são convertidas para escala de cinza antes do processamento para melhor visualização dos efeitos dos filtros

## 🎓 Conceitos Demonstrados

- Convolução 2D no domínio espacial
- Transformada de Fourier Discreta (DFT) 2D
- Filtros ideais passa-baixas e passa-altas
- Processamento de imagens no domínio da frequência
- Filtragem direcional (horizontal/vertical)

## 📄 Licença

Este projeto é um laboratório educacional para demonstração de técnicas de processamento de imagens.
