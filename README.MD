# 📌 Sprint 3 — IOT & IOB  
## Case 1: Apostas Compulsivas  

**Grupo:**  
- RM550161 - Eduardo Osorio Filho  
- RM550610 - Fabio Hideki Kamikihara  
- RM550260 - Pedro Moura Barros  
- RM98896 - Rodrigo Fernandes dos Santos  

---

## 🎯 Objetivo do Projeto

Este projeto faz parte da **Sprint 3 do curso de IOT & IOB** e tem como objetivo desenvolver uma **aplicação local (desktop/notebook) que realize detecção e reconhecimento facial de usuários** utilizando **OpenCV + Haar Cascade + LBPH**.

Esse módulo será um dos componentes da solução proposta para o **Case 1 — Apostas Compulsivas**, onde queremos:

- Detectar e identificar o usuário que está usando a aplicação
- Associar esse usuário a um nível de risco
- Permitir que o sistema principal possa aplicar restrições, enviar alertas e oferecer suporte psicológico quando um usuário for reconhecido como de risco

⚠️ **Importante:** Este módulo é independente e roda localmente, não havendo conexão com a aplicação final nesta sprint.

---

## ⚙️ Tecnologias Utilizadas

- **Python 3.13**
- **OpenCV (opencv-contrib-python)** — biblioteca para visão computacional
- **Haar Cascade** — detecção de rostos em tempo real
- **LBPHFaceRecognizer** — reconhecimento facial baseado em histogramas locais
- **NumPy** — manipulação de arrays e matrizes de imagem

---

## 📁 Estrutura do Projeto

```
SPRINT3-IOT-IOB/
├─ data/
│  ├─ dataset/           # imagens capturadas por usuário (treino)
│  └─ models/             # arquivos do modelo treinado (face_model.yml e labels.json)
├─ capture_faces.py       # captura imagens do rosto via webcam
├─ train_model.py         # treina o modelo usando as imagens capturadas
├─ recognize.py            # reconhece rostos em tempo real
├─ requirements.txt        # dependências do projeto
└─ README.md               # este arquivo
```

> ⚠️ Por questões de privacidade, os arquivos de imagens capturadas e os modelos treinados não estão no repositório público.

---

## 🖥️ Demonstração Visual Esperada

Durante a execução do `recognize.py`, a webcam exibe:

- Um **retângulo verde ao redor do rosto detectado**
- O **nome do usuário reconhecido**
- Um valor de **confiança (quanto menor, mais confiável)**

---

## 💻 Como Executar Localmente

### 1. Criar ambiente virtual e instalar dependências

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**requirements.txt**
```
opencv-contrib-python
numpy
```

---

### 2. Capturar imagens do usuário

```powershell
python capture_faces.py --id pedro --n 80
```

- Abre a webcam e captura 80 imagens do rosto
- Salva as imagens em `data/dataset/user_pedro/`
- Pressione `q` para sair antes, se desejar

---

### 3. Treinar o modelo

```powershell
python train_model.py
```

- Lê as imagens de `dataset/`
- Treina um modelo de reconhecimento facial
- Salva os arquivos `face_model.yml` e `labels.json` em `data/models/`

---

### 4. Executar o reconhecimento facial

```powershell
python recognize.py --threshold 60
```

- Abre a webcam e tenta reconhecer os rostos detectados
- Mostra o nome e nível de confiança do reconhecimento

---

## ⚙️ Parâmetros Importantes

- **capture_faces.py**
  - `scaleFactor` e `minNeighbors`: afetam a precisão e velocidade da detecção de rostos
- **train_model.py**
  - `radius`, `neighbors`, `grid_x`, `grid_y`: controlam a sensibilidade do LBPHFaceRecognizer
- **recognize.py**
  - `--threshold`: define o nível de confiança aceito (quanto **menor**, mais rigoroso)

💡 Exemplo:
```powershell
python recognize.py --threshold 40
```

---

## ⚠️ Limitações do Projeto

- Precisa de boa iluminação e rosto centralizado
- Funciona melhor com poucas pessoas conhecidas
- Reconhecimento sensível a mudanças bruscas (óculos, barba, cabelo, etc.)
- LBPH é simples e pode errar em cenários reais; ideal seria evoluir para redes neurais (ex: FaceNet, dlib)

---

## 📋 Nota Ética sobre o uso de dados faciais

Este projeto é **exclusivamente acadêmico** e não deve ser utilizado em ambientes de produção.  
Reconhecimento facial envolve **dados pessoais sensíveis** e só pode ser usado com **consentimento expresso do usuário**.  
Boas práticas obrigatórias:

- Garantir o consentimento informado do usuário antes de capturar imagens
- Armazenar os dados localmente e de forma segura
- Não compartilhar dados com terceiros
- Estar em conformidade com a **LGPD (Lei Geral de Proteção de Dados)**

---

## 📽️ Próximos Passos (Sprint 4)

- Enviar os dados de reconhecimento para uma API local
- Integrar ao sistema principal do **Case 1** para bloquear apostas e disparar alertas
- Criar uma interface gráfica simples e amigável
- Criar relatórios com estatísticas de uso e histórico de reconhecimento

---

## 📎 Entrega do Projeto

- **Repositório GitHub** com o código, README e instruções de execução
- **Vídeo demonstrativo (até 3 minutos)** mostrando:  
  - Contexto do problema
  - Tecnologia usada
  - Demonstração de funcionamento
  - Demonstração de parâmetros
  - Limitações e próximos passos
- O link do vídeo será disponibilizado aqui:

📹 [Clique aqui para assistir ao vídeo da apresentação](https://youtu.be/5jUpVD31BKM)

---
