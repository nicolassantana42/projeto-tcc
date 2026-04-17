# 🦺 Sistema Inteligente de Monitoramento de EPI
**TCC — Visão Computacional e Inteligência Artificial em Tempo Real**

---

## 📋 Visão Geral

Sistema automatizado de monitoramento de Equipamentos de Proteção Individual (EPI) utilizando **YOLOv8** e **OpenCV**, capaz de detectar em tempo real trabalhadores sem capacete e/ou colete de segurança, gerando alertas automáticos via **Telegram** e um **Dashboard** interativo.

### Contribuição Científica

O diferencial técnico deste projeto é o **motor de regras espaciais** (`src/rules/ppe_rules.py`) que, ao contrário de uma simples detecção de objetos, **associa cada EPI à pessoa correta** utilizando análise de bounding boxes e IoU (Intersection over Union). Isso permite monitorar múltiplas pessoas no mesmo frame de forma independente e precisa.

---

## 🏗️ Estrutura do Projeto

```
ppe_monitor/
├── main.py                          ← Ponto de entrada principal
├── config.py                        ← Configurações centrais
├── setup_and_test.py                ← Instalação e validação
├── requirements.txt                 ← Dependências Python
├── .env                             ← Suas configurações (não commitar!)
├── .env.example                     ← Modelo de configuração
│
├── src/
│   ├── camera/
│   │   └── capture.py               ← Captura de vídeo + FPS counter
│   ├── ai/
│   │   └── detector.py              ← Inferência YOLOv8
│   ├── rules/
│   │   └── ppe_rules.py             ← Motor de regras EPI (coração do TCC)
│   ├── alerts/
│   │   ├── logger.py                ← Log em arquivo + JSON + imagens
│   │   └── telegram.py              ← Alertas Telegram Bot
│   └── dashboard/
│       └── streamlit_app.py         ← Dashboard de monitoramento
│
├── violations/                      ← Imagens das infrações (gerado automaticamente)
└── logs/                            ← Logs e JSON (gerado automaticamente)
```

---

## 🚀 Instalação e Execução

### Pré-requisitos
- Python 3.9 ou superior
- **Câmera:** Webcam, USB, ou IP (Intelbras, Hikvision, etc)
- Conexão com internet (para baixar o modelo na 1ª vez)

### Passo 1 — Setup inicial

```bash
# Clone ou extraia o projeto
cd ppe_monitor

# Execute o setup (instala tudo e valida)
python setup_and_test.py
```

### Passo 2 — Configure a câmera

#### 🎥 Para Webcam/USB:
```bash
# Detecte câmeras automaticamente:
python test_camera.py --detect

# Teste uma câmera específica:
python test_camera.py --index 0

# Configure no .env o índice detectado:
CAMERA_INDEX=0  # 0=primeira câmera, 1=segunda, etc
```

#### 📡 Para Câmeras IP (Intelbras, Hikvision, etc):
```bash
# 1. Encontre a URL RTSP da sua câmera
# 2. Teste no VLC primeiro: Mídia > Abrir Fluxo de Rede
# 3. Configure no .env:

# Exemplo Intelbras:
CAMERA_INDEX=rtsp://admin:senha@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0

# Exemplo Hikvision:
CAMERA_INDEX=rtsp://admin:senha@192.168.1.64:554/Streaming/Channels/101

# Exemplo HTTP/MJPEG:
CAMERA_INDEX=http://192.168.1.108/video.mjpg
```

**💡 Dicas importantes:**
- Use IP fixo na câmera para evitar mudanças
- Verifique firewall e rede
- Para troubleshooting: veja [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

### Passo 3 — Configure alertas (opcional)

```env
TELEGRAM_TOKEN=seu_token_aqui      # Do @BotFather
TELEGRAM_CHAT_ID=seu_chat_id_aqui  # Do /getUpdates
```

### Passo 4 — Execute

```bash
# Terminal 1: Sistema principal (câmera + detecção)
python main.py

# Terminal 2: Dashboard (opcional, mas recomendado para a banca)
streamlit run src/dashboard/streamlit_app.py
```

### Teclas durante execução

| Tecla | Ação |
|-------|------|
| `Q` / `ESC` | Encerrar o sistema |
| `B` | Executar benchmark de performance |
| `S` | Salvar frame atual manualmente |

---

## 🤖 Modos de Operação

### Modo Demo (padrão — `DEMO_MODE=true`)
Usa o modelo `yolov8n.pt` pré-treinado no dataset COCO. Detecta apenas a classe `person`. Como o modelo COCO não possui classes de capacete/colete, **todas as pessoas aparecerão como "SEM CAPACETE"** — ideal para demonstrar a arquitetura e o pipeline completo.

### Modo Real (`DEMO_MODE=false`)
Requer um modelo treinado em dataset de EPI. Baixe um modelo PPE no [Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) e configure `MODEL_PATH` no `.env`.

---

## 📊 Métricas para o TCC

### Benchmark de Modelos
Durante a execução, pressione **`B`** para medir o desempenho. Repita com cada modelo (`yolov8n`, `yolov8s`, `yolov8m`) e preencha a tabela:

| Modelo | Parâmetros | Tempo médio | FPS equiv. | mAP@0.5 |
|--------|-----------|-------------|-----------|---------|
| yolov8n | 3.2M | — ms | — | — |
| yolov8s | 11.2M | — ms | — | — |
| yolov8m | 25.9M | — ms | — | — |

### Cenários de Teste (Capítulo de Resultados)

| Cenário | Descrição | Resultado Esperado |
|---------|-----------|-------------------|
| 1 | 1 pessoa COM capacete | Sem alerta |
| 2 | 1 pessoa SEM capacete | Alerta gerado |
| 3 | 2 pessoas (1 com, 1 sem) | Alerta seletivo |
| 4 | Iluminação baixa | — |
| 5 | Oclusão parcial | — |
| 6 | Distância longa | — |

---

## 🔬 Metodologia — Motor de Regras Espaciais

```
Para cada pessoa P detectada no frame:
│
├─ 1. REGIÃO DA CABEÇA
│     Extrai os 30% superiores da bounding box de P
│     head_box = [x1, y1, x2, y1 + (height × 0.30)]
│
├─ 2. ASSOCIAÇÃO DE CAPACETE (IoU-based)
│     Para cada capacete H detectado:
│       iou = intersection(head_box, H.bbox) / union(head_box, H.bbox)
│       Se iou ≥ 0.15 → capacete associado a P ✅
│
├─ 3. ASSOCIAÇÃO DE COLETE (centro-point)
│     Para cada colete V detectado:
│       Se centro(V) está dentro de bbox(P) → colete OK ✅
│
└─ 4. VIOLAÇÃO
      Se capacete obrigatório E não encontrado → "SEM CAPACETE" ⛔
      Se colete obrigatório E não encontrado   → "SEM COLETE"   ⛔
```

**Vantagem vs. detecção simples:**
- Detecção simples: "há um capacete no frame?" → muitos falsos negativos
- Nossa abordagem: "**esta pessoa específica** tem capacete?" → preciso para N pessoas

---

## 📱 Configuração do Telegram

1. Abra o **@BotFather** no Telegram
2. Digite `/newbot` e siga as instruções
3. Copie o **TOKEN** gerado
4. Inicie uma conversa com o seu bot
5. Acesse `https://api.telegram.org/bot<TOKEN>/getUpdates`
6. Copie o valor `"id"` do campo `"chat"` → esse é o **CHAT_ID**

```env
TELEGRAM_TOKEN=1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ
TELEGRAM_CHAT_ID=987654321
```

---

## 🛠️ Solução de Problemas

| Problema | Solução |
|----------|---------|
| `ModuleNotFoundError: config` | Execute sempre da raiz: `python main.py` |
| Câmera não encontrada | Mude `CAMERA_INDEX=1` (ou 2) no `.env` |
| Janela não abre | Verifique `SHOW_VIDEO=true` no `.env` |
| Detecção muito lenta | Use `yolov8n.pt` e reduza `FRAME_WIDTH=320` |
| Muitos falsos positivos | Aumente `CONFIDENCE=0.60` no `.env` |
| Telegram não envia | Verifique TOKEN e CHAT_ID; teste com `test_connection()` |

---

## 📚 Referências

- Ultralytics YOLOv8: https://docs.ultralytics.com
- OpenCV: https://docs.opencv.org
- Streamlit: https://docs.streamlit.io
- Dataset PPE: https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety
- COCO Dataset: https://cocodataset.org

---

*TCC — Sistema Inteligente de Monitoramento de EPI*
*Utilizando Visão Computacional e IA em Tempo Real*
