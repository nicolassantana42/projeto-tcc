# Monitor de EPIs — capacete, colete e bota

Sistema de visão computacional para o TCC: **câmera/imagem → YOLO de pessoas → YOLO de EPIs → decisão por pessoa → evidência e alerta**.

## ▶ Como rodar

1. Instale o **Python 3.12** (uma única vez): `winget install -e --id Python.Python.3.12`
2. Dê **dois cliques em `iniciar.bat`** (ou `python run.py`). Na 1ª vez ele cria a `.venv` e instala tudo (precisa de internet); depois abre direto.
3. No navegador (**http://localhost:8501**) escolha a fonte e clique **▶ Iniciar**.

**Para demonstrar:**

| Demonstração | Como |
|---|---|
| Vídeo (mostra o gráfico de conformidade) | Fonte **Arquivo de vídeo** → *Caminho no computador* → `data/demo/demo_obra.mp4` |
| Imagem | Fonte **Imagem** → *Caminho no computador* → `data/datasets/images/test/image611.jpg` |
| Webcam | Fonte **Webcam local** → índice `0` |

Roteiro completo da apresentação e respostas para a banca: [docs/APRESENTACAO.md](docs/APRESENTACAO.md).

> Para rodar em outro computador, copie **a pasta inteira**, incluindo `models/` (≈100 MB). Os pesos não estão no Git.

## O que o sistema faz

- Detecta pessoas (YOLO11n COCO) e, só quando há pessoas, roda o detector de EPIs (YOLO11n treinado).
- Associa cada EPI à pessoa pela região do corpo (cabeça, tronco, pés) e mostra para cada uma: ✅ Detectado · ❌ Ausente · ⚠️ Não detectado.
- Salva ocorrências (foto + JSON com câmera, local e horário) e envia alerta pelo **Telegram** (opcional).
- O analista **confirma ou descarta** cada ocorrência na aba *Ocorrências*; o painel calcula a precisão revisada.
- Aba **Modelo** mostra as métricas medidas no conjunto de teste.

Decisão conservadora: não ver um EPI **não prova** que ele está ausente. "Ausente" exige uma classe negativa explícita do modelo (ex.: `no_helmet`); o resto fica como "Não detectado".

## Modelo

`models/ppe/epi.pt` — YOLO11n treinado no dataset público **Construction-PPE** (Ultralytics): 1.132 imagens de treino, 143 de validação, 141 de teste; 10 + 40 épocas em CPU, 480 px (`runs/train/ppe_epi3`).

Conjunto de **teste** (141 imagens nunca usadas no treino), 480 px:

| Classe | Precisão | Recall | mAP50 | Modelo anterior (10 épocas) |
|---|---|---|---|---|
| Capacete | 0,89 | 0,90 | **0,94** | 0,91 |
| Colete | 0,79 | 0,87 | **0,89** | 0,89 |
| Bota | 0,73 | 0,72 | **0,76** | 0,72 |

**Limites conhecidos:** o dataset não tem a classe "sem colete" (colete não visto = ⚠️); "sem capacete" e "sem bota" têm poucos exemplos e baixo recall; não houve coleta de imagens no local de uso.

## Desempenho (Intel i7-1355U, só CPU)

| Configuração | FPS da cascata |
|---|---|
| Antes: 2 modelos, 640 px | 7 |
| **Padrão: OpenVINO, 2 modelos, 480 px** | **≈ 13** |
| *Avançado → Velocidade → Rápida · 1 modelo* | ≈ 18 |

Na interface o FPS observado é menor (≈ 7–10), porque o navegador e o redesenho da tela também usam a CPU; se precisar de mais fluidez, use o modo rápido. O modo rápido usa a classe *Person* do próprio modelo de EPI: ótimo em cenas de obra, pode perder pessoas em outros ambientes. A interface limita a 10 análises/s por padrão (ajustável em *Avançado*). Com GPU NVIDIA, o perfil PyTorch usa CUDA automaticamente.

## Linha de comando

```bash
.venv\Scripts\activate
python -m epi_monitor detect --source data/demo/demo_obra.mp4 --show --imgsz 480
python -m epi_monitor detect --source 0 --show                       # webcam
python -m epi_monitor validate --model models/ppe/epi.pt --data data/construction-ppe.yaml --split test --imgsz 480
python -m epi_monitor train --model models/ppe/best.pt --data data/construction-ppe.yaml --epochs 40 --imgsz 480 --batch 16 --workers 4 --device cpu --name ppe_epi3
```

Em um clone novo (sem pesos/dataset): `python -m epi_monitor download` baixa o YOLO11n COCO e `python scripts/prepare_ppe.py --dataset` baixa o Construction-PPE. Protocolo de ML: [docs/ML.md](docs/ML.md).

## Telegram

Aba **Alertas e integrações**: cadastre câmera e local, informe o token do bot (@BotFather) e o Chat ID, ative e salve. **Enviar teste aos canais ativos** faz um envio real. Tokens ficam só na sessão. Guia: [docs/ALERTS.md](docs/ALERTS.md).

## Docker (opcional)

```bash
docker compose up --build
```

Serve em http://localhost:8501 com `models/` e `reports/` montados como volumes. No container a webcam USB só funciona em Linux; prefira vídeo ou RTSP.

## Estrutura e testes

```text
src/epi_monitor/
  detection.py   cascata, associação EPI → pessoa e decisão
  inference.py   adaptador YOLO (PyTorch / OpenVINO / ONNX)
  events.py      ocorrências (foto + JSON), revisão do analista
  notifications.py  Telegram / e-mail
  ml.py, cli.py  treino, validação, exportação e comandos
  ui/            interface Streamlit (app.py, theme.py, alert_panels.py)
models/  data/  reports/  docs/  tests/
iniciar.bat, run.py   inicialização com um comando
```

```bash
python -m pytest -q
```

[Arquitetura](docs/ARCHITECTURE.md) · [Aderência ao roteiro do TCC](docs/TCC_ALIGNMENT.md) · [Validação](docs/VALIDATION.md) · [Design da interface](docs/DESIGN.md) · [Apresentação](docs/APRESENTACAO.md)
