# Modelo

## Arquitetura do pipeline

1. **Captura** (`capture.py`): imagem, vídeo, webcam ou RTSP.
2. **Pessoas** (`yolo11n.pt`, COCO): detecta pessoas. Sem pessoas, o 2º estágio não executa.
3. **EPIs** (`ppe/epi.pt`): YOLO11n treinado para capacete, colete e bota (e `no_helmet`, `no_boots`).
4. **Associação** (`detection.py`): cada EPI vai para a pessoa cuja região (cabeça, tronco ou pés) o contém; em disputa, fica com a mais próxima.
5. **Decisão por pessoa**: `ok` (todos os EPIs vistos), `unsafe` (classe negativa explícita) ou `uncertain` (EPI não visto, pessoa cortada ou oclusão).
6. **Evidência** (`events.py`, `notifications.py`): foto + JSON, Telegram opcional, revisão do analista.

## Treino

- Dataset: **Construction-PPE** (Ultralytics, público): 1.132 treino · 143 validação · 141 teste.
- YOLO11n, 10 épocas (a partir do COCO) + 40 épocas de ajuste, 480 px, batch 16, CPU Intel i7-1355U.
- Artefatos do treino: `runs/train/ppe_epi3/` (curvas, matriz de confusão, `training.json`).

Reproduzir (precisa de internet para baixar o dataset):

```bash
python scripts/prepare_ppe.py --dataset
python -m epi_monitor train --model models/yolo11n.pt --data data/construction-ppe.yaml --epochs 50 --imgsz 480 --batch 16 --workers 4 --device cpu --name ppe_epi
```

## Resultados (conjunto de teste, 141 imagens, 480 px)

| Classe | Precisão | Recall | mAP50 | mAP50-95 | Instâncias |
|---|---|---|---|---|---|
| Capacete | 0,89 | 0,90 | **0,94** | 0,52 | 192 |
| Colete | 0,79 | 0,87 | **0,89** | 0,58 | 178 |
| Bota | 0,73 | 0,72 | **0,76** | 0,41 | 211 |

Modelo anterior (só 10 épocas): 0,91 / 0,89 / 0,72. Valores em `models/ppe/epi.metrics.json`.

```bash
python -m epi_monitor validate --model models/ppe/epi.pt --data data/construction-ppe.yaml --split test --imgsz 480 --batch 1
```

## Desempenho (CPU i7-1355U)

| Configuração | FPS da cascata |
|---|---|
| PyTorch, 640 px | ≈ 4 |
| OpenVINO, 640 px | ≈ 7 |
| **OpenVINO, 480 px (padrão)** | **≈ 13** |
| OpenVINO, 480 px, 1 modelo (modo rápido) | ≈ 18 |

Exportar de novo para OpenVINO (necessário só se `epi.pt` for retreinado):

```bash
copy models\ppe\epi.pt models\ppe\epi_480.pt
python -c "from ultralytics import YOLO; YOLO('models/ppe/epi_480.pt').export(format='openvino', imgsz=480)"
del models\ppe\epi_480.pt
```

## Limitações

- Dataset público, sem imagens do local de uso.
- Sem classe "sem colete": colete não visto fica inconclusivo.
- "Sem capacete" e "sem bota" têm poucos exemplos (recall baixo).
- O modo rápido pode perder pessoas fora de cenas de obra.
