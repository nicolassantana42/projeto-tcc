# Modelos

O padrão da cascata é `yolo11n.pt` para pessoas e **`ppe/absence.pt`** para EPI.
O primeiro é COCO oficial: `python -m safeguard download` prepara esse arquivo,
que não reconhece EPIs. O segundo é o YOLO11n ajustado localmente no RF100 com
capacete, colete e suas ausências explícitas. O download de COCO não o produz.

Prepare dados/pesos e execute o treinamento conforme
[docs/ABSENCE_DATA.md](../docs/ABSENCE_DATA.md). Após revisar a validação:

```bash
python scripts/promote_absence_model.py --run-dir runs/train/ppe_absence
```

Execute na raiz do repositório. O script confere SHA-256 e classes do checkpoint
com `training.json`, exige classes únicas de capacete/colete e suas ausências
e publica `ppe/absence.pt` com `ppe/absence.provenance.json`. Ele reutiliza
conteúdo idêntico e recusa sobrescrever arquivos divergentes. A promoção
verifica integridade, não comprova precisão no local de instalação.

| Arquivo | Papel |
|---|---|
| `yolo11n.pt` | Detector de pessoas COCO, primeiro estágio |
| `ppe/absence.pt` | Segundo estágio padrão; ajuste local com ausência de colete |
| `ppe/absence-base.pt` | Ponto de partida externo YOLO11n; origem e hash preservados |
| `ppe/best.pt` | Experimento histórico Construction-PPE de 11 classes, sem `no_vest` |
| `ppe/best_int8_openvino_model/` | INT8 do modelo histórico; não é export do `absence.pt` |
| `ppe/baseline-public.pt` | Antigo candidato externo YOLOv8n, mantido para comparação |

O `absence.pt` tem dez saídas, porém o experimento anotou e avaliou somente
cinco. Não há evidência de qualidade para as outras cinco classes. O recall
de sem capacete permanece baixo e foram observadas regressões no dataset
anterior; veja [resultados e limites](../docs/VALIDATION.md).

Novos exports de `absence.pt` devem usar seus próprios nomes e ser avaliados
contra o mesmo checkpoint PyTorch. Os exports históricos não foram atualizados
pela promoção. Instruções de calibração e avaliação em
[docs/ML.md](../docs/ML.md).

Pesos, ONNX, TensorRT e OpenVINO são ignorados pelo Git. Registre a origem,
licença, classes, hash e dataset de cada modelo entregue. Artefatos TensorRT
devem ser gerados no ambiente NVIDIA de destino.
