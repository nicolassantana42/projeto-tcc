# Treinamento, avaliação e otimização

Execute na raiz com o ambiente ativado. Windows:
`.venv\Scripts\Activate.ps1`. Linux/macOS: `source .venv/bin/activate`.
Também é possível usar o caminho do Python da venv sem ativá-la.

## Dataset e treinamento

Copie `data/ppe.example.yaml` para `data/ppe.yaml`, ajuste classes/paths e
prepare imagens e rótulos no formato YOLO (`class x_center y_center width height`,
coordenadas normalizadas). Os diretórios `images/train`, `images/val`,
`images/test` devem corresponder a `labels/train`, `labels/val`, `labels/test`.
Paths relativos são resolvidos a partir do YAML, não do diretório de execução.

Separe treino/validação/teste por câmera, pessoa e sessão para evitar vazamento
entre frames quase idênticos. Use validação para escolher parâmetros; reserve
teste para a avaliação final. Não existe dataset EPI embutido.

```bash
python -m safeguard download --model yolo11n.pt
python -m safeguard train --model models/yolo11n.pt --data data/ppe.yaml --epochs 100 --batch 8 --name yolo11n
python -m safeguard download --model yolov8n.pt
python -m safeguard train --model models/yolov8n.pt --data data/ppe.yaml --epochs 100 --batch 8 --name yolov8n
```

Saídas: `runs/train/<nome>/weights/best.pt`, `last.pt`, `training.json`,
gráficos e `dataset.resolved.yaml`. Runs existentes recebem novo nome. O seed
padrão é 42; determinismo solicitado não garante resultados bit a bit em todo
hardware. YOLO11n é o baseline configurável, não uma promessa de superioridade.

## Validação

```bash
python -m safeguard validate --model runs/train/yolo11n/weights/best.pt --data data/ppe.yaml --split val --name yolo11n
python -m safeguard validate --model runs/train/yolov8n/weights/best.pt --data data/ppe.yaml --split val --name yolov8n
```

`plots=True` gera matriz de confusão, PR/F1/precision/recall e exemplos
anotados. `validation.json` registra mAP50, mAP50–95, precisão, recall, dados das
curvas/matriz quando fornecidos pelo backend, versões, dispositivo e paths dos
artefatos. `save_json=True` mantém predições da validação. Confiança padrão
0,001 preserva a faixa da curva PR; não confunda com o limiar da demonstração.

Se o conjunto não tiver anotações válidas de uma classe, suas métricas não
servem para concluir desempenho nessa classe. Inspecione erros, não apenas mAP.

## ONNX e quantização

ONNX FP32 é serialização para outro runtime; não é quantização INT8.

```bash
python -m pip install -e ".[onnx]"
python -m safeguard export --model models/yolo11n.pt --format onnx --precision fp32 --imgsz 640
python -m safeguard infer --model models/yolo11n.onnx --source data/demo.mp4 --max-frames 100
```

OpenVINO INT8 em CPU, com dados representativos explícitos:

```bash
python -m pip install -e ".[openvino]"
python -m safeguard export --model models/ppe/best.pt --format openvino --precision int8 --data data/ppe.yaml --device cpu
python -m safeguard validate --model models/ppe/best_int8_openvino_model --data data/ppe.yaml --name openvino-int8
```

TensorRT deve ser instalado no ambiente NVIDIA com CUDA/driver compatíveis.
O extra `.[tensorrt]` declara dependências Python, mas não instala driver/CUDA.
Após preparar esse ambiente:

```bash
python -m safeguard export --model models/ppe/best.pt --format engine --precision fp16 --device cuda:0
python -m safeguard export --model models/ppe/best.pt --format engine --precision int8 --data data/ppe.yaml --device cuda:0
```

Cada exportação grava `<artefato>.export.json` com precisão/configuração. O
nome do artefato é controlado pela Ultralytics; exportar outra precisão do mesmo
modelo/formato pode sobrescrever o anterior. Copie cada resultado para uma
pasta identificada antes de gerar o seguinte. Use `batch=1` e o mesmo `imgsz`
na exportação e na inferência dos artefatos estáticos. O dashboard solicita 640;
o adaptador adota a resolução fixa gravada no artefato quando diferente.
Mantenha a pasta OpenVINO com o sufixo `_openvino_model` e os arquivos
XML/BIN/metadata juntos. TensorRT INT8 recusa um cache `.cache` já existente;
mova esse cache antes de reexportar para evitar calibração com dados antigos.

`--fraction` seleciona a fração de validação usada na calibração. Use imagens
representativas de iluminação, distância, câmeras e classes, nunca apenas uma
imagem de teste. Reavalie INT8 contra FP32 sobre o mesmo split. O CLI bloqueia
INT8 sem dataset e ONNX INT8, que não faz parte desta implementação.

## Benchmark reproduzível

```bash
python -m safeguard benchmark --model models/yolo11n.pt --source data/demo.mp4 --frames 100 --warmup 10 --output runs/benchmark-pt.json
python -m safeguard benchmark --model models/yolo11n.onnx --source data/demo.mp4 --frames 100 --warmup 10 --output runs/benchmark-onnx.json
```

O primeiro frame é usado para warmup; os próximos são medidos. Meça o mesmo
vídeo fornecido por você (`data/demo.mp4` é apenas um caminho de exemplo).
O modo PyTorch usa FP16 em CUDA e FP32 em CPU/MPS; registre essa diferença
na comparação. Validação padrão usa FP32 para `.pt`; quantização dos artefatos
é definida na exportação.

Use a mesma resolução e número de frames, com energia/temperatura controladas. JSON
inclui média, p50/p95 e FPS de: chamada completa do detector; pipeline; e
captura + pipeline + renderização. UI e disco ficam fora desse benchmark.

Tabela sugerida para o TCC (preencha com resultados, não valores publicitários):

| Modelo/backend | Hardware | mAP50–95 | Recall capacete/colete | p95 ponta a ponta | FPS |
| --- | --- | --- | --- | --- | --- |
| YOLOv8n PyTorch (FP32/FP16 conforme dispositivo) | medir | medir | medir | medir | medir |
| YOLO11n PyTorch (FP32/FP16 conforme dispositivo) | medir | medir | medir | medir | medir |
| YOLO11n ONNX | medir | medir | medir | medir | medir |
| YOLO11n INT8 | medir | medir | medir | medir | medir |

## Referências

- [YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- [Exportador da versão fixada 8.3.203](https://github.com/ultralytics/ultralytics/blob/v8.3.203/ultralytics/engine/exporter.py)
- [Validação Ultralytics](https://docs.ultralytics.com/modes/val/)
- [OpenVINO](https://docs.ultralytics.com/integrations/openvino/)
- [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/)

A documentação online evolui; os comandos deste projeto usam a API da versão
fixada, com `half`/`int8`. Antes de atualizar dependências, repita os testes de
integração, exportação e validação dos pesos reais.
