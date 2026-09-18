# Modelos

`python -m safeguard download` baixa `yolo11n.pt` oficial para esta pasta.
Esse modelo COCO serve para demonstrar detecção de objetos; não reconhece EPIs.
Use `python -m safeguard train` para transfer learning com seu dataset anotado.
Copie o `best.pt` resultante para esta pasta e selecione-o na interface.

Pesos, ONNX, TensorRT e OpenVINO são ignorados pelo Git. Registre a origem,
licença, classes, hash e dataset de cada modelo entregue. Artefatos TensorRT
devem ser gerados no ambiente NVIDIA de destino.
