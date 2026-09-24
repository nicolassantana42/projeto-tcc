# Detecção, treinamento e avaliação do TCC

Execute na raiz com o ambiente ativado. Windows:
`.venv\Scripts\Activate.ps1`. Linux/macOS: `source .venv/bin/activate`.
Também é possível usar o caminho do Python da venv sem ativá-la.

## Modelo de trabalho e procedência

O fluxo principal usa **dois modelos**: YOLO11n COCO detecta pessoas e habilita
YOLO11n ajustado para EPI. O treinamento inicial local usou Construction-PPE,
10 épocas, resolução 416, batch 8 e CPU. Seus artefatos ficam em
`runs/train/ppe_tcc/`; a cópia de trabalho é `models/ppe/best.pt`, com
procedência local. Esse experimento não substitui treino com dados in loco,
análise de convergência ou validação do cenário final. Resultados efetivamente
medidos estão em [VALIDATION.md](VALIDATION.md).

Antes desse treinamento, foi avaliado um candidato externo YOLOv8n. O teste
externo encontrou desempenho insuficiente, especialmente em coletes, e ele
foi descartado como modelo principal. O candidato permanece disponível para
reproduzir a comparação em `models/ppe/baseline-public.pt`, acompanhado de
`baseline-public.provenance.json` com origem, revisão e SHA-256. O autor
publica o modelo em
[baskarmother/yolov8-ppe-construction](https://huggingface.co/baskarmother/yolov8-ppe-construction).
Sua documentação declara 17 classes, incluindo `hardhat`, `no-hardhat`,
`safety vest`, `no-safety vest` e `person`, e treinamento baseado em YOLOv8n.
O SafeGuard normaliza os nomes para o vocabulário da cascata.

O baseline não foi treinado pelos autores do TCC. Ele permite estudar erros
reais e exercitar o pipeline; não sustenta o mAP 0,841 citado no artigo nem
comprova desempenho no ambiente de instalação. Não há comparação concluída
com YOLOv5. Consulte a [matriz de aderência](TCC_ALIGNMENT.md).

Após preparar os dados e produzir os pesos conforme a seção de treinamento:

```bash
python -m safeguard detect --source data/minha-imagem.jpg --snapshot reports/imagem-anotada.jpg
python -m safeguard detect --source data/meu-video.mp4 --show --max-frames 1000 --output runs/detection/video.jsonl
```

Substitua os caminhos pelos seus arquivos. `--source 0` acessa a webcam do
computador local. `--save-events` persiste observações `unsafe` confirmadas;
`--snapshot` salva o último quadro, mesmo sem evento. Nenhum desses comandos
envia mensagens. O relatório guarda caixas, estados por pessoa, execução ou
salto do segundo estágio e tempos. Contagens são observações por quadro.

## Auditar os dados antes de medir

O preparo fornece o dataset público **Construction-PPE** em
`data/datasets/`, referenciado por `data/construction-ppe.yaml`. Segundo a
[documentação do dataset](https://docs.ultralytics.com/datasets/detect/construction-ppe/),
ele possui classes de pessoa, EPIs e algumas ausências explícitas. Não possui
classe `no_vest`; portanto, esse dataset não mede diretamente ausência de
colete. Preserve a origem, licença e atribuição dos dados na versão usada.

```bash
python -m safeguard audit-data --data data/construction-ppe.yaml --require-test --output runs/dataset-audit.json
```

A auditoria verifica decodificação das imagens, formato/IDs/limites dos
rótulos e duplicatas exatas entre splits. Corrija os problemas antes de
interpretar métricas. Imagens diferentes de uma mesma gravação também podem
vazar informação: hashes sem colisões não garantem independência. Arquivos
de rótulos ausentes exigem revisão; `--allow-background` só deve ser usado
quando a ausência foi confirmada como imagem sem objetos anotáveis.

## Avaliar a cascata completa

```bash
python -m safeguard evaluate-cascade --data data/construction-ppe.yaml --split test --confidence 0.4 --match-iou 0.5 --output runs/cascade-evaluation.json
```

Esse comando mede as **caixas finais dos dois estágios**, com pareamento
um a um por classe canônica e IoU, em um limiar de confiança fixo. O JSON
registra TP, FP, FN, precisão e recall por classe, erros individuais, cobertura,
hash do conjunto avaliado, hardware e latências por estágio/fluxo completo.
Classes fora do escopo são ignoradas e declaradas no relatório. Classes
canônicas não anotadas não são consideradas avaliadas. Precisão/recall sem
denominador são `null`, não zero nem 100%.

**Não é mAP**: não há varredura da confiança para construir curvas PR. A
latência inclui a primeira chamada e exclui leitura das imagens; não é FPS de
exibição. `--max-images` serve para um teste de execução, selecionando os
primeiros caminhos ordenados; não é uma amostra aleatória representativa.

A interseção entre dados públicos de avaliação e os dados que treinaram os
pesos externos é desconhecida. O relatório declara essa limitação e não
comprova teste independente. Caixas isoladas também não validam os estados
`ok`/`unsafe`/`uncertain`, a associação de cada EPI à pessoa ou a continuidade
temporal. Essas medidas exigem anotações específicas e revisão humana.

## Dataset e treinamento

Para reproduzir o primeiro experimento local, em um clone novo:

```bash
python scripts/prepare_ppe.py --dataset
python -m safeguard audit-data --data data/construction-ppe.yaml --require-test --output runs/dataset-audit.json
python -m safeguard train --model models/yolo11n.pt --data data/construction-ppe.yaml --epochs 10 --imgsz 416 --batch 8 --device cpu --name ppe_tcc
python -m safeguard validate --model runs/train/ppe_tcc/weights/best.pt --data data/construction-ppe.yaml --split val --imgsz 416 --name ppe_tcc_val
python -m safeguard evaluate-cascade --ppe-model runs/train/ppe_tcc/weights/best.pt --data data/construction-ppe.yaml --split test --imgsz 640 --output runs/ppe_tcc-cascade-test.json
```

O treino usa os 1.132 exemplos de treino e 143 de validação do dataset
preparado; o teste tem 141. Os conjuntos precisam ser auditados, pois a
divisão pública não certifica independência por câmera/sessão. A avaliação
individual acima é **validação**, não resultado do teste. A cascata em 640
usa outra resolução explicitamente registrada para comparação; não misture
suas métricas com mAP individual em 416.

Execuções repetidas recebem sufixos quando a pasta já existe; use o caminho
efetivamente informado pelo treino. Dez épocas constituem um experimento
inicial, não uma recomendação de convergência. O preparo apenas baixa os
artefatos públicos; ele não produz o modelo EPI treinado localmente.

Depois de revisar as métricas e evidências, a inferência aceita os pesos no
próprio diretório do treinamento:

```bash
python -m safeguard detect --source data/minha-imagem.jpg --ppe-model runs/train/ppe_tcc/weights/best.pt --show
```

Para adotar esses pesos no caminho padrão da instalação:

```bash
python -c "from pathlib import Path; import shutil; Path('models/ppe').mkdir(parents=True, exist_ok=True); shutil.copy2('runs/train/ppe_tcc/weights/best.pt', 'models/ppe/best.pt')"
```

Essa cópia substitui o modelo padrão local; preserve a procedência e o hash
dos novos pesos. `runs/`, dados baixados e pesos são ignorados pelo Git: um
clone novo precisa repetir o preparo/treino ou receber pesos compatíveis.

### Preparar dados próprios e comparar arquiteturas

Copie `data/ppe.example.yaml` para `data/ppe.yaml`, ajuste classes/paths e
prepare imagens e rótulos no formato YOLO (`class x_center y_center width height`,
coordenadas normalizadas). Os diretórios `images/train`, `images/val`,
`images/test` devem corresponder a `labels/train`, `labels/val`, `labels/test`.
Paths relativos são resolvidos a partir do YAML, não do diretório de execução.

Separe treino/validação/teste por câmera, pessoa e sessão para evitar vazamento
entre frames quase idênticos. Use validação para escolher parâmetros; reserve
teste para a avaliação final. O Git contém configurações; o preparo baixa os
dados públicos separadamente. Os dados coletados in loco ainda precisam ser
produzidos e anotados. O YAML de exemplo é apenas um esquema, não um dataset.

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

Para um experimento inicial público, substitua `data/ppe.yaml` por
`data/construction-ppe.yaml` nos comandos. Isso cria um treinamento próprio
novo, cuja qualidade depende da auditoria, duração, convergência e avaliação.
Não renomeie pesos externos para apresentá-los como resultado desse treino.

## Validação

O `validate` avalia um **detector individual**, com a mesma taxonomia e ordem
de IDs usadas no treinamento. Os 17 IDs do baseline público são diferentes
dos 11 IDs de Construction-PPE: **não valide esses pesos diretamente com
aquele YAML**. Use os pesos produzidos por treinamento no dataset correspondente.
`evaluate-cascade` faz uma avaliação separada por nomes canônicos, com as
limitações descritas acima.

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

Use `--split test` somente depois de fixar modelo e limiares com a validação.
Registre métricas do detector de pessoas: qualquer pessoa perdida pelo
primeiro estágio também fica sem análise no segundo. Para avaliar estados,
anote por pessoa os EPIs presentes/ausentes, visibilidade, oclusão e associação;
reporte falsos alertas, omissões e proporção de observações inconclusivas.

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
python -m safeguard export --model models/ppe/best.pt --format openvino --precision int8 --data data/construction-ppe-calibration.yaml --fraction 0.3 --imgsz 640 --device cpu
python -m safeguard validate --model models/ppe/best_int8_openvino_model --data data/construction-ppe.yaml --split test --name openvino-int8
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

O exportador usa a entrada `val` do YAML para calibrar. Por isso,
`construction-ppe-calibration.yaml` aponta essa entrada **somente para imagens
de treino**; nunca use esse arquivo para treinar ou avaliar métricas. A avaliação
usa o YAML original com splits separados. `--fraction 0.3` disponibilizou 340
imagens; o NNCF coletou estatísticas de 300 nesta execução. Para dados próprios,
crie uma configuração de calibração com a mesma separação. Use imagens
representativas de iluminação, distância, câmeras e classes, sem consumir o
teste para calibrar. Reavalie INT8 contra FP32 sobre o mesmo split. O CLI bloqueia
INT8 sem dataset e ONNX INT8, que não faz parte desta implementação.

## Benchmark reproduzível

```bash
python -m safeguard benchmark --model models/yolo11n.pt --source data/demo.mp4 --frames 100 --warmup 10 --output runs/benchmark-pt.json
python -m safeguard benchmark --model models/yolo11n.onnx --source data/demo.mp4 --frames 100 --warmup 10 --output runs/benchmark-onnx.json
python -m safeguard benchmark --model models/yolo11n.pt --ppe-model models/ppe/best.pt --source data/demo.mp4 --frames 100 --warmup 10 --output runs/benchmark-cascade-pt.json
python -m safeguard benchmark --model models/yolo11n.pt --ppe-model models/ppe/best_int8_openvino_model --source data/demo.mp4 --frames 100 --warmup 10 --output runs/benchmark-cascade-int8.json
```

O primeiro frame é usado para warmup; os próximos são medidos. Meça o mesmo
vídeo fornecido por você (`data/demo.mp4` é apenas um caminho de exemplo).
O modo PyTorch usa FP16 em CUDA e FP32 em CPU/MPS; registre essa diferença
na comparação. Validação padrão usa FP32 para `.pt`; quantização dos artefatos
é definida na exportação.

Use a mesma resolução e número de frames, com energia/temperatura controladas. JSON
inclui média, p50/p95 e FPS de: chamada completa do detector; pipeline; e
captura + pipeline + renderização. A leitura do vídeo está incluída; UI e
gravação de relatórios/snapshots ficam fora desse benchmark.

Tabela sugerida para o TCC (preencha com resultados, não valores publicitários):

| Modelo/backend | Hardware | mAP50–95 | Recall capacete/colete | p95 ponta a ponta | FPS |
| --- | --- | --- | --- | --- | --- |
| YOLOv5 de referência, ainda sem experimento reproduzido | medir | medir | medir | medir | medir |
| YOLOv8n PyTorch (FP32/FP16 conforme dispositivo) | medir | medir | medir | medir | medir |
| YOLO11n PyTorch (FP32/FP16 conforme dispositivo) | medir | medir | medir | medir | medir |
| YOLO11n ONNX | medir | medir | medir | medir | medir |
| YOLO11n INT8 | medir | medir | medir | medir | medir |

Sem `--ppe-model`, o benchmark mede um detector. Com esse argumento, mede os
dois estágios, seus tempos e quantos quadros executaram/pularam o segundo.
O primeiro frame deve conter uma pessoa para aquecer também o modelo de EPI.
Números de um estágio isolado não são o FPS total.
O comparativo com YOLOv5 ainda exige um ambiente/modelo de referência e o
mesmo protocolo de dados; não há evidência para afirmar que a migração aumenta
mAP ou reduz latência neste cenário.

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
