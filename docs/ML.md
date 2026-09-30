# Detecção, treinamento e avaliação do TCC

Execute na raiz com o ambiente ativado. Windows:
`.venv\Scripts\Activate.ps1`. Linux/macOS: `source .venv/bin/activate`.
Também é possível usar o caminho do Python da venv sem ativá-la.

## Modelo de trabalho e procedência

O fluxo principal usa **dois modelos**: `models/yolo11n.pt`, YOLO11n COCO,
detecta pessoas e habilita o YOLO11n de EPI em **`models/ppe/absence.pt`**.
O segundo foi ajustado localmente no RF100 construction-safety-gsnvb, a partir
de pesos externos com classes explícitas de ausência de capacete e colete:
10 épocas, resolução 416, batch 8, CPU e primeiras dez camadas congeladas.
Os artefatos ficam em `runs/train/ppe_absence/`. Origem, licenças, hashes,
mapeamento de classes e limitações estão em [ABSENCE_DATA.md](ABSENCE_DATA.md).

O modelo anterior de 11 classes, treinado em Construction-PPE, permanece em
`models/ppe/best.pt`; seus artefatos estão em `runs/train/ppe_tcc/`.
`best_int8_openvino_model` continua sendo a quantização **desse modelo anterior**,
que não possui `no_vest`. Esses arquivos históricos não foram substituídos
pelo novo modelo. Para uma comparação histórica, informe os pesos explicitamente.

A adoção de `absence.pt` habilita ausência de colete; não significa melhora
universal. No teste RF100, o recall de sem capacete foi apenas 25%; no teste
Construction-PPE houve perda de recall de colete e aumento de falsos positivos
de sem capacete. Resultados e protocolos completos constam em
[VALIDATION.md](VALIDATION.md). Nenhum dos experimentos substitui validação
com imagens e vídeos do ambiente de instalação.

Antes desse treinamento, foi avaliado um candidato externo YOLOv8n. O teste
externo encontrou desempenho insuficiente, especialmente em coletes, e ele
foi descartado como modelo principal. O candidato permanece disponível para
reproduzir a comparação em `models/ppe/baseline-public.pt`, acompanhado de
`baseline-public.provenance.json` com origem, revisão e SHA-256. O autor
publica o modelo em
[baskarmother/yolov8-ppe-construction](https://huggingface.co/baskarmother/yolov8-ppe-construction).
Sua documentação declara 17 classes, incluindo `hardhat`, `no-hardhat`,
`safety vest`, `no-safety vest` e `person`, e treinamento baseado em YOLOv8n.
A aplicação normaliza os nomes para o vocabulário da cascata.

O baseline não foi treinado pelos autores do TCC. Ele permite estudar erros
reais e exercitar o pipeline; não sustenta o mAP 0,841 citado no artigo nem
comprova desempenho no ambiente de instalação. Não há comparação concluída
com YOLOv5. Consulte a [matriz de aderência](TCC_ALIGNMENT.md).

Após preparar os dados e produzir os pesos conforme a seção de treinamento:

```bash
python -m epi_monitor detect --source data/minha-imagem.jpg --snapshot reports/imagem-anotada.jpg
python -m epi_monitor detect --source data/meu-video.mp4 --show --max-frames 1000 --output runs/detection/video.jsonl
```

Substitua os caminhos pelos seus arquivos. `--source 0` acessa a webcam do
computador local. `--save-events` persiste observações `unsafe` confirmadas;
`--snapshot` salva o último quadro, mesmo sem evento. Nenhum desses comandos
envia mensagens. O relatório guarda caixas, estados por pessoa, execução ou
salto do segundo estágio e tempos. Contagens são observações por quadro.

## Auditar os dados antes de medir

O experimento atual usa `data/ppe-absence.yaml` para avaliar a cascata e
`data/ppe-absence-transfer.yaml` para treino/validação individual do segundo
modelo. São as mesmas 997/119/90 imagens, com IDs remapeados para preservar
as dez saídas dos pesos de partida. Apenas cinco classes estão anotadas;
as demais aparecem como não avaliadas. Consulte [ABSENCE_DATA.md](ABSENCE_DATA.md).

O preparo histórico fornece o dataset público **Construction-PPE** em
`data/datasets/`, referenciado por `data/construction-ppe.yaml`. Segundo a
[documentação do dataset](https://docs.ultralytics.com/datasets/detect/construction-ppe/),
ele possui classes de pessoa, EPIs e algumas ausências explícitas. Não possui
classe `no_vest`; portanto, esse dataset não mede diretamente ausência de
colete. Preserve a origem, licença e atribuição dos dados na versão usada.

```bash
python -m epi_monitor audit-data --data data/ppe-absence.yaml --require-test --output runs/ppe-absence/audit-original-final.json
```

A auditoria verifica decodificação das imagens, formato/IDs/limites dos
rótulos e duplicatas exatas entre splits. Corrija os problemas antes de
interpretar métricas. Imagens diferentes de uma mesma gravação também podem
vazar informação: hashes sem colisões não garantem independência. Arquivos
de rótulos ausentes exigem revisão; `--allow-background` só deve ser usado
quando a ausência foi confirmada como imagem sem objetos anotáveis.

## Avaliar a cascata completa

```bash
python -m epi_monitor evaluate-cascade --ppe-model models/ppe/absence.pt --data data/ppe-absence.yaml --split val --imgsz 640 --confidence 0.4 --iou 0.45 --match-iou 0.5 --output runs/ppe-absence/cascade-val.json
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

### Experimento atual: capacete, colete e ausências explícitas

```bash
python scripts/prepare_absence_data.py --workers 4
python scripts/prepare_absence_model.py
python scripts/prepare_absence_transfer.py
python -m epi_monitor train --model models/ppe/absence-base.pt --data data/ppe-absence-transfer.yaml --epochs 10 --imgsz 416 --batch 8 --workers 0 --seed 42 --device cpu --freeze 10 --project runs/train --name ppe_absence
python -m epi_monitor validate --model runs/train/ppe_absence/weights/best.pt --data data/ppe-absence-transfer.yaml --split val --imgsz 640 --device cpu --name ppe_absence_val
python -m epi_monitor evaluate-cascade --ppe-model runs/train/ppe_absence/weights/best.pt --data data/ppe-absence.yaml --split val --imgsz 640 --confidence 0.4 --iou 0.45 --match-iou 0.5 --device cpu --output runs/ppe-absence/trained-val.json
```

Escolha os pesos e limiares na validação; depois avalie o teste reservado
com `--split test` e outro arquivo de saída. O `train` registra hashes,
configuração e métricas em `training.json`. O conjunto tem apenas 94 caixas
de sem capacete no treino, contra 2.116 de capacete. A sobreposição com o
pré-treinamento externo é desconhecida, e foram encontrados erros semânticos
em rótulos públicos, mantidos na avaliação original.

Depois de revisar a validação, publique os pesos no caminho padrão:

```bash
python scripts/promote_absence_model.py --run-dir runs/train/ppe_absence
python -m epi_monitor detect --source data/minha-imagem.jpg --ppe-model models/ppe/absence.pt --show
```

A promoção verifica SHA-256, nomes/IDs do checkpoint, classes de presença e
ausência e correspondência com `training.json`. Grava `absence.pt` e
`absence.provenance.json`; arquivos já existentes com conteúdo divergente
são preservados e a operação falha. Repetir a mesma promoção é permitido.
O script não atesta precisão nem altera `best.pt` ou seus exports anteriores.

### Experimento histórico: Construction-PPE sem classe de ausência de colete

Para reproduzir o primeiro experimento local, em um clone novo:

```bash
python scripts/prepare_ppe.py --dataset
python -m epi_monitor audit-data --data data/construction-ppe.yaml --require-test --output runs/dataset-audit.json
python -m epi_monitor train --model models/yolo11n.pt --data data/construction-ppe.yaml --epochs 10 --imgsz 416 --batch 8 --device cpu --name ppe_tcc
python -m epi_monitor validate --model runs/train/ppe_tcc/weights/best.pt --data data/construction-ppe.yaml --split val --imgsz 416 --name ppe_tcc_val
python -m epi_monitor evaluate-cascade --ppe-model runs/train/ppe_tcc/weights/best.pt --data data/construction-ppe.yaml --split test --imgsz 640 --output runs/ppe_tcc-cascade-test.json
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
python -m epi_monitor detect --source data/minha-imagem.jpg --ppe-model runs/train/ppe_tcc/weights/best.pt --show
```

Esses pesos históricos não devem substituir `absence.pt`: sua taxonomia não
contém ausência de colete. `runs/`, dados baixados e pesos são ignorados pelo
Git: um clone novo precisa repetir o preparo/treino e a promoção do experimento
atual, ou receber os artefatos locais com sua procedência.

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
python -m epi_monitor download --model yolo11n.pt
python -m epi_monitor train --model models/yolo11n.pt --data data/ppe.yaml --epochs 100 --batch 8 --name yolo11n
python -m epi_monitor download --model yolov8n.pt
python -m epi_monitor train --model models/yolov8n.pt --data data/ppe.yaml --epochs 100 --batch 8 --name yolov8n
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
de IDs usadas no treinamento. `absence.pt` tem dez IDs e exige
`data/ppe-absence-transfer.yaml` para validação individual; o YAML RF100
original de cinco IDs não é intercambiável. Os 17 IDs do baseline público
também diferem dos 11 IDs de Construction-PPE: **não valide esses pesos
diretamente com aquele YAML**. Use o mapeamento correspondente ao checkpoint.
`evaluate-cascade` faz uma avaliação separada por nomes canônicos, com as
limitações descritas acima.

```bash
python -m epi_monitor validate --model runs/train/yolo11n/weights/best.pt --data data/ppe.yaml --split val --name yolo11n
python -m epi_monitor validate --model runs/train/yolov8n/weights/best.pt --data data/ppe.yaml --split val --name yolov8n
python -m epi_monitor validate --model models/ppe/absence.pt --data data/ppe-absence-transfer.yaml --split val --imgsz 640 --name absence_val
```

`plots=True` gera matriz de confusão, PR/F1/precision/recall e exemplos
anotados. `validation.json` registra mAP50, mAP50–95, precisão, recall, dados das
curvas/matriz quando fornecidos pelo backend, versões, dispositivo e paths dos
artefatos. `save_json=True` mantém predições da validação. Confiança padrão
0,001 preserva a faixa da curva PR; não confunda com o limiar da demonstração.

Se o conjunto não tiver anotações válidas de uma classe, suas métricas não
servem para concluir desempenho nessa classe. No RF100, cinco das dez saídas
de `absence.pt` têm métricas `null` e `evaluated: false`. O mAP médio corresponde
às cinco classes anotadas. Inspecione erros, não apenas mAP.

Use `--split test` somente depois de fixar modelo e limiares com a validação.
Registre métricas do detector de pessoas: qualquer pessoa perdida pelo
primeiro estágio também fica sem análise no segundo. Para avaliar estados,
anote por pessoa os EPIs presentes/ausentes, visibilidade, oclusão e associação;
reporte falsos alertas, omissões e proporção de observações inconclusivas.

## ONNX e quantização

ONNX FP32 é serialização para outro runtime; não é quantização INT8.

```bash
python -m pip install -e ".[onnx]"
python -m epi_monitor export --model models/ppe/absence.pt --format onnx --precision fp32 --imgsz 640
python -m epi_monitor detect --ppe-model models/ppe/absence.onnx --source data/demo.mp4 --max-frames 100
```

O `absence.pt` promovido é PyTorch. Nesta instalação, os exports OpenVINO
**FP32 em 640** de pessoas e EPI foram preparados e reavaliados em 29/09;
a interface usa esse perfil automaticamente em CPU quando disponível.
Veja [configuração, reprodução e medições](PRESENTATION_CHECK.md).
Em um clone novo, os exports precisam ser gerados e avaliados separadamente.
O INT8 já existente de `best.pt` pertence ao experimento histórico e não serve
como versão quantizada de `absence.pt`.

Para preparar uma nova calibração INT8, crie
`data/ppe-absence-calibration.yaml` com o conteúdo abaixo. A entrada `val`
aponta intencionalmente **somente para o treino**. Use esse arquivo apenas
para exportação, nunca para treinar ou avaliar:

```yaml
path: datasets/ppe-absence-transfer
train: train/images
val: train/images
names: [Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery, vehicle]
```

OpenVINO INT8 em CPU, após criar essa configuração:

```bash
python -m pip install -e ".[openvino]"
python -m epi_monitor export --model models/ppe/absence.pt --format openvino --precision int8 --data data/ppe-absence-calibration.yaml --fraction 1 --imgsz 640 --device cpu
python -m epi_monitor validate --model models/ppe/absence_int8_openvino_model --data data/ppe-absence-transfer.yaml --split val --imgsz 640 --name absence_openvino_int8
```

TensorRT deve ser instalado no ambiente NVIDIA com CUDA/driver compatíveis.
O extra `.[tensorrt]` declara dependências Python, mas não instala driver/CUDA.
Após preparar esse ambiente:

```bash
python -m epi_monitor export --model models/ppe/absence.pt --format engine --precision fp16 --device cuda:0
python -m epi_monitor export --model models/ppe/absence.pt --format engine --precision int8 --data data/ppe-absence-calibration.yaml --device cuda:0
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

O exportador usa a entrada `val` do YAML para calibrar. A avaliação deve usar
o YAML de dez nomes com splits separados, `ppe-absence-transfer.yaml`.
No experimento histórico, `construction-ppe-calibration.yaml` disponibilizou
340 imagens de treino com `--fraction 0.3`, e o NNCF coletou estatísticas de
300. Esses números não descrevem uma quantização do novo `absence.pt`.
Para dados próprios, crie uma configuração com a mesma separação. Use imagens
representativas de iluminação, distância, câmeras e classes, sem consumir o
teste para calibrar. Reavalie INT8 contra FP32 sobre o mesmo split. O CLI bloqueia
INT8 sem dataset e ONNX INT8, que não faz parte desta implementação.

## Benchmark reproduzível

O dashboard usa `runner.postScriptGC = false` em `.streamlit/config.toml`.
Na versão fixada do Streamlit, a configuração padrão força `gc.collect(2)`
ao terminar cada fragmento, incluindo cada análise de vídeo. Desativar essa
coleta completa por fragmento reduziu o custo da interface no teste local;
o coletor automático do Python continua ativo. Não há pausa adicional após
a inferência: seu tempo já conta no limite de análises selecionado.
O histórico mantém até 300 registros sem guardar todos os frames, e a captura
é liberada ao parar ou por inatividade. Uma sessão longa ainda precisa de
avaliação de memória. As medições da interface e seus limites estão em
[PRESENTATION_CHECK.md](PRESENTATION_CHECK.md); não confunda FPS da interface
com o benchmark abaixo, que não usa Streamlit.

```bash
python -m epi_monitor benchmark --model models/ppe/absence.pt --source data/demo.mp4 --frames 100 --warmup 10 --output runs/benchmark-absence-pt.json
python -m epi_monitor benchmark --model models/ppe/absence.onnx --source data/demo.mp4 --frames 100 --warmup 10 --output runs/benchmark-absence-onnx.json
python -m epi_monitor benchmark --model models/yolo11n.pt --ppe-model models/ppe/absence.pt --source data/demo.mp4 --frames 100 --warmup 10 --output runs/benchmark-cascade-absence-pt.json
python -m epi_monitor benchmark --model models/yolo11n.pt --ppe-model models/ppe/absence_int8_openvino_model --source data/demo.mp4 --frames 100 --warmup 10 --output runs/benchmark-cascade-absence-int8.json
```

O primeiro frame é usado para warmup; os próximos são medidos. Meça o mesmo
vídeo fornecido por você (`data/demo.mp4` é apenas um caminho de exemplo).
Os comandos com ONNX ou INT8 exigem que esses artefatos tenham sido exportados.
Para estudar o experimento anterior, use explicitamente `models/ppe/best.pt`
e `models/ppe/best_int8_openvino_model`, com nomes de relatório próprios.
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
