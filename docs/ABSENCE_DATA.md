# Dados e treinamento para ausência de capacete e colete

O experimento `ppe_absence` acrescenta supervisão explícita para **sem colete**
e novos exemplos de **sem capacete**. O Construction-PPE usado no primeiro
treino não possui `no_vest`. Não detectar uma caixa de colete continua sendo
evidência insuficiente para afirmar sua ausência: o pipeline exige uma
detecção negativa associada à pessoa e mantém casos ambíguos como inconclusivos.

## Origem e divisão dos dados

Foi utilizada a versão RF100 de
[construction-safety-gsnvb, espelhada por LibreYOLO](https://huggingface.co/datasets/LibreYOLO/construction-safety-gsnvb/tree/342e545489a6b5f76d6c8225f1ef2629c5a4770a),
fixada na revisão `342e545489a6b5f76d6c8225f1ef2629c5a4770a`.
O publicador declara **CC BY 4.0**. A atribuição preservada é
**Anonymous (computer-vision/worker-safety); Roboflow 100; espelho LibreYOLO**.
O projeto de origem é
[worker-safety](https://universe.roboflow.com/computer-vision/worker-safety).
Os READMEs originais acompanham o download.

| Split | Imagens | Capacete | Sem capacete | Sem colete | Pessoa | Colete |
|---|---:|---:|---:|---:|---:|---:|
| Treino | 997 | 2.116 | 94 | 741 | 2.362 | 1.073 |
| Validação | 119 | 232 | 11 | 90 | 241 | 141 |
| Teste | 90 | 195 | 24 | 61 | 214 | 129 |

As colunas de classes contam **caixas anotadas**, não pessoas únicas. São
1.206 imagens e 7.724 caixas no total. A exportação original aplica orientação
automática e redimensionamento por esticamento para 640 × 640, sem aumento
de dados na exportação. O treinamento pode aplicar suas próprias transformações.

A auditoria estrutural completa de 24/09/2026, registrada em
`runs/ppe-absence/audit-original-final.json`, aprovou as 1.206 imagens:
nenhuma imagem ilegível, rótulo ausente, erro, aviso ou duplicata exata
identificada entre os splits. O arquivo anterior `audit-original.json` foi
gerado durante o download incompleto e não representa o conjunto final.
Hashes verificam cópias idênticas; não descartam cenas parecidas ou quadros
próximos de uma mesma gravação.

Há forte desequilíbrio em capacetes: **94 caixas de sem capacete contra
2.116 de capacete no treino**, com apenas 11 negativas na validação.
Uma única detecção muda o recall dessa classe em cerca de 9,1 pontos
percentuais na validação. A avaliação deve apresentar TP, FP e FN junto
às porcentagens.

## Pesos iniciais e mapeamento das classes

O ponto de partida é um YOLO11n externo de
[yihong1120/Construction-Hazard-Detection](https://huggingface.co/yihong1120/Construction-Hazard-Detection/tree/212ee245136b4e330f84b409f67f3d35eff59f42),
revisão `212ee245136b4e330f84b409f67f3d35eff59f42`, arquivo
`models/yolo11/pt/yolo11n.pt`. O publicador declara **AGPL-3.0**.
A cópia local é `models/ppe/absence-base.pt`, acompanhada de procedência.
Seu SHA-256 é
`f55600c106ba7952c64d2aec70dc673240b422bdbd989d395d301b0d5426b02b`.

Os pesos carregados têm dez saídas reais: `Hardhat`, `Mask`, `NO-Hardhat`,
`NO-Mask`, `NO-Safety Vest`, `Person`, `Safety Cone`, `Safety Vest`,
`machinery` e `vehicle`. A descrição atual do repositório pode se referir
a outras arquiteturas ou versões; o preparo verifica os nomes do próprio
arquivo `.pt`.

Para manter a cabeça de classificação compatível com esse ponto de partida,
o derivado altera somente os IDs das anotações:

| ID original | Classe original | ID nos pesos | Nome nos pesos |
|---:|---|---:|---|
| 0 | helmet | 0 | Hardhat |
| 1 | no-helmet | 2 | NO-Hardhat |
| 2 | no-vest | 4 | NO-Safety Vest |
| 3 | person | 5 | Person |
| 4 | vest | 7 | Safety Vest |

O script preserva coordenadas, conteúdo das imagens e participação em cada
split. Ele cria cópias independentes das imagens e novos rótulos em
`data/datasets/ppe-absence-transfer/`, sem modificar os originais em
`data/datasets/ppe-absence/`. Não gera classes negativas a partir de caixas
ausentes. IDs incompatíveis ou ambíguos nos pesos interrompem o preparo.

**Somente cinco das dez saídas possuem anotações.** Máscara, ausência de
máscara, cone, máquina e veículo não estão validados por esse experimento;
não há garantia de manutenção de seu desempenho após o ajuste. O relatório
marca essas classes como `evaluated: false`, com métricas `null`. A auditoria
do derivado tem 15 avisos esperados de classes ausentes: cinco em cada split.
Esses avisos não são imagens corrompidas nem rótulos faltantes. O mAP agregado
se refere às cinco classes anotadas, não a dez classes ou aos alertas finais.

## Reprodução

Execute na raiz do repositório, com a venv preparada e ativada. No Windows,
é possível substituir `python` por `.venv\Scripts\python.exe` em todos os
comandos. Os dois primeiros scripts acessam a rede; o preparo da transferência
e o treinamento usam os arquivos já baixados.

```bash
python scripts/prepare_absence_data.py --workers 4
python scripts/prepare_absence_model.py
python -m epi_monitor audit-data --data data/ppe-absence.yaml --require-test --output runs/ppe-absence/audit-original-final.json
python scripts/prepare_absence_transfer.py
python -m epi_monitor audit-data --data data/ppe-absence-transfer.yaml --require-test --output runs/ppe-absence/audit-transfer.json
python -m epi_monitor train --model models/ppe/absence-base.pt --data data/ppe-absence-transfer.yaml --epochs 10 --imgsz 416 --batch 8 --workers 0 --seed 42 --device cpu --freeze 10 --project runs/train --name ppe_absence
```

O download verifica os hashes remotos de cada arquivo e recusa sobrescrever
um cache divergente. `source-plan.json` registra a revisão fixada e os arquivos.
Cada chamada gera `provenance.<splits>.json`; no preparo original, validação
e treino/teste foram baixados em chamadas separadas, produzindo
`provenance.valid.json` e `provenance.train-test.json`. Um download de todos
os splits em uma única chamada gera `provenance.train-valid-test.json`.

O derivado registra hashes de origem e destino, mapeamento, nomes efetivos,
SHA-256 dos pesos e divisões em `transfer.provenance.json`. A repetição com
conteúdo idêntico reutiliza os arquivos; uma divergência é preservada e
reportada como erro para revisão.

O treinamento local de dez épocas foi concluído em 24/09/2026. O registro
`runs/train/ppe_absence/training.json` confirma CPU, imagem 416, batch 8,
seed 42, workers 0, AMP desativado e `freeze=10` nas primeiras camadas.
O congelamento reduz o custo do ajuste; dez épocas não demonstram, por si
só, convergência. Foram usados Python 3.12.14, PyTorch 2.8.0 e
Ultralytics 8.3.203.

O melhor checkpoint dessa execução está em
`runs/train/ppe_absence/weights/best.pt`, com SHA-256
`14ac39c777e94864006bf5842a941c00e1b50a7b21f60e3b676d4beab6949d0b`.
O registro também guarda hashes dos pesos iniciais, pesos finais, YAML e
manifestos. Repetir um nome existente cria uma pasta com sufixo; use o
caminho devolvido pela nova execução nos comandos seguintes.

Após escolher os pesos pela validação, a promoção para o caminho padrão é:

```bash
python scripts/promote_absence_model.py --run-dir runs/train/ppe_absence
```

O script verifica o SHA-256 de `best.pt`, sua correspondência com
`training.json`, os nomes/IDs e a presença única de classes de capacete,
colete e suas ausências. Publica `models/ppe/absence.pt` e
`models/ppe/absence.provenance.json` sem sobrescrever conteúdo divergente.
Repetir a mesma promoção reutiliza conteúdo idêntico. Os pesos anteriores
`models/ppe/best.pt` e seu export INT8 permanecem históricos e independentes.
A promoção não mede precisão; novos exports do `absence.pt` precisam de
calibração e avaliação próprias, conforme [ML.md](ML.md).

## Validação e teste

Para curvas PR, matriz de confusão e mAP do segundo detector, use o YAML de
dez nomes, compatível com seus IDs reais. Para a cascata, o YAML original de
cinco classes é apropriado: o avaliador normaliza nomes e mede as caixas
finais dos dois estágios.

```bash
python -m epi_monitor validate --model runs/train/ppe_absence/weights/best.pt --data data/ppe-absence-transfer.yaml --split val --imgsz 640 --device cpu --project runs/validate --name ppe_absence_val
python -m epi_monitor evaluate-cascade --person-model models/yolo11n.pt --ppe-model runs/train/ppe_absence/weights/best.pt --data data/ppe-absence.yaml --split val --imgsz 640 --confidence 0.4 --iou 0.45 --match-iou 0.5 --device cpu --output runs/ppe-absence/trained-val.json
```

Compare candidatos na mesma validação e configuração; escolha modelo e
limiares antes de medir o teste reservado. Após essa escolha, a avaliação
final da cascata usa `--split test`, conservando os demais parâmetros e
gravando outro relatório. Não ajuste o modelo ou limiares para melhorar
resultados observados no teste. As métricas medidas e a decisão de adoção
devem ser consultadas em [VALIDATION.md](VALIDATION.md).

`evaluate-cascade` mede precisão/recall em confiança fixa, com pareamento
de caixas por classe; não calcula mAP nem valida a associação de EPI a cada
pessoa ou a confirmação temporal dos alertas. O relatório do treinamento
também não substitui essa avaliação do fluxo que roda na aplicação.

## Limitações identificadas

Na revisão visual da validação, o arquivo
`ppe_0441_jpg.rf.74dcd28b13e10eb7abaa7fea2e5e3bd5.jpg` apresenta uma caixa
original de `no-vest` sobre a região da cabeça, com centro normalizado
aproximado `(0,524; 0,139)`. A imagem exibe capacete e colete. Trata-se de
um problema semântico que passa pela auditoria de formato.
**O rótulo e a imagem foram mantidos intactos na avaliação**, tanto na origem
quanto no derivado; a transferência apenas mapeia o ID 2 para 4. Uma futura
revisão de anotações deve gerar outra versão documentada do dataset,
aplicando critérios consistentes e preservando os resultados da versão original.

A sobreposição entre o treinamento externo dos pesos e este dataset público
é **desconhecida**. Os splits preservados permitem separar o ajuste local
da avaliação, mas não comprovam independência em relação ao pré-treinamento.
As imagens públicas também incluem cenários diferentes da instalação do TCC.
Validar no ambiente real exige imagens ou vídeos locais separados do treino,
com revisão humana de capacete, colete, ausência, oclusão e pessoa associada.
Até essa etapa, os resultados sustentam um experimento público reproduzível,
sem comprovar a precisão dos alertas no local de instalação.
