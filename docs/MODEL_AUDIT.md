# Auditoria do modelo — 01/10/2026

**O modelo ativo aprendeu a detectar capacete e colete, mas ainda não está
validado como um sistema confiável de alertas de ausência.** A convergência do
treino não foi demonstrada, as classes negativas têm poucos exemplos e o
ambiente real ainda não foi avaliado. Testes de software aprovados não eliminam
essas limitações estatísticas.

Esta auditoria lê os resultados e as anotações locais; não representa um novo
treino ou uma nova medição de acurácia. A referência de implementação é o estado
anterior às melhorias experimentais de 01/10. Os números abaixo pertencem aos
pesos e configurações identificados; não devem ser atribuídos automaticamente
a modelos ou limiares posteriores.

## O treino foi suficiente?

O peso ativo `models/ppe/absence.pt` tem SHA-256
`14ac39c777e94864006bf5842a941c00e1b50a7b21f60e3b676d4beab6949d0b`.
Foi ajustado a partir de um YOLO11n externo durante **10 épocas**, em CPU,
resolução 416, batch 8, seed 42 e primeiras dez camadas congeladas.
O tempo registrado das épocas foi **2.192,51 s, cerca de 36,5 minutos**.

| Indicador | Época 1 | Época 10 |
| --- | ---: | ---: |
| mAP@0,5 de validação | 0,77568 | 0,83693 |
| mAP@0,5:0,95 de validação | 0,43281 | 0,47940 |
| Perda de classificação no treino | 1,08609 | 0,74347 |
| Perda de classificação na validação | 0,77410 | 0,70212 |

O melhor mAP@0,5:0,95 apareceu na **última época**. As perdas de treino e
validação ainda diminuíam. Isso justifica experimentar uma continuação
controlada, mas **não demonstra que mais épocas necessariamente melhorarão o
teste**. As curvas também não bastam para afirmar sobreajuste: não há subida
clara da perda de validação nesse intervalo. É preciso comparar candidatos
pela validação, por classe, e conservar o teste fora das decisões.

O mAP 0,837 do relatório de treinamento é da validação em 416; o mAP 0,7506
do teste nativo e o 0,7592 do teste OpenVINO são medições distintas em 640.
Não são números intercambiáveis nem estimativas de precisão dos alertas.

## Gargalos dos dados

Contagens extraídas dos manifestos auditados; caixas da mesma imagem não são
amostras independentes de cenário.

| Dataset / classe | Treino: imagens / caixas | Validação: imagens / caixas | Teste: imagens / caixas |
| --- | ---: | ---: | ---: |
| RF100: capacete | 938 / 2.116 | 117 / 232 | 82 / 195 |
| RF100: sem capacete | **49 / 94** | **6 / 11** | 11 / 24 |
| RF100: colete | 504 / 1.073 | 74 / 141 | 57 / 129 |
| RF100: sem colete | 390 / 741 | 52 / 90 | 31 / 61 |
| Construction-PPE: bota | 530 / 1.235 | 64 / 151 | 75 / 211 |
| Construction-PPE: sem bota | **28 / 88** | **2 / 4** | 6 / 23 |

No treino RF100, capacetes são aproximadamente 22,5 vezes mais frequentes que
cabeças sem capacete. Em resolução 416, o menor lado de 11 das 94 caixas de
sem capacete tem menos de 16 pixels; a mediana é 27,5 pixels. Essa é uma
aproximação geométrica a partir das anotações, antes das transformações de treino.

Na avaliação de validação da cascata, os **seis falsos negativos de sem
capacete se concentram em apenas duas imagens**: cinco em `ppe_0164...` e um
em `ppe_0260...`. As caixas têm menor lado de 17,5 a 29,5 pixels na fonte 640.
Portanto, pequenas cabeças em cenas com várias pessoas merecem revisão visual.
Escolher uma configuração para acertar somente essas duas cenas seria uma
forma frágil de selecionar o modelo.

As auditorias estruturais não encontraram imagens inválidas ou duplicatas
exatas entre splits. **Isso não verifica a correção semântica das anotações**
nem detecta todos os quadros parecidos. Há ao menos um rótulo suspeito de
sem colete sobre a cabeça, documentado em [ABSENCE_DATA.md](ABSENCE_DATA.md).
A sobreposição com os dados usados no pré-treino externo é desconhecida.

## O que os pesos disponíveis realmente cobrem

O `absence.pt` tem classes explícitas de capacete, colete e suas ausências,
mas nenhuma classe de bota. No teste da cascata com confiança 0,40 e IoU de
correspondência 0,50:

| Classe | TP / FP / FN | Precisão | Recall |
| --- | --- | ---: | ---: |
| Capacete | 178 / 30 / 17 | 85,58% | 91,28% |
| Colete | 93 / 15 / 36 | 86,11% | 72,09% |
| Sem capacete | 6 / 3 / 18 | 66,67% | **25,00%** |
| Sem colete | 39 / 17 / 22 | 69,64% | 63,93% |

O peso histórico `models/ppe/best.pt` e o dataset Construction-PPE já estão
disponíveis localmente para uma investigação de botas. Na validação individual
do detector histórico em 141 imagens de teste, **bota obteve AP@0,5 0,6935,
precisão 78,08% e recall 59,09%**; **sem bota teve recall zero e AP@0,5 0,0153**.
O export INT8 histórico também teve recall zero de sem bota. Essas métricas
usam o procedimento do detector individual, não os limiares da tabela acima.

Ter um nome de classe no arquivo não basta: a cascata base só associava
capacete/colete. Para botas, é necessário associar caixas à região dos pés,
tratar pessoas sobrepostas e pés fora do quadro, medir essa associação e
mostrar o resultado como experimental até validá-lo. Ausência de uma caixa
de bota não é prova de que a pessoa está sem bota. Imagens comuns também não
comprovam a certificação ou as propriedades de proteção do calçado.

## Melhorias de maior impacto

1. **Revisar e ampliar dados negativos**, priorizando sem capacete e sem bota,
   com novos cenários e câmeras. Revisar todos os exemplos raros existentes;
   corrigir erros em uma nova versão, preservando a original e a procedência.
   Repetir uma imagem rara pode ajudar a amostragem, mas não cria diversidade.
2. **Comparar limiares somente na validação**, por classe, registrando TP, FP,
   FN e latência. Reduzir confiança pode recuperar casos, mas também aumenta
   falsos alarmes. Não declarar ganho de treino quando só mudou um limiar.
3. **Investigar botas com os artefatos locais**, mantendo o caminho atual de
   capacete/colete estável. Se um terceiro detector for usado, medir o custo
   extra de CPU. Um modelo unificado exige anotações completas e coerentes:
   misturar datasets com EPIs não anotados pode ensinar falsos fundos.
4. **Comparar um treino mais longo**, selecionando pela validação e usando
   outra pasta. Continuar congelado em 416 é o experimento inicial de menor
   custo; 640 e descongelamento são candidatos posteriores, com mais custo.
   Não promover pesos automaticamente pelo mAP médio quando a classe crítica
   piora. Registrar resultados negativos também.
5. **Medir alertas por pessoa e por evento em vídeos reais**. O avaliador de
   caixas não mede associação, continuidade de identidade, confirmação de
   dois segundos ou falsos alarmes por hora.

Exemplo de candidato controlado com a CLI existente, ainda não executado por
esta auditoria. Não sobrescreve os pesos ativos nem o experimento anterior:

```powershell
.\.venv\Scripts\python.exe -m epi_monitor train --model models/ppe/absence.pt --data data/ppe-absence-transfer.yaml --epochs 20 --imgsz 416 --batch 8 --workers 0 --seed 42 --device cpu --freeze 10 --project runs/train --name ppe_absence_candidate_20
```

São vinte novas épocas a partir dos pesos, não uma retomada exata do estado do
otimizador. O histórico de 36,5 minutos por dez épocas indica que esse trabalho
pode durar mais de uma hora e ocupar bastante a CPU; não é uma previsão
garantida. Avalie depois em 640, usando o caminho efetivo devolvido pelo treino,
na mesma configuração de validação do modelo ativo. Preserve os pesos ativos
e os exports até concluir a comparação.

## Critérios para considerar o trabalho finalizado

- Escopo acadêmico fechado: quais EPIs são obrigatórios e quais integrações
  do roteiro permanecem exigidas.
- Dados locais separados por sessão/câmera, com teste final reservado e
  revisão humana consistente de oclusões e regiões fora do quadro.
- Metas explícitas de precisão, recall, falsos alarmes por hora e atraso de
  alerta aceitas pelo projeto; relatar também a taxa de inconclusivos.
- Presença e ausência de cada EPI obrigatório avaliadas com amostra suficiente
  e incerteza. Onze ou quatro caixas negativas na validação são insuficientes
  para uma alegação forte de generalização.
- Sessão prolongada de câmera/RTSP, reconexão, memória/CPU e entrega ao Telegram
  exercitadas no ambiente do operador. Testes simulados não confirmam a rede,
  a câmera ou as credenciais reais.

Até isso ocorrer, o enquadramento correto é **protótipo demonstrável com
validação em dados públicos e limitações conhecidas**.

## Evidências locais

- `runs/train/ppe_absence/results.csv`, `training.json`, `args.yaml` e
  `dataset-audit.json`: treino ativo e distribuição.
- `runs/train/ppe_tcc/results.csv`, `training.json` e `dataset-audit.json`:
  treino e dados do candidato histórico de botas.
- `runs/ppe-absence/finetuned-val.json`: localização dos erros de validação.
- `runs/validate/ppe_tcc_test/validation.json` e
  `runs/validate/ppe_tcc_int8_test/validation.json`: botas no detector histórico.
- `runs/presentation-review/production-openvino-test.json`: cascata ativa.
- [Resumo reproduzível desta auditoria](experiments/model-audit-2026-10-01.json),
  com hashes dos arquivos consultados e contagens sem copiar o dataset.
- [Revisão de apresentação](PRESENTATION_CHECK.md): desempenho e testes de
  software já realizados, com suas condições.
