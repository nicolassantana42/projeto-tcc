# Revisão para apresentação — ensaios de 29 e 30/09/2026

Relatório consolidado em 30/09/2026.

**O projeto serve para demonstrar o protótipo de capacete e colete, com suas
limitações declaradas. Ainda não atende a uma demonstração que prometa avaliar
capacete, colete e bota, nem está validado para operação no ambiente real.**

## O que foi corrigido

- Prévia limitada a 720 pixels, em área fixa 16:9 antes e depois de iniciar.
  Imagens 4:3 e verticais são ajustadas sem distorção; evidências mantêm o
  quadro original. Somente a prévia usa JPEG reduzido.
- Removida a espera adicional após a inferência. O limite de 3, 10, 25 ou
  30 análises/s controla a frequência máxima; não garante esse FPS.
- Desativada a coleta completa de memória que o Streamlit forçava após cada
  atualização (`runner.postScriptGC=false`). A coleta automática normal do
  Python continua ativa. Não há mudança nos pesos ou regras de detecção.
- Perfil **CPU otimizada (OpenVINO)** com os dois modelos exportados em 640,
  quatro threads e um stream. Nesta instalação CPU, ele é o padrão da interface.
  PyTorch permanece disponível e é recomendado automaticamente quando há GPU
  CUDA/MPS ou faltam os artefatos/dependências OpenVINO.
- O limite padrão de inferência PyTorch CPU é quatro threads. `EPI_CPU_THREADS`
  permite outro valor antes de iniciar o processo. O orçamento PyTorch é
  compartilhado pelo processo; OpenVINO configura cada modelo. Treino deve ser
  executado em um processo separado, como na CLI.
- Nome e comandos próprios do projeto passam a usar `epi_monitor` / `epi-monitor`.
  Relatórios antigos permanecem legíveis. Reinicie o servidor após a atualização.

## Desempenho medido nesta máquina

CPU Intel Core i7-1355U, 12 processadores lógicos, Windows, modelos em 640.
Cinco imagens de validação repetidas, após aquecimento. Medição de cascata,
sem captura, interface, gravação ou rede; não é promessa de FPS da webcam.

| Configuração | FPS | Tempo médio | Uso médio de núcleos lógicos |
| --- | ---: | ---: | ---: |
| PyTorch original, 8 threads | 3,49 | 285,8 ms | 2,63 |
| PyTorch, 4 threads | 3,19 | 313,4 ms aproximadamente | 2,30 |
| OpenVINO final, 4 threads / 1 stream | **7,31** | **136,8 ms** | 3,24 |

O OpenVINO foi cerca de 2,1 vezes mais rápido, consumindo menos CPU por quadro,
mas não menos CPU por segundo quando opera continuamente no máximo. Para
reduzir a carga total, use o limite de **3 análises/s**. O limite de 10 aproveita
mais a capacidade medida; 25/30 não acelera um modelo além da capacidade do
hardware. **25–30 FPS reais não foram atingidos**. Exigiriam outra configuração
e nova medição, por exemplo uma GPU apropriada ou modelos menores revalidados.

O ensaio preliminar OpenVINO marcou 7,59 FPS; o valor final integrado foi 7,31.
Essa variação faz parte do benchmark local. Os registros completos e a
comparação de saídas estão no [relatório da revisão](experiments/presentation-2026-09-29.json).

## Verificação da interface e do código

O teste com um vídeo local manteve a área de prévia em **720 × 405 pixels**
antes e depois de iniciar, conforme medição do DOM durante a sessão. A captura
da interface mostra **3,1 FPS**; outra leitura pontual da mesma sessão foi
**3,3 FPS**. São observações com interface, não uma média formal. Elas não
substituem o benchmark de 7,31 FPS da cascata isolada nem demonstram 25/30 FPS.

[Captura da interface durante a detecção](images/presentation-2026-09-29-ui.png).

Em **30/09**, o teste A/B usou o mesmo vídeo de 103 quadros, com OpenVINO,
limite de 10 análises/s, confiança 0,40 e IoU 0,45. Os relatórios ZIP exportados
pela própria interface permitiram medir os intervalos entre os quadros 10–100,
excluindo o aquecimento e os três quadros finais vazios:

| Configuração Streamlit | FPS com interface | Inferência média |
| --- | ---: | ---: |
| Coleta completa forçada a cada atualização | 5,43 | 124,2 ms |
| Apenas coleta automática normal do Python | **7,57** | 120,1 ms |

As caixas, scores, contagens, estados e limiares foram **idênticos em todos os
103 quadros**. O ganho foi de aproximadamente 39% nesta comparação local.
São dois ensaios curtos de uma cena pública repetida: não garantem a taxa em
webcam, outras cenas ou sessões longas. A variação entre dias também mostra
a influência da carga da máquina. A estabilidade de memória em uso prolongado
ainda precisa de ensaio. Fontes: `runs/presentation-review/ui-gc-default.zip`,
`ui-gc-normal.zip` e os respectivos arquivos `*-summary.json`.

A medição DOM repetida em 30/09 confirmou **720 × 405** antes e depois de iniciar;
o arquivo bruto é `runs/presentation-review/ui-geometry-2026-09-30.json`.
As imagens das ocorrências permaneceram em **640 × 756**, resolução da fonte.
O teste abaixo é inferência de uma imagem estática pública, sem alegação de FPS:

![Interface final com detecções e estados por pessoa](images/presentation-2026-09-30-ui.png)

O vídeo gerou uma ocorrência local no quadro 51, após a confirmação temporal,
com imagem e JSON. O registro contém três pessoas, três capacetes, um colete
e uma detecção explícita de ausência de colete. Não houve envio externo
(`deliveries: {}`). É um teste funcional com cena pública repetida, não uma
validação de acurácia em vídeo real.

A suíte automatizada concluiu **558 testes aprovados em 22,25 s**; o log está em
`runs/presentation-2026-09-29-pytest.log`. Hashes do log, pesos, exports, fontes
dos resultados e imagens estão no relatório versionado. Testes de software
não comprovam, por si só, o desempenho do modelo em campo.

Após o ajuste de coleta de memória, **46 testes de interface, prévia e perfis**
passaram novamente. A configuração foi lida pelo Streamlit como `false`, com
`gc.isenabled()` ainda `true`. Nenhuma regra de detecção foi alterada nesse ajuste.

## Capacete, colete e bota: verificação real

Os nomes foram lidos diretamente dos três arquivos `.pt`, não inferidos dos
nomes dos arquivos. O modelo ativo `absence.pt` tem `Hardhat`, `Safety Vest`,
`NO-Hardhat` e `NO-Safety Vest`, **sem classe de bota**.

O peso histórico `best.pt` possui `boots` e `no_boots`. Entretanto, a cascata
atual só associa capacete/colete e filtra outras classes. Selecionar esse peso
não torna o projeto um avaliador de botas. A avaliação anterior de `no_boots`
também teve recall zero. Não foi adicionada uma caixa ou um rótulo fictício.

Repetição da cascata otimizada em **119 imagens de validação e 90 de teste**:
mesmos TP, FP e FN do modelo nativo nos mesmos limiares (confiança 0,40,
NMS IoU 0,45, correspondência IoU 0,50). No teste:

| Classe | TP / FP / FN | Precisão | Recall |
| --- | --- | ---: | ---: |
| Capacete | 178 / 30 / 17 | 85,58% | 91,28% |
| Colete | 93 / 15 / 36 | 86,11% | 72,09% |
| Sem capacete | 6 / 3 / 18 | 66,67% | **25,00%** |
| Sem colete | 39 / 17 / 22 | 69,64% | 63,93% |
| Bota | Não avaliada pelo modelo ativo | — | — |

O detector individual OpenVINO também foi revalidado, com novas curvas PR e
matriz de confusão: mAP@0,5 **0,7592**; mAP@0,5:0,95 **0,3901**, média das
cinco classes anotadas. Não é mAP dos alertas por pessoa. Os exports mantêm
pesos FP32, sem calibração INT8 nesta revisão; backends podem diferir no
pré-processamento e na aritmética, por isso foram avaliados novamente.

Resultados completos: `runs/presentation-review/production-openvino-val.json`,
`production-openvino-test.json` e
`runs/validate/presentation_openvino_test/validation.json`.

Gráficos gerados pela nova avaliação do detector individual, com seus próprios
limiares de validação; não representam diretamente a tabela da cascata acima:

- [Curva Precision–Recall](images/presentation-pr-test.png).
- [Matriz de confusão](images/presentation-confusion-test.png).
- [Matriz de confusão normalizada](images/presentation-confusion-normalized-test.png).

Os dados públicos têm anotações imperfeitas e a sobreposição com o pré-treino
externo é desconhecida. Repetir esse teste verifica regressões; não cria um novo
teste independente. [Origem e limitações dos dados](ABSENCE_DATA.md).

## O que ainda falta

1. Treinar/selecionar pesos e implementar associação de **botas** por pessoa,
   com avaliação de presença e ausência e de casos com os pés fora do quadro.
2. Melhorar o recall de sem capacete e os falsos alertas de ausência de colete.
3. Coletar e anotar vídeo no ambiente real, separar por sessão/câmera e avaliar
   o resultado por pessoa e ao longo do tempo. Não basta medir caixas isoladas.
4. Testar webcam/RTSP reais e o bot Telegram do operador. Nesta revisão não
   houve acesso à câmera física ou envio externo.
5. Resolver as diferenças do roteiro: YOLOv5 e comparação de arquiteturas,
   Metabase/N8N/Next e fluxo persistente de revisão humana, se mantidos no escopo.

## Executar após atualizar

```powershell
cd C:\projeto-tcc\projeto-tcc
.\.venv\Scripts\python.exe run.py
```

Abra `http://localhost:8501`, mantenha **CPU otimizada (OpenVINO)** nesta máquina,
escolha a fonte e inicie. Use **3 análises/s** para aliviar a CPU ou **10** para
priorizar fluidez. O campo **FPS observado** mostra o valor realmente entregue.

Em um clone novo, os exports não estão no Git. Após preparar os pesos:

```powershell
python -m pip install -e ".[openvino]"
python -m epi_monitor export --model models/yolo11n.pt --format openvino --precision fp32 --imgsz 640
python -m epi_monitor export --model models/ppe/absence.pt --format openvino --precision fp32 --imgsz 640
```

Preserve a avaliação dos pesos e reavalie exports em outro ambiente. O modo
PyTorch funciona quando OpenVINO não estiver preparado. Detalhes de configuração
e integração estão em [ML.md](ML.md) e [ALERTS.md](ALERTS.md).
