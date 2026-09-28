# Aderência ao roteiro e ao artigo do TCC

Revisão baseada nos documentos fornecidos pelo usuário: **Roteiro para
desenvolvimento do projeto.docx** e **ARTIGO_CC_REV1.docx**, incluindo os
comentários de revisão. Os documentos são fontes de requisitos e de afirmações
a verificar. Seus arquivos originais não foram alterados.

**A implementação anterior não atendia integralmente ao núcleo do roteiro.**
A demonstração COCO mostrava pessoas e havia infraestrutura de interface,
persistência e alertas, mas isso não demonstrava detecção de EPI. A revisão
prioriza o Projeto 1 do roteiro: dois modelos, análise de imagem/vídeo,
resultados explicáveis e avaliação reproduzível. O usuário informou que não
possui pesos, dataset anotado ou artefatos do resultado de mAP citado no artigo.

## Matriz de aderência

| Requisito dos documentos | Implementação atual | Situação e evidência necessária |
| --- | --- | --- |
| Detectar pessoa e então invocar outro modelo para EPI | `CascadePipeline` em `detection.py`: YOLO de pessoas habilita YOLO de EPI | Implementado; segundo estágio não executa sem pessoa. Testes verificam a condição. |
| Análise por imagem e vídeo | `detect --source` aceita imagem, vídeo, webcam e RTSP | Implementado; janela OpenCV opcional e relatório JSONL independente da interface. |
| Responder OK / Não seguro | Estados `ok`, `unsafe` e `uncertain` por pessoa | Implementado com incerteza explícita; falta validar a classificação final em cenas rotuladas por pessoa. |
| Capacetes e coletes | `absence.pt` com classes positivas e negativas explícitas; associação por região | Implementado e medido em dados públicos; recall de sem capacete ainda baixo e decisão por pessoa pendente de validação local. Outros EPIs exigem classes, regras e avaliação próprias. |
| Armazenar imagem, câmera e momento | `EventStore` salva JPEG e JSON, com contexto de câmera e horário UTC | Implementado localmente; não é um banco de gestão multicâmera. |
| Telegram com imagem e horário | Transporte direto `sendPhoto`, configurável na interface | Implementado e testável sem envio real; entrega real depende do bot e destino do operador. |
| Metabase → N8N → Telegram | Não existe esse encadeamento | Pendente. O transporte direto é uma escolha do protótipo e difere do roteiro. |
| Dashboard Metabase com frontend Next | Visualização simples Streamlit; CLI funciona sem ela | Pendente em relação à tecnologia do roteiro; interface não é o foco desta etapa. |
| Analista confirma ou descarta irregularidades e alimenta gráficos | Consulta/baixar evidências | Pendente: não há fluxo persistente de revisão humana nem base de rótulos revisados. |
| Containerização | Dockerfile e Compose existentes | Implementados; acesso à câmera depende do host. Execução do container deve constar nas evidências quando testada. |
| YOLOv5 descrito no artigo e solicitado nos comentários | Modelo atual: YOLO11n para pessoa + YOLO11n ajustado localmente para EPI | Divergência explícita. Falta comparação com YOLOv5 ou justificativa e atualização formal do texto. |
| Treinamento com dados públicos e coleta in loco | Treino inicial em Construction-PPE e ajuste de 10 épocas em RF100, com ausência explícita de colete, auditoria e avaliação | Parcial: os dois experimentos locais foram executados; coleta/anotação in loco e avaliação no cenário final pendentes. |
| Precisão, recall, mAP, FP/FN, FPS/tempo de resposta | `validate`, `evaluate-cascade`, `benchmark` e relatórios | Ferramentas implementadas; resultados e limitações reais constam em `VALIDATION.md`. |
| mAP@0.5 = 0,841 apresentado no resumo | Sem pesos, split, logs e execução que sustentem o número para este TCC | **Não comprovado. Não apresentar como resultado próprio.** |

## Decisões para o núcleo de detecção

Um candidato público YOLOv8n de EPI foi avaliado e descartado como modelo
principal por desempenho insuficiente, especialmente para coletes. Em seguida,
foi executado treinamento local de YOLO11n por 10 épocas em Construction-PPE,
preservado em `models/ppe/best.pt`. A etapa seguinte ajustou um YOLO11n externo
com classes de ausência explícita usando RF100 construction-safety-gsnvb:
997 imagens de treino, 119 de validação e 90 de teste, dez épocas e primeiras
dez camadas congeladas. O resultado promovido, `models/ppe/absence.pt`, é o
padrão atual. A promoção verifica hash e correspondência de classes com o
treinamento e preserva os arquivos anteriores, incluindo seu export INT8.

No teste RF100, a cascata encontrou 39 das 61 caixas de sem colete (recall
63,93%, precisão 69,64%) e 6 das 24 de sem capacete (recall 25%, precisão
66,67%). Esse recall baixo mantém a detecção de ausência como limitação.
A comparação no mesmo teste Construction-PPE também revelou uma troca:
sem capacete passou de 5/40 para 10/40 acertos, com falsos positivos de 1
para 13; colete caiu de 139/178 para 126/178 acertos. Portanto, a mudança
amplia a cobertura de classes, mas não demonstra superioridade universal.

Resultados e parâmetros completos estão em [VALIDATION.md](VALIDATION.md).
O ajuste local utiliza dados públicos e pesos externos com sobreposição
prévia desconhecida; não representa coleta in loco, convergência garantida
ou validação no cenário final. O detector tem dez saídas, mas somente cinco
classes anotadas nesse experimento. Procedência e protocolo constam em
[ABSENCE_DATA.md](ABSENCE_DATA.md).
O primeiro YOLO detecta pessoas. Se houver alguma, o segundo examina o quadro
completo uma vez, preservando o contexto de treinamento; as caixas são então
associadas às pessoas. Isso atende à condição de duas instâncias do roteiro
sem exigir um recorte por pessoa.

A resposta binária é insuficiente quando o equipamento não aparece por
oclusão ou falha do modelo. `ok` significa que os EPIs exigidos foram
detectados, `unsafe` exige uma classe negativa explícita sem associação
ambígua, e `uncertain` preserva casos inconclusivos. `ok` não certifica
conformidade normativa; `unsafe` é uma observação a revisar. Em especial,
ausência de detecção não é ausência comprovada de EPI.

O dataset do treinamento inicial não inclui `no_vest`; seus pesos históricos
podem reconhecer colete, mas não sua ausência explícita. O modelo atual inclui
`NO-Safety Vest`, aprendida com caixas negativas anotadas. Em ambos os casos,
deixar de encontrar colete continua inconclusivo. A interface informa a
capacidade real do modelo carregado e separa o estado de capacete e colete
por pessoa; a classe negativa precisa ser associada sem ambiguidade para
produzir uma observação `unsafe`.

Eventos de vídeo usam o tempo da fonte para confirmar persistência; webcam e
RTSP usam tempo monotônico de observação. A data/hora da evidência continua
sendo o instante da análise, não o instante original da filmagem. Imagem
estática é uma observação única e não simula uma confirmação de dois segundos.

## O que precisa ser concluído para sustentar o TCC

1. Definir o cenário e o escopo final: capacete e colete, distâncias, câmeras,
   iluminação, situações de oclusão e significado dos estados.
2. Coletar e anotar dados do cenário, incluindo positivos, negativos e casos
   inconclusivos. Separar por sessão/câmera; não distribuir frames vizinhos
   aleatoriamente entre treino e teste.
3. Auditar rótulos, duplicatas e cobertura. Um split chamado `test` de um
   dataset público não garante independência dos pesos pré-treinados.
4. Treinar e registrar sementes, versões, hiperparâmetros, hashes dos pesos,
   composição dos splits e gráficos. O número de épocas é configuração,
   não evidência de convergência ou de generalização.
5. Comparar YOLOv5 com a arquitetura candidata nos mesmos dados e hardware.
   Medir o detector de pessoa, o detector de EPI e a cascata completa: erros do
   primeiro estágio também podem impedir a análise do segundo.
6. Medir mAP e PR dos detectores, TP/FP/FN do fluxo completo, qualidade da
   decisão por pessoa e latência/FPS. A avaliação das caixas não substitui
   a validação dos estados ou do comportamento temporal.
7. Revisar falsos alertas e omissões nas evidências, testar Telegram com o bot
   real e registrar limitações. Implementar gestão/revisão humana e o caminho
   Metabase/N8N/Next somente se mantidos como entregáveis do escopo final.
8. Atualizar o artigo com os resultados efetivamente reproduzidos. O resumo
   usa afirmações de resultado enquanto a metodologia descreve etapas futuras;
   esse contraste precisa ser resolvido com evidências ou redação prospectiva.

Os comandos para executar essas etapas estão em [ML.md](ML.md). Os resultados
já executados e seus limites estão em [VALIDATION.md](VALIDATION.md). Esta
matriz não converte funções implementadas em alegações de precisão científica.
