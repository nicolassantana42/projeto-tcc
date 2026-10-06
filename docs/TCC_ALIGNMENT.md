# Aderência ao roteiro e ao artigo do TCC

> Atualizado em 06/10/2026.

Revisão baseada nos documentos fornecidos pelo usuário: **Roteiro para
desenvolvimento do projeto.docx** e **ARTIGO_CC_REV1.docx**, incluindo os
comentários de revisão. Os documentos são fontes de requisitos e de afirmações
a verificar. Seus arquivos originais não foram alterados.

## Matriz de aderência

| Requisito dos documentos | Implementação atual | Situação e evidência necessária |
| --- | --- | --- |
| Detectar pessoa e então invocar outro modelo para EPI | `CascadePipeline` em `detection.py`: YOLO de pessoas habilita YOLO de EPI | Implementado; segundo estágio não executa sem pessoa. Testes verificam a condição. |
| Análise por imagem e vídeo | `detect --source` aceita imagem, vídeo, webcam e RTSP | Implementado; janela OpenCV opcional e relatório JSONL independente da interface. |
| Responder OK / Não seguro | Estados `ok`, `unsafe` e `uncertain` por pessoa | Implementado com incerteza explícita; falta validar a classificação final em cenas rotuladas por pessoa. |
| Capacetes, coletes e botas | `epi.pt` (YOLO11n, Construction-PPE, 50 épocas) com capacete, colete, bota e ausências de capacete/bota; associação por região | Implementado. Teste (141 imagens, 480 px): mAP50 capacete 0,94, colete 0,89, bota 0,76. Sem classe "sem colete" no dataset. |
| Armazenar imagem, câmera e momento | `EventStore` salva JPEG e JSON, com contexto de câmera e horário UTC | Implementado localmente; não é um banco de gestão multicâmera. |
| Telegram com imagem e horário | Transporte direto `sendPhoto`, configurável na interface | Implementado e testável sem envio real; entrega real depende do bot e destino do operador. |
| Metabase → N8N → Telegram | Envio direto pela API do Telegram | **Trabalho futuro** declarado no texto; o envio direto cumpre a função de alerta no protótipo. |
| Dashboard Metabase com frontend Next | Painel Streamlit com KPIs, gráfico de conformidade, ocorrências e métricas do modelo | **Trabalho futuro** quanto à tecnologia; funcionalidade de painel atendida em Streamlit. |
| Analista confirma ou descarta irregularidades e alimenta gráficos | Botões Confirmar/Descartar na aba Ocorrências; revisão salva no `event.json`; KPIs de pendentes, confirmadas e precisão revisada | Implementado. |
| Containerização | Dockerfile e Compose existentes | Implementados; acesso à câmera depende do host. Execução do container deve constar nas evidências quando testada. |
| YOLOv5 descrito no artigo e solicitado nos comentários | YOLO11n adotado (geração mais recente, nano para CPU) | Decisão de projeto; atualizar o texto do artigo (ver `APRESENTACAO.md`). |
| Treinamento com dados públicos e coleta in loco | YOLO11n treinado 10 + 40 épocas no Construction-PPE (público) | Parcial: coleta/anotação in loco fica como trabalho futuro. |
| Precisão, recall, mAP, FP/FN, FPS/tempo de resposta | `validate`, `evaluate-cascade`, `benchmark` | Medidos: mAP50 0,94 / 0,89 / 0,76 no teste; ≈ 13 FPS em CPU. Ver `MODELO.md`. |
| mAP@0.5 = 0,841 apresentado no resumo | Substituir pelos valores medidos (0,94 / 0,89 / 0,76) | **Não usar 0,841**; texto corrigido em `APRESENTACAO.md`. |

Métricas, treino e limitações do modelo: [MODELO.md](MODELO.md).
