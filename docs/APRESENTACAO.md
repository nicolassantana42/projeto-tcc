# Apresentação para a banca

## Checklist (véspera e 30 min antes)

- [ ] Copiar a **pasta inteira** do projeto, com `models/` e `.venv` (ou ter internet para a 1ª instalação).
- [ ] Python 3.12 instalado no computador da apresentação (`py -3.12 --version`).
- [ ] Dar dois cliques em `iniciar.bat` e abrir http://localhost:8501 **antes** de chamar a banca (o 1º carregamento do modelo leva alguns segundos).
- [ ] Testar a webcam em *Fonte → Webcam local* (feche Teams/Zoom, que bloqueiam a câmera).
- [ ] Tomada ligada: em bateria o Windows reduz a CPU e o FPS cai.
- [ ] Telegram (opcional): configurar e clicar **Enviar teste aos canais ativos** uma vez.

## Roteiro da demonstração (≈ 7 min)

1. **Problema** (30 s): fiscalização manual de EPI é cara e falha; proposta é apoiar o fiscal, não substituí-lo.
2. **Como funciona** (1 min): explicar o pipeline (pessoas → EPIs → associação por região do corpo → decisão por pessoa → ocorrência). Detalhes em `docs/MODELO.md`.
3. **Imagem** (1 min): *Imagem → Caminho no computador →* `data/datasets/images/test/image611.jpg` → ▶ Iniciar. Mostrar os cards por pessoa (capacete, colete, bota).
4. **Vídeo** (1,5 min): *Arquivo de vídeo → Caminho no computador →* `data/demo/demo_obra.mp4`. Mostrar os números (pessoas, EPIs completos, alertas, FPS) mudando.
5. **Webcam** (1 min): mostrar detecção ao vivo (pessoa sem EPI → "Não detectado"; explicar a decisão conservadora).
6. **Ocorrências** (1 min): foto + registro salvos, envio ao Telegram, botões **Confirmar / Descartar** e a precisão revisada.
7. **Modelo** (1 min): rodapé da barra lateral mostra o mAP50 por classe; tabela completa em `docs/MODELO.md`.

## Perguntas prováveis da banca

| Pergunta | Resposta curta |
|---|---|
| Por que dois modelos? | O 1º (COCO) acha pessoas em qualquer ambiente; o 2º só roda se houver pessoas e é especializado em EPI. Economiza processamento e permite avaliar **por pessoa**. |
| Como sabe de quem é o capacete? | Pela posição da caixa: cabeça (topo), tronco (meio) e pés (base) da caixa da pessoa. Se duas pessoas disputam o mesmo EPI, fica com a mais próxima; se empatar, o sistema não conclui. |
| Por que "Não detectado" e não "Sem EPI"? | Não ver não prova ausência (oclusão, distância, luz). "Ausente" só com classe negativa explícita. Reduz falsos alarmes. |
| Qual a precisão? | mAP50 no teste (141 imagens inéditas): capacete 0,94, colete 0,89, bota 0,76. |
| Por que YOLO11 e não YOLOv5? | YOLO11 é a geração mais recente da Ultralytics: para o mesmo tamanho (nano), tem menos parâmetros e mAP igual ou maior que o YOLOv5 no COCO, o que importa para rodar em CPU de notebook. |
| Roda em tempo real? | ≈ 13 FPS em CPU de notebook (OpenVINO, 480 px); ≈ 18 FPS no modo rápido. Com GPU NVIDIA tende a ser bem mais rápido (não medido). |
| E se errar? | Toda ocorrência fica para revisão humana (Confirmar/Descartar); a precisão revisada mede o desempenho real no local. |
| Limitações? | Dataset público (sem fotos do local), sem classe "sem colete", recall baixo para "sem capacete/sem bota", e Metabase/N8N ficaram como trabalho futuro. |

## Correções para o artigo / texto do TCC

1. **Remover o mAP@0,5 = 0,841**: não foi medido neste projeto. Substituir por:

   > O detector de EPIs (YOLO11n, ajustado por 50 épocas no conjunto público Construction-PPE) obteve, no conjunto de teste com 141 imagens não utilizadas no treinamento, mAP@0,5 de 0,94 para capacete, 0,89 para colete e 0,76 para bota, com precisão de 0,89, 0,79 e 0,73 e recall de 0,90, 0,87 e 0,72, respectivamente (entrada de 480 px).

2. **Arquitetura**: onde o texto cita YOLOv5 como modelo usado, trocar por *YOLO11n (Ultralytics)*, justificando pela geração mais recente e melhor relação precisão/custo em CPU.
3. **Desempenho**: "≈ 13 quadros por segundo em CPU Intel Core i7-1355U, com OpenVINO e entrada de 480 px".
4. **Escopo**: capacete, colete **e bota**; o sistema avalia cada pessoa individualmente.
5. **Trabalhos futuros** (itens do roteiro não implementados):
   > A integração com Metabase e N8N e um frontend em Next.js ficam como trabalhos futuros; no protótipo, os alertas são enviados diretamente pela API do Telegram e a visualização foi implementada em Streamlit. Também se propõe a coleta e anotação de imagens no ambiente real e a inclusão da classe "sem colete".
