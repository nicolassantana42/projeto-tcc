# Plano executado e roteiro de adoção

## 1. Diagnóstico

Base auditada: commit `c76cbe4` do repositório `nicolassantana42/projeto-tcc`.
A cópia de trabalho estava limpa. O código já usava Ultralytics/YOLOv8; não
havia YOLOv5 em produção nem pesos/dataset para comparar mAP.

Problemas encontrados:

- `main.py` importava `EPIDetector`, `VideoCapture`, `draw_ppe_status` e
  `src.alerts.telegram`, ausentes na árvore, além de constantes inexistentes.
- README orientava instalar `requirements-core.txt` e consultar `VALIDATION.md`,
  ambos ausentes; a documentação prometia rastreamento que o código não tinha.
- Múltiplos pontos de entrada Tk/Streamlit/industrial e configurações divergentes.
- IDs fixos 0/1/2 com COCO podiam interpretar objetos gerais como EPIs.
- Dependências sem limites superiores, `plotly` usado mas não declarado,
  captura em thread sem sincronização e sem fim finito para vídeo.
- Nenhum teste unitário descobrível, Docker ou configuração de avaliação/exportação.

## 2. Substituição estrutural

1. Criar pacote instalável `src/safeguard` e contratos `Detection`/`FrameResult`.
2. Isolar `capture`, `preprocessing`, `inference`, `rules`, `pipeline`, `rendering`.
3. Substituir GUIs duplicadas por `ui/app.py` e exportação por `reporting.py`.
4. Criar CLI única em `cli.py` e workflows científicos em `ml.py`.
5. Fixar dependências diretas e separar extras de exportação.
6. Adicionar bootstrap, Docker, CI, testes e documentação verificável.

Os scripts legados `app_interface.py`, `ppe_monitor_app.py`, `main_industrial.py`,
`setup_and_test.py`, `validate_system.py`, `test_camera.py`, `RODAR_SISTEMA.vbs`,
`config.py`, `cameras.json`, `.env.example` e os módulos antigos `src/ai`,
`src/camera`, `src/dashboard`, `src/alerts`, `src/rules` foram removidos.
`main.py` virou adaptador para a nova entrada. A documentação antiga foi
substituída por README e `docs/`; o histórico permanece no Git.

Email, Telegram, cadastro multicâmera, banco SQLite e empacotamento EXE eram
fluxos separados do escopo deste dashboard e não fazem parte da nova aplicação.
Para recuperar código histórico sem sobrescrever a refatoração:

```powershell
git show c76cbe4:src/alerts/telegram_sender.py
git diff --stat
git status --short
```

Nenhum peso duplicado, cache ou notebook estava versionado no snapshot auditado.
A limpeza não inventa remoções: novos caches, pesos, vídeos, relatórios e
datasets ficam ignorados. Prints de documentação podem ser versionados.

## 3. Preparar apresentação

```powershell
python run.py
```

Selecione **Prévia ilustrativa** para ensaio da interface. Para inferência real,
selecione webcam/vídeo/RTSP e pesos existentes. COCO não avalia EPI. Em modo
**EPI treinado**, os pesos devem conter as classes positivas pessoa, capacete
e colete com nomes reconhecidos, descritos em `ARCHITECTURE.md`.

## 4. Treinar e decidir a arquitetura por evidências

Execute os comandos de `ML.md` com o mesmo dataset/split para YOLOv8n e YOLO11n.
Escolha pelo compromisso mAP50–95, recall de cada EPI e latência p95 no hardware
da apresentação. Migração de pesos YOLOv5 originais requer novo treinamento;
renomear o arquivo não converte a arquitetura.

## 5. Otimizar, medir e validar

Exporte ONNX FP32 como caminho portátil; OpenVINO INT8 para CPU e TensorRT
FP16/INT8 em NVIDIA. INT8 exige calibração representativa. Reavalie os
artefatos sobre o mesmo split e compare quedas de precisão antes de adotá-los.

```powershell
python -m pytest -q
python -m pip check
docker compose config
docker compose up --build
```

Consulte `VALIDATION.md` para distinguir evidências obtidas de verificações que
ainda exigem ambiente/dados externos. Não há upload ou push automático ao GitHub.
