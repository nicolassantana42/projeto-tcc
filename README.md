# Monitoramento de EPI com YOLO e regras espaciais

Pipeline do TCC: captura de vídeo → YOLO → associação pessoa/EPI → confirmação temporal → evidência e alerta.

## Executar

Use Python 3.10 ou superior (validado com Python 3.12). Na raiz do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-core.txt
Copy-Item .env.example .env
python main.py
```

Em Linux/macOS, ative com `source .venv/bin/activate` e copie com `cp .env.example .env`.
O primeiro uso baixa os pesos oficiais `yolov8n.pt`, se ainda não existirem.

```text
python main.py --source 0
python main.py --source video.mp4 --headless --max-frames 100
python main.py --source rtsp://endereco-da-camera/stream
```

`CAMERA_INDEX` no `.env` aceita índice USB, arquivo ou URL RTSP/HTTP. `SHOW_VIDEO=false` também desativa a janela. Sem janela, use Ctrl+C para encerrar. O fim de um arquivo encerra o processo; a captura principal também encerra se a fonte parar de entregar frames. A classe opcional `IPCameraStream` mantém captura em thread para o dashboard.

| Tecla | Ação |
| --- | --- |
| Q / ESC | Encerrar e liberar recursos |
| B | Benchmark de 50 inferências no mesmo frame |
| S | Salvar o frame exibido em PNG |
| D | Mostrar/ocultar a região da cabeça |

O HUD exibe modo, FPS do pipeline, frames e registros do dia. Registros são eventos após confirmação e cooldown; não representam quantidade de pessoas únicas.

## DEMO e REAL

`DEMO_MODE=true` é o padrão. O modelo COCO detecta pessoas e o motor considera ausentes todos os EPIs exigidos. Com as configurações padrão aparecem SEM CAPACETE / SEM COLETE, inicialmente em âmbar e depois em vermelho. Isso demonstra o fluxo do sistema; não mede a precisão da detecção de EPI.

Para usar um modelo PPE:

1. Defina `DEMO_MODE=false` e `MODEL_PATH` para os pesos treinados.
2. Configure `PERSON_ID`, `HELMET_ID`, `VEST_ID` e `BOOT_ID` conforme os nomes do próprio modelo. Os valores no `.env.example` são exemplos.
3. Defina `REQUIRE_HELMET`, `REQUIRE_VEST` e `REQUIRE_BOOT` conforme o cenário. Botas são opcionais por padrão.

O detector lê `model.names`, valida as classes obrigatórias e ignora classes opcionais incompatíveis. Nomes positivos como `helmet`, `Hardhat`, `Safety Vest` e `capacete` são reconhecidos; classes como `NO-Hardhat` não contam como equipamento presente. Modelos com nomes genéricos precisam de metadados corrigidos. COCO em modo REAL falha com uma mensagem de configuração, evitando interpretar bicicletas/carros como EPI.

O `.env` é carregado a partir da raiz do projeto, mesmo quando a execução começa em outra pasta. Variáveis do processo prevalecem. Caminhos relativos de modelo, imagens e logs são resolvidos a partir dessa raiz.

## Regras espaciais e confirmação

Para cada pessoa, o motor calcula:

- **Cabeça:** 30% superiores da bounding box (`HEAD_REGION_RATIO=0.30`). Associa capacete quando `IoU(cabeça, capacete) >= 0.15` (`HELMET_IOU_THRESHOLD`).
- **Colete:** o centro da caixa do colete deve estar dentro da caixa da pessoa.
- **Bota, quando exigida:** o centro deve estar nos 25% inferiores da pessoa (`FOOT_REGION_RATIO=0.25`). Uma detecção é suficiente; não há verificação separada dos dois pés.

Cada EPI é atribuído a no máximo uma pessoa. Candidatos são ordenados por IoU (capacete) ou proximidade normalizada do centro (colete/bota), com desempate geométrico. Essa associação gulosa é determinística, mas pode errar quando pessoas se sobrepõem.

Um rastreamento leve por IoU mantém um contador por pessoa. `FRAMES_TO_CONFIRM=10` exige a mesma combinação de ausências por 10 frames avaliados consecutivos. Conformidade, mudança da combinação ou ausência da pessoa em um frame reiniciam o contador. `TRACK_MAX_MISSED` limita a retenção de IDs; `TRACK_IOU_THRESHOLD` controla a associação entre frames. Os IDs são temporários e podem trocar em cruzamentos, movimento rápido ou oclusão.

`PersonStatus.is_compliant` informa o resultado espacial atual. `confirmed` informa se a ausência passou pelo filtro temporal. A renderização distingue verde (conforme), âmbar (pendente) e vermelho (confirmado). Apenas confirmados geram eventos.

## Evidências e Telegram

`ViolationLogger` registra PNG, `logs/violations.json` e log de texto. `SAVE_FRAMES=false` mantém o registro estruturado sem imagem. Nomes de imagem únicos evitam sobrescrita; falhas de gravação são tratadas explicitamente.

`ALERT_COOLDOWN_SECONDS=300` limita eventos por câmera. O Telegram fica desabilitado enquanto `TELEGRAM_TOKEN` e `TELEGRAM_CHAT_ID` estiverem vazios. Quando configurado, usa um único worker, fila limitada e timeout; a inferência não aguarda o envio. Enfileirar um alerta não garante sua entrega. Nenhum alerta é enviado pelos testes automatizados.

## Estrutura e interfaces

- `config.py`: ambiente, classes, captura e parâmetros espaciais.
- `src/ai/detector.py`: `Detection`, `FrameResult`, `EPIDetector`, warm-up e benchmark.
- `src/rules/ppe_rules.py`: associação, `PersonStatus`, debounce e desenho.
- `src/camera/capture.py`: `VideoCapture`, `IPCameraStream` e FPS.
- `src/alerts/`: persistência e Telegram assíncrono.
- `main.py`: execução principal e atalhos.
- `tests/`: testes automatizados sem câmera e sem mensagens reais.

`PPEDetector` e `check_violation()` permanecem como adaptadores para a interface antiga. Para instalar também as interfaces desktop/dashboard, use `python -m pip install -r requirements.txt`. O dashboard industrial utiliza seu banco SQLite próprio; o JSON do pipeline principal é utilizado pela interface `ppe_monitor_app.py`.

O servidor `main_industrial.py` também usa o motor espacial, com estado independente por câmera e avaliação apenas de frames novos. Seu adaptador mantém a persistência SQLite e o cooldown do fluxo industrial. O cadastro e o banco são inicializados pelo dashboard: execute `streamlit run src/dashboard/streamlit_app.py` e, em outro terminal, `python main_industrial.py`.

## Validação do TCC

```text
python -m pip install pytest
python -m pytest
```

A descoberta de testes se limita a `tests/`, sem executar os scripts de diagnóstico de câmera da raiz. Veja `VALIDATION.md` para os resultados obtidos nesta implementação.

O benchmark informa latência de inferência e FPS equivalente; não calcula mAP nem representa o FPS completo com captura, desenho e persistência. Valide separadamente com imagens PPE anotadas: precisão/recall por classe, mAP, taxa de falsos alertas por pessoa e desempenho por iluminação, distância e oclusão. Os limiares 0.30/0.15 são os da metodologia fornecida, não resultados de calibração. Pessoa agachada, cabeça parcialmente fora do frame e pés não visíveis podem gerar falsas ausências.

## Referências técnicas

- [Ultralytics: modo Predict e estrutura Results](https://docs.ultralytics.com/modes/predict/)
- [Ultralytics: modelos YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- [OpenCV: VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)
- [Telegram Bot API](https://core.telegram.org/bots/api)
