# Ensaio de estabilidade local

`scripts/soak_test.py` executa a cascata real sequencialmente em uma imagem ou
vídeo **local**, sem interface, webcam, RTSP ou envio de alertas. Registra um
JSON e amostras CSV de memória RSS, uso de CPU, vazão e latência. Repetir uma
cena verifica continuidade operacional; não mede acurácia e não produz um
novo teste independente do modelo.

Na raiz do repositório, com os exports já preparados:

```powershell
.\.venv\Scripts\python.exe scripts/soak_test.py --source "data/meu-video.avi" --person-model models/yolo11n_openvino_model --ppe-model models/ppe/absence_openvino_model --cpu-threads 4 --loop --max-seconds 180 --max-frames 5000 --output runs/soak/local-180s.json
```

O script usa CPU explicitamente, exige arquivos/pesos existentes e desativa
instalações automáticas do backend. Não baixe novos pesos para executá-lo. Para
usar os modelos nativos, informe os arquivos `.pt`. Um modelo de botas opcional
pode ser informado com `--boots-model` se a factory instalada já oferecer esse
suporte. Aumentar o número de modelos também altera carga, latência e memória.

Pare outras inferências antes de comparar resultados. Para repetir com o mesmo
conjunto, mantenha hashes, resolução, limiares, threads, fonte e duração. Os
hashes das fontes e modelos e as versões do ambiente ficam no JSON. Dados dos
frames são agregados; imagens, caixas e identidades não são persistidas por
esse script.

## Como interpretar

- `completed` significa que a execução chegou ao EOF ou a um dos limites sem
  erro. Não significa que um critério científico ou operacional foi aprovado.
- `--loop` reabre a mesma fonte ao chegar ao EOF. Sem essa opção, um vídeo curto
  termina cedo, ainda que `--max-seconds` seja maior.
- Os limites de tempo são verificados **entre quadros**. O ensaio não interrompe
  uma chamada nativa bloqueada; não é um watchdog de produção.
- Carregamento e aquecimento são cronometrados separadamente. `fps` inclui
  leitura local, inferência e escrita das amostras; não é FPS da interface.
- RSS é memória residente do processo, amostrada a cada 25 quadros por padrão
  e ao final. O pico amostrado pode não capturar picos curtos. A referência para
  crescimento é tomada **após** o aquecimento. Essa métrica não cobre VRAM.
- Média de latência usa todos os quadros; mediana/p95/máximo usam os últimos
  10 mil. O histórico em memória é limitado e o CSV é escrito incrementalmente.
- Observações repetidas não são pessoas únicas, e `uncertain` não equivale a
  ausência de EPI. O programa não mede acertos ou falsos alertas.
- Falhas retornam código 2, preservam relatório parcial quando a execução já
  começou e fecham a fonte. Credenciais de erros internos não são copiadas.

Critérios de aceite devem ser definidos antes do ensaio, para o hardware e
cenário propostos. Exemplo de **limites ilustrativos, não homologados**:

```powershell
# Acrescente ao comando anterior apenas após definir os limites do projeto:
--min-fps 5 --max-rss-growth-mib 200
```

Com esses argumentos, `requested_checks_passed` só é verdadeiro se a execução
terminar sem erro e os dois limites forem atendidos. Mesmo assim, estabilidade
de longo prazo requer uma sessão mais extensa, diferentes cenas e análise das
amostras de RSS. Crescimento inicial pode ser cache legítimo; estabilidade em
três minutos não prova ausência de vazamento em horas.

## O que esse ensaio não substitui

| Camada | Evidência necessária |
| --- | --- |
| Software | Testes unitários e integração; erros de entrada; código de saída. |
| Modelo | Ground truth, splits independentes, métricas por classe e por pessoa. |
| Interface | Sessão longa com captura/renderização e coleta de memória do Streamlit. |
| Campo | Vídeo da câmera real, iluminação/distância/oclusão, alertas ao longo do tempo. |
| Integrações | Entrega real de imagem/local para o bot configurado e recuperação de falhas. |
| Portabilidade | Execução real de container, CUDA e MPS em seus ambientes. |

A configuração de CI inclui Windows, Linux, macOS e build/healthcheck Docker.
Isso não é evidência de que todas essas execuções passaram. Na inspeção local
de 01/10/2026, Docker não estava disponível no PATH nem no caminho padrão do
Docker Desktop; o container não foi executado nessa máquina. Sem esse runtime,
um YAML/Dockerfile revisado continua **não validado em execução local**.
