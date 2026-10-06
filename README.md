# Monitor de EPIs — capacete, colete e bota

Visão computacional para o TCC: **câmera/imagem → YOLO11 de pessoas → YOLO11 de EPIs → resultado por pessoa → ocorrência e alerta**.

## Como rodar

1. Instale o **Python 3.12** (uma vez): `winget install -e --id Python.Python.3.12`
2. Clone o repositório (os modelos já vêm junto) ou copie a pasta do projeto.
3. Dê **dois cliques em `iniciar.bat`** (ou rode `python run.py`).
   Na 1ª vez ele cria a `.venv` e instala as dependências (precisa de internet, ~5 min). Depois abre direto.
4. O navegador abre em **http://localhost:8501**. Escolha a fonte na barra lateral e clique **▶ Iniciar**.

| Para testar | Fonte | Caminho |
|---|---|---|
| Vídeo | Arquivo de vídeo → *Caminho no computador* | `data/demo/demo_obra.mp4` (já preenchido) |
| Imagem | Imagem → *Enviar arquivo* | qualquer foto com pessoas |
| Câmera | Webcam local | índice `0` |

Se algo falhar, rode `python run.py` num terminal e leia a mensagem de erro.

## O que aparece

- Cada pessoa com ✅ Detectado · ❌ Ausente · ⚠️ Não detectado para capacete, colete e bota.
- **Ocorrências**: foto + registro de quem apareceu sem EPI; o analista **confirma ou descarta**.
- **Configurações**: nome da câmera, local e alerta pelo **Telegram** (opcional, [guia](docs/ALERTS.md)).

Não ver um EPI não prova que ele está ausente: "Ausente" só aparece quando o modelo detecta explicitamente `sem capacete` ou `sem bota`.

## Modelo

YOLO11n treinado no dataset público Construction-PPE. Conjunto de teste (141 imagens não usadas no treino):

| | Capacete | Colete | Bota |
|---|---|---|---|
| mAP50 | **0,94** | **0,89** | **0,76** |

≈ 13 FPS em CPU de notebook (OpenVINO, 480 px). Em *Avançado → Velocidade → Rápida* ≈ 18 FPS.
Detalhes, treino e limitações: [docs/MODELO.md](docs/MODELO.md).

## Linha de comando (opcional)

```bash
.venv\Scripts\activate
python -m epi_monitor detect --source data/demo/demo_obra.mp4 --show
python -m epi_monitor detect --source 0 --show
python -m pytest -q
```

## Estrutura

```text
iniciar.bat, run.py      inicialização com um comando
src/epi_monitor/         código (detecção, captura, ocorrências, alertas, interface em ui/)
models/                  pesos YOLO11 usados pela aplicação
data/                    YAML do dataset e vídeo de demonstração
docs/                    modelo, alertas, aderência ao TCC e roteiro da apresentação
tests/                   testes automatizados
```

[Apresentação](docs/APRESENTACAO.md) · [Modelo](docs/MODELO.md) · [Alertas](docs/ALERTS.md) · [Aderência ao roteiro](docs/TCC_ALIGNMENT.md)
