# Validação executada

Execução local em **18/09/2026**, Windows 11, Intel Core i7-1355U, CPU,
Python 3.12.14. PyTorch 2.8.0 CPU, torchvision 0.23.0, Ultralytics 8.3.203,
OpenCV 4.12.0.88, NumPy 2.2.6, Streamlit 1.49.1, ONNX 1.17.0,
ONNX Runtime 1.22.1, OpenVINO 2025.4.1 e NNCF 2.19.0.

## Resultados

| Verificação | Evidência |
| --- | --- |
| `python -m pytest -q -p no:cacheprovider` | **82 passed** |
| `python -m pip check` | Nenhuma dependência incompatível |
| Compilação `compileall` | Código Python compilado sem erro |
| `run.py --install-only` e segunda execução | Instalação editável e reutilização do ambiente aprovadas |
| Streamlit AppTest | Iniciar, alterar confiança/IoU, preparar PNG/ZIP, parar e liberar recursos; upload ausente tratado |
| Navegador | Prévia ilustrativa em funcionamento; print em `images/dashboard.png` |
| YOLO11n PyTorch CPU | 8 frames de vídeo processados; pessoa/ônibus detectados; JSONL e snapshot gravados |
| Exportação ONNX FP32 e inferência CPU | Artefato exportado; 8 frames processados com as mesmas contagens do PyTorch na fixture |
| Resolução estática ONNX | Exportação 320, solicitação 640, 3 inferências consecutivas e validação adotaram 320 corretamente |
| Validação real da API | PNGs de matriz de confusão e curvas PR/F1/P/R, predições JSON, matriz/curvas numéricas em JSON |
| OpenVINO INT8 | Exportação com calibração de fixture e duas inferências CPU consecutivas aprovadas |
| Tratamento de falhas | Testes de modelo/dependência ausente, fallback, frame inválido, desconexão, EOF, corrupção e classes EPI incompatíveis |

Os testes automatizados usam mocks para GPU/câmera e não baixam pesos nem
enviam mensagens externas. Os testes AppTest exercitam Streamlit de verdade.
As execuções de integração usam pesos COCO oficiais baixados explicitamente.

## Alcance das integrações

`scripts/smoke.py` cria um vídeo MJPG com a imagem `bus.jpg` incluída na
Ultralytics, executa PyTorch/ONNX, snapshots e benchmark. É uma verificação de
integração, não um dataset de detecção de EPI. Na execução de 20 frames após 3
warmups, o tempo captura + pipeline + desenho ficou em aproximadamente:

| Backend CPU | Média | p95 | FPS equivalente |
| --- | --- | --- | --- |
| PyTorch | 74,6 ms | 109,3 ms | 13,4 |
| ONNX Runtime | 51,8 ms | 56,0 ms | 19,3 |

Uma cena repetida e uma amostra curta não sustentam promessa de desempenho.
Os números excluem UI/disco e variam com energia, carga, resolução e conteúdo.
Relatórios originais estão em `runs/smoke/benchmark-*.json`, ignorados pelo Git.
O script permite reproduzir as medições no computador de destino.

`scripts/smoke_validation.py --int8` usa seis cópias da imagem e um rótulo
aproximado, exclusivamente para conferir o contrato das APIs. O exportador
avisou que a calibração tem menos de 300 imagens recomendadas. Os gráficos e
valores mAP dessa fixture não devem ser apresentados como resultados do TCC.
O artefato fica isolado em `runs/smoke-validation/smoke_only_int8_openvino_model`
e não é modelo de EPI pronto para uso. A integração terminou com sucesso,
mesmo com o aviso NNCF de preferência por PyTorch 2.9; a versão usada foi 2.8.

## Ainda depende do ambiente real

- mAP, precision/recall por EPI e comparação YOLOv8/YOLO11 em dataset anotado.
- Treinamento completo, qualidade de calibração INT8 e sua perda de precisão.
- GPU NVIDIA, TensorRT, CUDA e Apple MPS: seleção/falhas cobertas por mocks,
  mas não executadas em hardware real nesta máquina.
- Webcam física, RTSP e reconexão com os drivers/câmeras da demonstração.
- Build/execução Docker e CI nos três sistemas: arquivos preparados; Docker
  não está instalado neste ambiente e os jobs remotos não foram executados.
- Setup limpo em uma máquina sem ambiente existente: o inicializador foi
  exercitado no ambiente local, com instalação editável e reutilização.

O Python global desta máquina aponta para o atalho da Microsoft Store. Para
executar imediatamente a cópia já preparada, na raiz do repositório use:

```powershell
.\.venv\Scripts\python.exe run.py
```

Durante testes, o sandbox Windows restringiu ACLs de diretórios temporários do
pytest/Streamlit. As execuções finais ocorreram com as permissões necessárias;
nenhuma alteração de segurança do sistema foi exigida pelo produto.
