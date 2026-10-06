# Modelos

| Arquivo | Uso |
|---|---|
| `yolo11n.pt` | Detector de pessoas (COCO oficial), 1º estágio — perfil PyTorch |
| `yolo11n_480_openvino_model/` | Mesmo detector exportado para OpenVINO 480 px — perfil padrão em CPU |
| `ppe/epi.pt` | Detector de EPIs (YOLO11n treinado no Construction-PPE): capacete, colete, bota |
| `ppe/epi_480_openvino_model/` | Mesmo detector exportado para OpenVINO 480 px — perfil padrão em CPU |
| `ppe/epi.metrics.json` | Métricas no conjunto de teste, exibidas na interface |

Estes arquivos são versionados no Git para o projeto rodar logo após o clone. Treino, métricas e exportação: [docs/MODELO.md](../docs/MODELO.md).
