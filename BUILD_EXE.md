# Como Gerar o Executável .exe (Windows)

## Pré-requisitos
```bash
pip install pyinstaller
```

## Comando único (recomendado)
```bash
pyinstaller ^
  --onefile ^
  --windowed ^
  --name "SafeGuard_EPI_Monitor" ^
  --add-data "yolov8n.pt;." ^
  --add-data "src;src" ^
  --add-data ".env;." ^
  --icon=icon.ico ^
  ppe_monitor_app.py
```

## Resultado
O arquivo `dist/SafeGuard_EPI_Monitor.exe` é o executável final.
Inclui todas as dependências — basta copiar para qualquer PC com Windows.

## Resolução de problemas comuns

### Erro "No module named ultralytics"
```bash
pyinstaller --hidden-import=ultralytics ppe_monitor_app.py
```

### Antivírus bloqueia o .exe
Isso é normal para executáveis gerados com PyInstaller.
Adicione exceção no Windows Defender ou use `--onedir` em vez de `--onefile`.

### Tamanho grande do .exe
Normal: PyTorch + YOLOv8 + OpenCV = ~400-600MB.
Use `--onedir` para manter as DLLs separadas (carrega mais rápido).
