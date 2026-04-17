# 🔧 Guia de Troubleshooting - Problemas com Câmeras

Este guia ajuda a resolver os problemas mais comuns com câmeras no Sistema de Monitoramento de EPI.

---

## 📋 Índice Rápido

- [Webcam/USB não detectada](#webcam-não-detectada)
- [Câmera IP não conecta](#câmera-ip-não-conecta)
- [Preview congelado ou lento](#preview-congelado-ou-lento)
- [Erro "Failed to load image from stream"](#erro-de-stream)
- [Câmera funciona mas não detecta pessoas](#não-detecta-pessoas)

---

## 🎥 Webcam/USB Não Detectada

### Sintoma
```
❌ Não foi possível abrir a câmera 0
```

### Soluções

#### 1. **Detectar câmeras automaticamente**
```bash
python test_camera.py --detect
```

Isso vai escanear índices 0-10 e mostrar todas as câmeras disponíveis.

#### 2. **Testar índices manualmente**
A webcam pode não estar no índice 0. Tente:

```bash
# No arquivo .env, teste diferentes valores:
CAMERA_INDEX=0  # primeira tentativa
CAMERA_INDEX=1  # se 0 não funcionar
CAMERA_INDEX=2  # se 1 não funcionar
```

#### 3. **Verificar permissões (Linux)**
```bash
# Adicione seu usuário ao grupo video
sudo usermod -a -G video $USER
# Faça logout e login novamente

# Ou execute com sudo temporariamente
sudo python main.py
```

#### 4. **Verificar privacidade (Windows)**
1. Configurações > Privacidade > Câmera
2. Permitir que aplicativos acessem a câmera: **ON**
3. Permitir aplicativos de área de trabalho: **ON**

#### 5. **Fechar outros programas**
Outros programas podem estar usando a câmera:
- Zoom, Teams, Skype, Discord
- OBS Studio
- Aplicativos de videoconferência
- Chrome/Firefox com permissão de câmera

**Solução:** Feche todos esses programas e tente novamente.

#### 6. **Reinstalar drivers (Windows)**
1. Gerenciador de Dispositivos
2. Câmeras > Sua Webcam
3. Botão direito > Desinstalar dispositivo
4. Reinicie o computador (drivers serão reinstalados)

---

## 📡 Câmera IP Não Conecta

### Sintoma
```
❌ Não foi possível abrir a câmera rtsp://...
```

### Checklist Inicial

#### 1. **Testar a URL no VLC primeiro**
```
VLC Media Player > Mídia > Abrir Fluxo de Rede
Cole a URL RTSP
```

Se não funcionar no VLC, a URL está incorreta ou a câmera tem problemas.

#### 2. **Formato correto da URL**

**Intelbras (maioria dos modelos):**
```
rtsp://admin:senha@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0
```

**Hikvision:**
```
rtsp://admin:senha@192.168.1.64:554/Streaming/Channels/101
```

**Dahua:**
```
rtsp://admin:senha@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0
```

**MJPEG/HTTP:**
```
http://192.168.1.108/video.mjpg
http://usuario:senha@192.168.1.108/video.mjpg
```

#### 3. **Verificar conectividade de rede**
```bash
# Pingar o IP da câmera
ping 192.168.1.108

# Se não pingar:
# - Câmera está desligada?
# - Está na mesma rede?
# - IP está correto?
```

#### 4. **Verificar firewall**
```bash
# Windows: Desabilite temporariamente o firewall para testar
# Linux: Libere a porta 554
sudo ufw allow 554/tcp
```

#### 5. **Testar sem autenticação primeiro**
Alguns modelos permitem acesso sem senha:
```
rtsp://192.168.1.108:554/cam/realmonitor?channel=1&subtype=0
```

#### 6. **Usar IP fixo**
Câmeras com DHCP podem mudar de IP. Configure IP fixo:
1. Acesse interface web da câmera
2. Rede > TCP/IP
3. Configure IP estático (ex: 192.168.1.108)

#### 7. **Verificar codec de vídeo**
Algumas câmeras usam H.265 (HEVC) que pode não ser suportado.
Mude para H.264 na configuração da câmera:
1. Interface web > Vídeo > Encoding
2. Codec: **H.264**

---

## 🐌 Preview Congelado ou Lento

### Sintoma
```
FPS: 3.2  (muito baixo)
Frame congela frequentemente
```

### Soluções

#### 1. **Reduzir resolução**
```bash
# No .env:
FRAME_WIDTH=320   # ao invés de 1920
FRAME_HEIGHT=240  # ao invés de 1080
```

#### 2. **Usar modelo mais leve**
```bash
# No .env:
MODEL_PATH=yolov8n.pt  # mais rápido
# ao invés de yolov8m.pt ou yolov8l.pt
```

#### 3. **Desabilitar recursos pesados**
```bash
# No .env:
SAVE_FRAMES=false  # não salva todas as violações
SHOW_VIDEO=false   # não mostra preview (headless)
```

#### 4. **Câmeras IP: reduzir qualidade do stream**
Configure substream (baixa qualidade) ao invés de mainstream:
```
# Mainstream (alta qualidade):
rtsp://...&subtype=0

# Substream (baixa qualidade, mais rápido):
rtsp://...&subtype=1
```

#### 5. **Limpar buffer da câmera**
O buffer já está otimizado na nova versão (`buffer_size=1`), mas você pode testar:
```python
# Em src/camera/capture.py, linha ~250:
self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # já está configurado
```

---

## ⚠️ Erro de Stream

### Sintoma
```
⚠️ Frame inválido recebido
Câmera parou de fornecer frames
```

### Soluções

#### 1. **Reconexão automática já está implementada**
A nova versão tenta reconectar automaticamente. Se falhar muito:

```bash
# Aumente tentativas em src/camera/capture.py:
self._max_reconnection_attempts = 5  # padrão: 3
```

#### 2. **Câmeras IP: Verificar timeout**
```python
# No código, adicione timeout:
cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 segundos
cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)  # 10 segundos
```

#### 3. **Reiniciar câmera fisicamente**
- Desligue e ligue a câmera
- Para câmeras IP: reinicie via interface web

---

## 🤖 Não Detecta Pessoas

### Sintoma
```
Câmera funciona, mas "0 pessoas detectadas"
```

### Soluções

#### 1. **Verificar modo DEMO**
```bash
# No .env:
DEMO_MODE=true   # Detecta apenas pessoas (esperado)
```

#### 2. **Ajustar confiança**
```bash
# No .env:
CONFIDENCE=0.25  # mais permissivo (padrão: 0.45)
```

#### 3. **Verificar iluminação**
- Muita escuridão: aumente iluminação
- Muita luz: reduza brilho/contraste

#### 4. **Distância da câmera**
- Muito longe: pessoas aparecem muito pequenas
- Muito perto: pessoas cortadas

**Distância ideal:** 2-5 metros

---

## 🆘 Suporte Adicional

### Executar diagnóstico completo
```bash
python test_camera.py --detect
python test_camera.py --index 0 --duration 30
```

### Verificar logs
```bash
# Logs são salvos em:
logs/violations.json
logs/violations.log
```

### Informações do sistema
```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import platform; print('OS:', platform.system(), platform.release())"
```

### Reportar problema
Ao reportar um problema, inclua:
1. Resultado de `python test_camera.py --detect`
2. Versão do OpenCV
3. Sistema operacional
4. Modelo da câmera (se IP)
5. Logs de erro completos

---

## 📚 Links Úteis

- [Documentação OpenCV VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)
- [RTSP URLs para câmeras IP](https://www.ispyconnect.com/sources.aspx)
- [Manual Intelbras](https://backend.intelbras.com/sites/default/files/2023-08/manual-instalacao-linha-ip-06-23-web_1.pdf)
- [Teste de câmera online](https://webcamtests.com/)

---

**Última atualização:** Abril 2026  
**Versão do sistema:** 2.0 (com suporte robusto a câmeras)
