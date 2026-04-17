# ⚡ Guia Rápido de Início

**Sistema de Monitoramento de EPI - TCC**  
**Tempo estimado:** 5-10 minutos

---

## 🚀 Instalação Express

### 1. Pré-requisitos
```bash
# Verifique Python 3.9+
python --version

# Instale dependências
pip install -r requirements.txt
```

### 2. Configure o ambiente
```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite o .env conforme necessário
nano .env  # ou seu editor favorito
```

---

## 🎥 Configuração da Câmera (IMPORTANTE!)

### Opção A: Webcam/USB (Mais Simples)

```bash
# 1. Detecte sua câmera
python test_camera.py --detect

# Saída esperada:
# ✅ Câmera encontrada no índice 0
# ✅ Câmera encontrada no índice 1

# 2. Teste a câmera
python test_camera.py --index 0

# 3. Configure no .env
# CAMERA_INDEX=0
```

### Opção B: Câmera IP (Intelbras, Hikvision, etc)

```bash
# 1. Encontre a URL RTSP da sua câmera
# Exemplo Intelbras:
# rtsp://admin:senha@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0

# 2. Teste no VLC primeiro
# VLC > Mídia > Abrir Fluxo de Rede > Cole a URL

# 3. Teste no sistema
python test_camera.py --rtsp "rtsp://admin:senha@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"

# 4. Configure no .env
# CAMERA_INDEX=rtsp://admin:senha@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0
```

---

## ✅ Validação do Sistema

```bash
# Execute o validador completo
python validate_system.py

# Deve mostrar:
# ✅ 11/11 testes passaram
# 🎉 Sistema totalmente validado
```

---

## 🎬 Execução

### Terminal 1: Sistema Principal
```bash
python main.py
```

**Controles:**
- `Q` ou `ESC` → Encerrar
- `B` → Benchmark de performance
- `S` → Salvar frame manualmente

### Terminal 2: Dashboard (Opcional)
```bash
streamlit run src/dashboard/streamlit_app.py
```

Abra: http://localhost:8501

---

## 🔧 Problemas Comuns

### ❌ "Câmera não encontrada"

```bash
# Solução 1: Detectar automaticamente
python test_camera.py --detect

# Solução 2: Testar índices manualmente
python test_camera.py --index 0
python test_camera.py --index 1
python test_camera.py --index 2

# Solução 3: Fechar outros programas
# Zoom, Teams, Skype, etc
```

### ❌ "ModuleNotFoundError"

```bash
# Reinstale dependências
pip install -r requirements.txt --upgrade
```

### ❌ "Modelo não encontrado"

```bash
# O modelo será baixado automaticamente na primeira vez
# Aguarde o download (cerca de 6MB para yolov8n.pt)
```

### ❌ Câmera IP não conecta

```bash
# 1. Teste no VLC
# 2. Verifique IP, usuário e senha
# 3. Pinge o IP: ping 192.168.1.108
# 4. Verifique firewall (porta 554)
```

**Para mais soluções:** Veja [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 📊 Configurações Importantes (.env)

### Câmera
```env
# Webcam
CAMERA_INDEX=0

# Ou câmera IP
CAMERA_INDEX=rtsp://admin:senha@IP:554/stream
```

### Detecção
```env
# Mais rápido = menor confiança (mais detecções)
CONFIDENCE=0.35

# Mais preciso = maior confiança (menos detecções)
CONFIDENCE=0.60
```

### Performance
```env
# Melhor FPS (baixa qualidade)
FRAME_WIDTH=320
FRAME_HEIGHT=240

# Melhor qualidade (baixo FPS)
FRAME_WIDTH=1280
FRAME_HEIGHT=720
```

### Modelo
```env
# Mais rápido
MODEL_PATH=yolov8n.pt

# Mais preciso (mais lento)
MODEL_PATH=yolov8m.pt
```

---

## 📱 Configurar Telegram (Opcional)

### 1. Criar Bot
```
1. Abra o Telegram
2. Busque: @BotFather
3. Digite: /newbot
4. Escolha nome e username
5. Copie o TOKEN
```

### 2. Obter Chat ID
```
1. Inicie conversa com seu bot
2. Acesse: https://api.telegram.org/bot<SEU_TOKEN>/getUpdates
3. Envie qualquer mensagem para o bot
4. Recarregue a URL
5. Copie o "id" do campo "chat"
```

### 3. Configure .env
```env
TELEGRAM_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
TELEGRAM_CHAT_ID=987654321
```

---

## 🎯 Checklist de Funcionamento

- [ ] Python 3.9+ instalado
- [ ] Dependências instaladas (`pip install -r requirements.txt`)
- [ ] Câmera detectada (`python test_camera.py --detect`)
- [ ] `.env` configurado
- [ ] Validação passou (`python validate_system.py`)
- [ ] Sistema executa sem erros (`python main.py`)
- [ ] Dashboard abre (opcional)
- [ ] Telegram configurado (opcional)

---

## 📚 Próximos Passos

### Para Desenvolvimento:
1. Leia [CHANGELOG.md](CHANGELOG.md) - Todas as melhorias
2. Leia [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Resolução de problemas
3. Veja `src/` - Código fonte comentado

### Para o TCC:
1. Execute benchmarks: Pressione `B` durante execução
2. Teste cenários: 1 pessoa, 2 pessoas, iluminação baixa
3. Colete métricas: FPS, precisão, recall
4. Documente resultados

### Para Produção:
1. Treine modelo customizado em dataset de EPI
2. Configure alertas Telegram
3. Ajuste `CONFIDENCE` para seu ambiente
4. Configure IP fixo na câmera

---

## 🆘 Ajuda

**Problemas?**
1. Execute `python validate_system.py`
2. Leia `TROUBLESHOOTING.md`
3. Teste câmera: `python test_camera.py --detect`

**Dúvidas sobre câmeras IP?**
- [Manual Intelbras](https://backend.intelbras.com/sites/default/files/2023-08/manual-instalacao-linha-ip-06-23-web_1.pdf)
- [RTSP URLs Database](https://www.ispyconnect.com/sources.aspx)

---

**Sistema pronto!** 🎉  
Execute `python main.py` e comece a monitorar!
