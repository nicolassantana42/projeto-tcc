# 🖥️ INTERFACE GRÁFICA MODERNA - Sistema de Monitoramento de EPI

**Dashboard Profissional com Visualização em Tempo Real**

---

## 📸 PREVIEW

```
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║  🦺 Monitor EPI                    ┌──────────────────────────────┐ ║
║  Sistema Inteligente de Segurança │  Visualização da Câmera      │ ║
║                                    │  ● Câmera ativa              │ ║
║  ┌──────────────────┐              │                              │ ║
║  │   Controles      │              │   [VÍDEO AO VIVO COM        │ ║
║  │                  │              │    DETECÇÕES EM TEMPO REAL]  │ ║
║  │ ▶ Iniciar        │              │                              │ ║
║  │ ⏹ Parar          │              │   FPS: 28.5 | Pessoas: 2    │ ║
║  │ 📸 Capturar      │              │                              │ ║
║  └──────────────────┘              └──────────────────────────────┘ ║
║                                                                      ║
║  ┌──────────────────┐              ┌──────────────────────────────┐ ║
║  │  Estatísticas    │              │  Alertas Recentes            │ ║
║  │                  │              │                              │ ║
║  │  FPS: 28.5       │              │ [10:23] ⚠️ Violação: SEM     │ ║
║  │  Frames: 1420    │              │         CAPACETE             │ ║
║  │  Pessoas: 2      │              │ [10:22] ✅ Câmera iniciada   │ ║
║  │  Violações: 3    │              │                              │ ║
║  └──────────────────┘              └──────────────────────────────┘ ║
║                                                                      ║
║  ┌──────────────────┐                                              ║
║  │ EPIs Monitorados │                                              ║
║  │                  │                                              ║
║  │ ☑ ⛑️  Capacete   │                                              ║
║  │ ☑ 🦺 Colete      │                                              ║
║  │ ☐ 👢 Bota        │                                              ║
║  └──────────────────┘                                              ║
║                                                                      ║
║  ⚙️ Configurações                                                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 🚀 COMO USAR

### Instalação
```bash
# Certifique-se de ter todas as dependências
pip install -r requirements.txt
```

### Executar
```bash
python app_interface.py
```

**Pronto!** A interface gráfica abrirá automaticamente.

---

## 📋 FUNCIONALIDADES

### 1. **Dashboard Moderno**
- ✅ Design profissional com tema escuro
- ✅ Interface responsiva e intuitiva
- ✅ Estatísticas em tempo real
- ✅ Alertas visuais de violações

### 2. **Visualização de Câmera**
- ✅ Vídeo ao vivo com detecções
- ✅ Caixas delimitadoras coloridas:
  - 🟢 Verde = Pessoa conforme (todos EPIs OK)
  - 🔴 Vermelho = Violação detectada
- ✅ Lista de violações em tempo real
- ✅ FPS e contador de pessoas no canto

### 3. **Controles Intuitivos**
- **▶ Iniciar Câmera** - Inicia captura e detecção
- **⏹ Parar Câmera** - Para o sistema
- **📸 Capturar Foto** - Salva snapshot do frame atual

### 4. **Estatísticas em Tempo Real**
- **FPS** - Taxa de quadros processados por segundo
- **Frames** - Total de quadros processados
- **Pessoas** - Pessoas detectadas no frame atual
- **Violações** - Total de violações encontradas

### 5. **Configuração de EPIs**
Marque/desmarque quais EPIs monitorar:
- ⛑️ **Capacete** - Proteção para cabeça
- 🦺 **Colete** - Colete de segurança/visibilidade
- 👢 **Bota** - Botas de segurança (requer modelo treinado)

**As configurações são aplicadas imediatamente!**

### 6. **Alertas em Tempo Real**
Painel de log mostra:
- ⚠️ Violações detectadas
- ✅ Status do sistema
- ℹ️ Informações gerais
- ❌ Erros (se houver)

### 7. **Configurações Avançadas**
Clique em **⚙️ Configurações** para:
- Alterar câmera (índice ou URL RTSP)
- Ajustar confiança de detecção (0-1)
- Configurar parâmetros avançados

---

## 🎯 FLUXO DE USO

### Cenário 1: Webcam/USB
```
1. Abra a aplicação: python app_interface.py
2. Clique em "▶ Iniciar Câmera"
3. O sistema detecta automaticamente sua webcam
4. Configure os EPIs desejados (marque/desmarque)
5. Monitore em tempo real!
```

### Cenário 2: Câmera IP
```
1. Abra a aplicação
2. Clique em "⚙️ Configurações"
3. Em "Índice/URL da Câmera", cole:
   rtsp://admin:senha@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0
4. Clique em "Salvar"
5. Clique em "▶ Iniciar Câmera"
6. Configure os EPIs desejados
7. Monitore em tempo real!
```

---

## 🎨 INTERFACE DETALHADA

### Painel Lateral Esquerdo

#### 📊 Controles
- Botões grandes e claros
- Feedback visual (botões ficam desabilitados quando não aplicáveis)
- Cores intuitivas (verde=iniciar, vermelho=parar)

#### 📈 Estatísticas
- Atualização em tempo real
- Cores contextuais (laranja para violações)
- Fonte clara e legível

#### ☑️ EPIs Monitorados
- Checkboxes com ícones
- Mudanças aplicadas instantaneamente
- Indicação visual clara do que está ativo

### Painel Central

#### 🎥 Visualização de Câmera
- Vídeo redimensionado automaticamente
- Mantém proporção original
- Detecções desenhadas sobre o vídeo:
  - Caixas verdes/vermelhas ao redor de pessoas
  - Labels de status ("OK" ou "VIOLACAO")
  - Lista de violações abaixo de cada pessoa

### Painel Inferior

#### 📝 Alertas Recentes
- Log cronológico
- Ícones por tipo de mensagem
- Auto-scroll para últimos alertas
- Histórico limitado (últimas 100 mensagens)

---

## ⚙️ CONFIGURAÇÕES AVANÇADAS

### Janela de Configurações

#### Câmera
- **Índice/URL**: Configure qual câmera usar
  - Índice: 0, 1, 2... para webcams/USB
  - URL: rtsp://... para câmeras IP

#### Detecção
- **Confiança (0-1)**: Ajusta sensibilidade
  - Menor (0.3) = Mais detecções (mais falsos positivos)
  - Maior (0.7) = Menos detecções (mais precisas)
  - Padrão: 0.45

---

## 🎯 CASOS DE USO

### Para o TCC
1. **Demonstração para banca**
   - Interface profissional impressiona
   - Estatísticas em tempo real
   - Fácil de explicar e demonstrar

2. **Coleta de métricas**
   - FPS médio visível
   - Contagem de violações
   - Screenshots automáticos

3. **Testes de cenários**
   - Fácil ligar/desligar câmera
   - Trocar configurações rapidamente
   - Testar diferentes EPIs

### Para Produção
1. **Monitoramento de canteiro**
   - Interface clara para operadores
   - Alertas visuais imediatos
   - Log de eventos

2. **Treinamento de funcionários**
   - Visualização clara das detecções
   - Feedback imediato
   - Configurável por tipo de obra

---

## 🔧 TECLAS DE ATALHO

| Tecla | Ação |
|-------|------|
| `Ctrl+S` | Iniciar câmera |
| `Ctrl+Q` | Parar câmera |
| `Ctrl+P` | Capturar foto |
| `Ctrl+,` | Abrir configurações |

---

## 🐛 SOLUÇÃO DE PROBLEMAS

### Interface não abre
```bash
# Verifique se CustomTkinter está instalado
pip install customtkinter

# Execute com logs
python app_interface.py
```

### Câmera não inicia
1. Clique em "⚙️ Configurações"
2. Tente outro índice (0, 1, 2...)
3. Ou execute: `python test_camera.py --detect`

### Interface lenta
1. Reduza a resolução em Configurações
2. Use modelo mais leve (yolov8n.pt)
3. Feche outros programas

### Detecções não aparecem
1. Verifique se o modelo está carregado
2. Ajuste a confiança nas Configurações
3. Certifique-se de estar no modo correto (DEMO_MODE no .env)

---

## 📊 DIFERENÇAS: Interface vs. Terminal

### Interface Gráfica (app_interface.py)
- ✅ Visual e intuitivo
- ✅ Dashboard com estatísticas
- ✅ Configurações em tempo real
- ✅ Ideal para demonstrações
- ✅ Melhor para operadores não técnicos

### Terminal (main.py)
- ✅ Mais leve (menos recursos)
- ✅ Funciona em servidores (headless)
- ✅ Mais logs detalhados
- ✅ Ideal para desenvolvimento
- ✅ Melhor para automação

**Ambos funcionam igualmente bem!** Escolha baseado no seu uso.

---

## 🎓 PARA O TCC

### Adicione ao seu documento:

**Capítulo de Implementação:**
```
"Foi desenvolvida uma interface gráfica moderna utilizando CustomTkinter,
permitindo operação intuitiva do sistema por usuários não técnicos. A
interface apresenta visualização em tempo real das detecções, estatísticas
de performance, e configuração dinâmica dos EPIs monitorados."
```

**Capítulo de Resultados:**
```
"A interface gráfica demonstrou facilitar significativamente a operação
do sistema, com feedback visual imediato das detecções e configuração
sem necessidade de edição de arquivos de configuração."
```

---

## 🚀 MELHORIAS FUTURAS

Possíveis expansões:
- [ ] Gráficos de histórico (FPS ao longo do tempo)
- [ ] Mapa de calor de violações
- [ ] Múltiplas câmeras simultâneas
- [ ] Gravação de vídeo
- [ ] Exportação de relatórios PDF
- [ ] Integração com banco de dados
- [ ] API REST para integração externa
- [ ] Modo tela cheia para painéis grandes

---

## 📞 SUPORTE

**Problemas com a interface?**
1. Verifique se CustomTkinter está instalado
2. Execute o terminal primeiro: `python main.py`
3. Se funcionar no terminal mas não na interface, é problema de GUI
4. Leia TROUBLESHOOTING.md

---

## ✅ CHECKLIST DE FUNCIONAMENTO

- [ ] CustomTkinter instalado
- [ ] Aplicação abre sem erros
- [ ] Botão "Iniciar Câmera" funciona
- [ ] Vídeo aparece na tela central
- [ ] Detecções são desenhadas no vídeo
- [ ] Estatísticas atualizam em tempo real
- [ ] Checkboxes de EPI funcionam
- [ ] Alertas aparecem no painel inferior
- [ ] Botão "Parar Câmera" funciona

---

**🎉 Interface pronta para uso!**

Execute: `python app_interface.py` e tenha um sistema profissional de monitoramento! 🚀

---

**Desenvolvido para TCC**  
**Versão:** 2.0 com Interface Gráfica  
**Data:** Abril 2026
