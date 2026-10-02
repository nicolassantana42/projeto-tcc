# Design da interface

Implementação: `src/epi_monitor/ui/theme.py` (tokens, ícones, componentes) e `src/epi_monitor/ui/app.py` (layout).
Stack: Streamlit + CSS próprio. Não migramos para React: isso exigiria criar uma API e um segundo servidor, e o projeto deixaria de rodar com dois cliques.

## Layout

| Área | Conteúdo |
|---|---|
| Sidebar (colapsável) | Marca → Fonte → Sensibilidade → ▶ Iniciar / ■ Parar → Avançado (modelos, OpenVINO, IoU) → Sessão (câmera, gravação, Telegram) |
| Header | Breadcrumb (Monitor de EPIs / Painel) · título · subtítulo · chips Capacete/Colete/Bota (ativos conforme o modelo carregado) · botão **Como funciona** (modal) |
| Navegação | Abas segmentadas: Monitoramento · Ocorrências · Alertas e integrações · Modelo |
| Monitoramento | 4 KPI cards → Câmera (2,2 col) + Por pessoa (1 col) → Conformidade ao longo do tempo |
| Ocorrências | Busca + filtro por tipo → tabela paginada → foto + detalhes + downloads |
| Modelo | KPIs de mAP50 por classe com variação → gráfico Anterior × Atual → tabela com barras de progresso → timeline do pipeline |

## Cores

| Token | Hex | Uso |
|---|---|---|
| bg / surface / surface-2 | #F8F9FA / #FFFFFF / #F3F4F6 | fundo, cards, áreas neutras |
| border / border-strong | #E5E7EB / #D1D5DB | bordas, hover |
| text / text-2 / text-3 | #111827 / #4B5563 / #6B7280 | título, corpo, legenda |
| primary / 600 / 50 / 100 | #0F766E / #0D655E / #F0FDFA / #CCFBF1 | ações, marca, foco |
| accent | #F59E0B | âmbar de segurança (também “inconclusivo”) |
| success / warning / danger / info | #10B981 / #F59E0B / #EF4444 / #3B82F6 | estados; textos usam tons 700 sobre fundos 50 (contraste AA) |
| viewport | #0F172A | área de vídeo escura, destaca a imagem |

## Tipografia

Inter (Google Fonts; fallback system-ui) · Display 30/800 · H1 24/700 · H2 18/700 · Body 14/400–600 · Caption 12/500. Tracking negativo em títulos.

## Componentes

KPI card (ícone, valor, variação ↑↓→ em relação ao quadro anterior) · Card de pessoa (avatar P1, badge de estado, 3 células de EPI com ícone) · Badge (ok, unsafe, uncertain, info, neutral, *live* pulsante) · Chip · Empty state com passos · Skeleton · Timeline · Botões Primary/Secondary/Ghost (Destructive = Parar) · Tabela com busca, filtro e paginação · Modal (`st.dialog`) · Toasts · Gráficos Altair (área empilhada, barras agrupadas).

Ícones: Lucide, traço de 1,75 px, inline SVG (não depende de rede).

## Estados

| Estado | Tratamento |
|---|---|
| Hover | cards sobem 2 px + sombra média; botões −1 px; dropzone fica teal |
| Focus | anel de 3 px em `primary-100` |
| Active | botão volta a 0 px |
| Disabled | opacidade 45%, sem sombra |
| Loading | skeleton com o formato do painel enquanto o modelo carrega |
| Empty | ícone + título + texto útil + passos (Monitoramento, Ocorrências, Gráfico) |
| Error | `st.error` com mensagem humana e próxima ação; erro de busca vira empty state |
| Live | badge “Ao vivo” com pulso verde |

## Micro-interações

Entrada dos cards de pessoa (fade + 6 px, 350 ms) · pulso do badge ao vivo · shimmer do skeleton · transições de 180–200 ms com `cubic-bezier(.2,.8,.2,1)` · toast ao iniciar, parar, salvar imagem e salvar configurações.

## Justificativa para a banca

- **Fundo claro e cor sóbria (teal)** transmitem confiabilidade; o âmbar remete a sinalização de segurança, que é o domínio do trabalho.
- **Hierarquia**: o olho vai dos KPIs (resumo) para a câmera (evidência) e depois para cada pessoa (decisão). O gráfico vem por último, como contexto.
- **Verde, âmbar e vermelho, sempre com ícone e texto**: a cor nunca é o único sinal, o que ajuda quem é daltônico.
- **Aba Modelo** mostra métricas reais do conjunto de teste e a comparação com o modelo anterior, que são as perguntas típicas da banca.
- **Como funciona** explica o pipeline em cascata em 6 passos, sem sair da tela.
- Opções técnicas ficam escondidas em *Avançado*, o que mantém a tela limpa na demonstração.
