# Monitoramento, ocorrências e alertas

O fluxo da demonstração é **ver a detecção → registrar uma ocorrência → enviar
uma evidência para um canal ativado**. Telegram é o primeiro canal. E-mail é
opcional; a integração Microsoft aceita um token OAuth2 fornecido pelo operador,
mas ainda não oferece login Microsoft nem renovação automática desse token.

A detecção funciona pela CLI sem interface ou Telegram. Esta página descreve
os recursos opcionais da interface. As abas de histórico e integrações também
ficam disponíveis na visualização simples. O caminho atual envia diretamente
à API do Telegram; não implementa a cadeia Metabase → N8N do roteiro.

## Onde aparecem as pessoas detectadas

1. Em **Alertas e integrações**, informe **Nome da câmera** e **Local / setor**
   reconhecíveis e clique **Salvar configurações**.
2. Abra **Monitoramento** e selecione uma fonte. Para capacetes e coletes, use
   o modo EPI com dois modelos: o detector de pessoas e pesos EPI compatíveis.
   Os pesos de trabalho ficam em `models/ppe/best.pt`; consulte o treinamento
   inicial e seus limites em [VALIDATION.md](VALIDATION.md). **Demo COCO** com `models/yolo11n.pt` demonstra
   apenas pessoas/objetos gerais; esses pesos não reconhecem EPIs.
3. Clique **Iniciar**. O quadro da câmera mostra as caixas, classes e confiança
   das detecções e os estados por pessoa. Os indicadores mostram as contagens
   do quadro atual. Nenhuma dessas contagens representa pessoas únicas.

A webcam é a do computador que executa o servidor Streamlit. Abrir a página em
outro dispositivo não transfere sua câmera para o servidor. No Docker, prefira
vídeo ou RTSP, ou configure o acesso ao dispositivo conforme o README.

A **prévia ilustrativa** demonstra o desenho da interface e não executa o modelo.
Ela não gera ocorrências automáticas nem alertas externos.

## Quando uma detecção vira ocorrência

Configure **Quando registrar uma ocorrência** na aba **Alertas e integrações**.
As duas regras têm significados diferentes. Registrar presença com COCO não
significa registrar uma infração de EPI:

| Regra | Uso | Modelo necessário |
| --- | --- | --- |
| Pessoa detectada (também funciona com COCO) | Registrar presença, sem julgar uso de EPI | COCO ou pesos com classe de pessoa |
| Possível ausência de EPI (modelo treinado) | Registrar estado `unsafe` por pessoa | Cascata com classe negativa explícita de EPI |

Não basta aparecer uma caixa em um quadro. Por padrão, a pessoa deve satisfazer
a regra durante **2 segundos da fonte de vídeo**, com associação temporal das caixas entre
quadros. Depois de registrar uma ocorrência, há um intervalo de **60 segundos
por câmera na sessão atual** para limitar novas ocorrências. Esses valores
podem ser ajustados. Uma segunda pessoa na mesma câmera também está sujeita
a esse intervalo. Parar e iniciar o monitoramento reinicia esse controle;
sessões paralelas não compartilham o intervalo.

A associação temporal não identifica quem é a pessoa e não fornece contagem
permanente de pessoas únicas. O estado `unsafe` exige uma classe negativa
explícita, como `no_helmet`, associada sem ambiguidade. Apenas não encontrar
capacete/colete mantém a avaliação `uncertain`, sem alerta de ausência.
`ok` significa EPIs detectados, não certificação de segurança. Mesmo uma
classe negativa pode estar errada: use a foto para revisão humana.

Em arquivos, a confirmação usa timestamps da mídia, com fallback de
índice/FPS; em webcam e RTSP, usa tempo monotônico entre observações. A
velocidade da análise não altera a duração exigida na linha do tempo do vídeo.
Imagens estáticas são observações únicas, identificadas como tal; não simulam
persistência temporal. A CLI `detect --save-events` também pode salvá-las e
nunca ativa os canais da interface.

O dataset Construction-PPE não contém `no_vest`. Pesos treinados somente
nessa taxonomia podem detectar coletes, mas não fornecem uma classe negativa
de ausência de colete. Nesse caso, colete não detectado permanece inconclusivo.

## Onde ficam as imagens

Com **Salvar ocorrências automaticamente** ativado (padrão), cada ocorrência
fica no computador do servidor, em uma subpasta própria:

```text
reports/
  settings.json                 # Preferências; não contém credenciais
  occurrences/
    <identificador-da-ocorrência>/
      snapshot.jpg              # Quadro anotado: pessoas e contexto da câmera
      event.json                # Horário, local, motivo e estado dos envios
```

Nesta instalação, a pasta base é
`C:\projeto-tcc\projeto-tcc\reports\occurrences` quando o app é iniciado na
raiz do projeto. A aba **Ocorrências** permite consultar e baixar a imagem e seu
JSON. Clique **Atualizar histórico** para consultar os registros mais recentes.
As imagens são quadros completos anotados, não um cadastro de rostos ou uma
gravação contínua do vídeo. **Salvar imagem agora**, em **Monitoramento**,
registra manualmente o quadro atual no histórico local, sem enviá-lo aos canais.
A exportação manual de snapshot/relatório continua disponível; o navegador
decide a pasta desses downloads.

O horário nos arquivos e nas mensagens é o **momento da análise em UTC**, não a
data/hora da gravação original. A listagem da interface converte para o fuso do
servidor. Sem configurar o local antes de iniciar, a ocorrência
registra “Local não informado”. Use `SAFEGUARD_REPORTS_DIR` para escolher outra
pasta base de relatórios antes de iniciar o aplicativo.

A retenção padrão busca manter as **500 ocorrências mais recentes**. Ao
ultrapassar o limite, ocorrências antigas e suas imagens podem ser removidas;
registros com envio pendente são preservados e podem fazer o total exceder esse
limite. Uma falha na limpeza gera aviso e mantém válida a imagem que já foi
salva, sem reiniciar o intervalo entre ocorrências. Exporte as evidências que
deseja conservar. A pasta `reports/` é ignorada pelo Git; no Docker, o volume
de relatórios mantém os arquivos fora do ciclo de vida do container.

## Conectar o Telegram

1. No Telegram, abra o [BotFather oficial](https://t.me/BotFather), envie
   `/newbot` e siga as instruções. Guarde o token como uma senha.
2. Abra a conversa com seu novo bot e envie `/start`. Para um grupo, adicione
   o bot e envie um comando dirigido a ele no grupo, por exemplo
   `/start@nome_do_seu_bot`.
3. Obtenha o identificador do chat conforme o procedimento abaixo.
4. Em **Alertas e integrações**, preencha **Token do bot** e **Chat ID de destino**. Informe
   também câmera e local para que a evidência seja útil para quem a receber.
5. Marque **Ativar Telegram para novas ocorrências** e clique **Salvar
   configurações** antes de iniciar o monitoramento. O canal começa desativado;
   somente preencher credenciais não inicia o envio.

Com o monitoramento parado, **Enviar teste aos canais ativos** envia uma imagem
de teste desenhada, identificada como teste, com os dados de câmera e local.
Esse botão realiza um envio real aos destinos configurados. Consulte o resultado
em **Ocorrências**. Os canais ativados valem para a sessão atual; ative-os novamente
ao iniciar outra sessão. Pare o monitoramento antes de alterar destinos e regras.

O usuário precisa iniciar a conversa com o bot antes de receber mensagens
privadas. Os passos de criação e contato seguem o
[tutorial oficial do Telegram](https://core.telegram.org/bots/tutorial).

### Descobrir o Chat ID localmente

Depois de enviar `/start`, execute no PowerShell da raiz do projeto. O token é
solicitado sem eco e não fica no texto do comando nem no histórico do navegador.
O comando consulta atualizações do bot e imprime somente IDs e tipos de chats;
ele não envia uma mensagem.

```powershell
.\.venv\Scripts\python.exe -c '
import getpass, json, urllib.request
token = getpass.getpass("Token do bot (oculto): ").strip()
try:
    url = "https://api.telegram.org/bot" + token + "/getUpdates"
    with urllib.request.urlopen(url, timeout=20) as response:
        result = json.load(response)
    chats = {}
    for update in result.get("result", []):
        message = update.get("message", update.get("channel_post", {}))
        chat = message.get("chat", {})
        if "id" in chat:
            chats[chat["id"]] = chat.get("type", "")
    for chat_id, chat_type in chats.items():
        print("Chat ID:", chat_id, "| Tipo:", chat_type)
    if not chats:
        print("Sem chats: envie /start ao bot e tente novamente.")
except Exception:
    print("Consulta falhou. Confira token, rede e webhook do bot.")
'
```

No Linux/macOS, use `.venv/bin/python` no lugar do executável Windows. Copie o
ID do chat desejado preservando o sinal negativo, se houver. Se outro programa
consumiu as atualizações, envie um novo comando ao bot. Um webhook já configurado
impede o uso de `getUpdates`; use um bot dedicado para esta demonstração.
Referência: [Telegram — getUpdates](https://core.telegram.org/bots/api#getupdates).

### O que chega ao destino

O envio usa uma foto anotada com legenda contendo câmera, local, horário e
motivo. Exemplo de conteúdo de uma ocorrência:

```text
SafeGuard — Possível ausência de EPI
Câmera: Entrada da obra
Local: Bloco B — acesso ao canteiro
Data/hora UTC: 2026-09-18T17:35:00+00:00
Motivo: classe explícita de ausência de capacete associada à pessoa
```

O local é o texto cadastrado pelo operador. Não há GPS automático ou envio de
um pin de mapa. O transporte usa a
[operação sendPhoto do Telegram](https://core.telegram.org/bots/api#sendphoto).

## Credenciais e preferências

Credenciais digitadas na tela permanecem na sessão. Para carregá-las em novas
sessões, copie `.streamlit/secrets.example.toml` para
`.streamlit/secrets.toml` e preencha **apenas o arquivo local**. Reinicie o app
após alterar esse arquivo. Alternativamente, forneça `TELEGRAM_BOT_TOKEN` e
`TELEGRAM_CHAT_ID` no ambiente do processo. Para e-mail, as variáveis seguem o
nome do campo com prefixo `EMAIL_`, como `EMAIL_HOST`, `EMAIL_USERNAME`,
`EMAIL_PASSWORD` e `EMAIL_ACCESS_TOKEN`.

As preferências sem segredos são salvas em `reports/settings.json`. Token do
Telegram, senha SMTP e token OAuth2 não devem estar nesse JSON nem nos arquivos
de ocorrência. `.streamlit/secrets.toml` é ignorado pelo Git e não deve ser
incluído na imagem Docker. Restrinja o acesso à pasta de relatórios e ao arquivo
de segredos às pessoas responsáveis pela demonstração.

## E-mail e Outlook como segundo canal

O formulário de e-mail permite configurar host, porta, usuário, remetente,
destinatário, segurança de transporte e autenticação. A foto segue como anexo.

| Provedor | Host | Porta / transporte | Autenticação |
| --- | --- | --- | --- |
| Outlook.com pessoal | `smtp-mail.outlook.com` | `587` / STARTTLS | OAuth2 |
| Microsoft 365 corporativo | `smtp.office365.com` | `587` / STARTTLS | OAuth2, sujeito à política de SMTP AUTH da organização |
| Outro SMTP | Informado pelo provedor | STARTTLS ou SSL conforme o provedor | Senha, se o provedor permitir |

Configurações verificadas na documentação oficial de
[Outlook.com](https://support.microsoft.com/en-US/Outlook/pop-imap-and-smtp-settings-for-outlook-com)
e [Microsoft 365](https://learn.microsoft.com/en-us/exchange/mail-flow-best-practices/how-to-set-up-a-multifunction-device-or-application-to-send-email-using-microsoft-365-or-office-365).

Para Microsoft, selecione OAuth2 e forneça um **access token válido para SMTP**.
Um token obtido apenas para Microsoft Graph não serve para esse protocolo.
A obtenção do token exige um aplicativo Microsoft Entra configurado e as
permissões aplicáveis; no fluxo delegado, o escopo de envio SMTP é
`https://outlook.office.com/SMTP.Send`. Consulte a
[autenticação OAuth para SMTP da Microsoft](https://learn.microsoft.com/en-us/exchange/client-developer/legacy-protocols/how-to-authenticate-an-imap-pop-smtp-application-by-using-oauth).

O SafeGuard não implementa o fluxo de consentimento/login nem o refresh token.
Quando o access token expirar, atualize-o manualmente; a falha será registrada
na ocorrência. Portanto, Telegram é o caminho indicado para a primeira
demonstração. A senha comum de uma conta Outlook não substitui o fluxo OAuth2.

## Como conferir um envio

Cada ocorrência mantém o estado separado por canal:

| Estado | Significado |
| --- | --- |
| Pendente (`pending`) | Envio ainda não concluído |
| Aceito (`accepted`) | O serviço aceitou a solicitação |
| Falhou (`failed`) | Houve falha de configuração, conexão ou recusa do serviço |

**Aceito não confirma entrega nem leitura.** Confira o chat/destinatário na
demonstração. Canais desativados não enviam, e as ocorrências locais podem
continuar sendo registradas sem conexão com Telegram ou e-mail. A configuração
e os testes automatizados do projeto não enviam mensagens reais.

Os envios usam uma fila em memória. **Parar** permite concluir o que já foi
agendado, mas encerrar o processo pode deixar registros pendentes; eles não são
reenviados nem reclassificados automaticamente na próxima execução. Permanecem
protegidos da retenção e precisam de revisão do operador. Uma fila cheia mantém a imagem
local e registra falha de agendamento. Em falhas de rede ou timeout, confira o
destino antes de tentar novamente: a solicitação pode ter sido aceita sem que
o aplicativo tenha recebido a resposta.

Se a imagem aparecer no monitor, mas nenhuma ocorrência surgir, confira a regra
selecionada, o registro automático, o tempo de confirmação e o intervalo entre
ocorrências. Se uma ocorrência existir sem mensagem no Telegram, confira a
ativação do canal, seu estado de envio, o Chat ID e se o bot recebeu `/start`.
