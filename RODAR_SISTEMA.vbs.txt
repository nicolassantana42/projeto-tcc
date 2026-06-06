' RODAR_SISTEMA.vbs
' Inicialização oculta para o ecossistema de monitoramento de EPIs - Ajustado para o Drive D
Set WshShell = CreateObject("WScript.Shell")

' ── FORÇA O SCRIPT A IR PARA O DISCO D E ENTRAR NA PASTA DO PROJETO ──
WshShell.CurrentDirectory = "D:\tcc\projeto-tcc"

' 1. Inicializa o motor de processamento de IA (YOLOv5) em background oculto
WshShell.Run "cmd /c python main_industrial.py", 0, False

' 2. Aguarda 3 segundos para a estabilização do banco de dados SQLite
WScript.Sleep 3000

' 3. Inicializa a interface web do Streamlit em background oculto
WshShell.Run "cmd /c streamlit run src/dashboard/streamlit_app.py --server.port 8501 --server.address 0.0.0.0", 0, False

Set WshShell = Nothing