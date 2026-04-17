# ============================================================
#  src/alerts/telegram.py
#  Envio de alertas via Telegram Bot API
#  TCC: Monitoramento de EPI com Visão Computacional
# ============================================================

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests
from loguru import logger

# Garante que a raiz do projeto esteja no sys.path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, ALERT_COOLDOWN_SECONDS


class TelegramAlerter:
    """
    Envia alertas de infração de EPI via Telegram Bot API.

    Configuração (.env):
      TELEGRAM_TOKEN   = token do bot (@BotFather → /newbot)
      TELEGRAM_CHAT_ID = ID do chat de destino

    Para obter o CHAT_ID após criar o bot:
      https://api.telegram.org/bot<TOKEN>/getUpdates
    """

    def __init__(self,
                 token:    str = TELEGRAM_TOKEN,
                 chat_id:  str = TELEGRAM_CHAT_ID,
                 cooldown: int = ALERT_COOLDOWN_SECONDS):
        self.token    = token
        self.chat_id  = chat_id
        self.cooldown = cooldown
        self._last_alert: float = 0.0
        self._enabled   = bool(token and chat_id and
                               token   != "seu_token_aqui" and
                               chat_id != "seu_chat_id_aqui")

        if self._enabled:
            logger.info("📱 Telegram Bot configurado e pronto!")
        else:
            logger.warning(
                "📱 Telegram NÃO configurado — alertas desabilitados.\n"
                "   → Preencha TELEGRAM_TOKEN e TELEGRAM_CHAT_ID no .env"
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def send_violation_alert(self,
                             violations:   List[str],
                             image_path:   Optional[Path] = None,
                             person_count: int = 1) -> bool:
        """
        Envia alerta completo (texto + foto).

        Args:
            violations   : Ex: ["SEM CAPACETE", "SEM COLETE"]
            image_path   : Frame salvo da infração
            person_count : Pessoas em infração no frame

        Returns:
            True se enviado com sucesso.
        """
        elapsed = time.time() - self._last_alert
        if elapsed < self.cooldown:
            logger.debug(f"Telegram: cooldown ({int(self.cooldown - elapsed)}s restantes)")
            return False

        if not self._enabled:
            return False

        now    = datetime.now().strftime("%d/%m/%Y às %H:%M:%S")
        viols  = "\n".join(f"  ⛔ {v}" for v in violations)
        plural = "s" if person_count > 1 else ""
        msg = (
            f"🚨 *ALERTA DE EPI — INFRAÇÃO DETECTADA*\n\n"
            f"🕐 *Horário:* {now}\n"
            f"👥 *Pessoa{plural} em infração:* {person_count}\n"
            f"📋 *Violações:*\n{viols}\n\n"
            f"_Sistema automático de monitoramento de EPI_"
        )

        success = False
        if image_path and Path(image_path).exists():
            success = self._send_photo(msg, Path(image_path))
        else:
            success = self._send_message(msg)

        if success:
            self._last_alert = time.time()
            logger.success(f"✅ Alerta Telegram enviado | {violations}")

        return success

    def test_connection(self) -> bool:
        """Envia mensagem de teste para validar a configuração."""
        if not self._enabled:
            logger.warning("Configure TELEGRAM_TOKEN e TELEGRAM_CHAT_ID primeiro.")
            return False

        msg = (
            "✅ *Sistema EPI — Online*\n\n"
            "Monitoramento de Equipamentos de Proteção Individual iniciado.\n\n"
            "_TCC — Visão Computacional e IA_"
        )
        ok = self._send_message(msg)
        if ok:
            logger.success("✅ Conexão Telegram validada!")
        else:
            logger.error("❌ Falha no Telegram. Verifique TOKEN e CHAT_ID.")
        return ok

    # ── Métodos internos ──────────────────────────────────────

    def _send_message(self, text: str) -> bool:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            r = requests.post(
                url,
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"},
                timeout=10
            )
            return r.status_code == 200
        except requests.RequestException as e:
            logger.error(f"Telegram sendMessage falhou: {e}")
            return False

    def _send_photo(self, caption: str, image_path: Path) -> bool:
        url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
        try:
            with open(image_path, "rb") as f:
                r = requests.post(
                    url,
                    data={"chat_id": self.chat_id, "caption": caption[:1024], "parse_mode": "Markdown"},
                    files={"photo": f},
                    timeout=15
                )
            return r.status_code == 200
        except (requests.RequestException, OSError) as e:
            logger.error(f"Telegram sendPhoto falhou: {e}")
            return self._send_message(caption)
