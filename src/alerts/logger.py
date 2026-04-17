# ============================================================
#  src/alerts/logger.py
#  Sistema de registro de infrações
#  TCC: Monitoramento de EPI com Visão Computacional
# ============================================================

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from loguru import logger

# Garante que a raiz do projeto esteja no sys.path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import VIOLATIONS_DIR, LOGS_DIR
from src.rules.ppe_rules import PersonStatus


class ViolationLogger:
    """
    Registra infrações de EPI em três formatos:
      1. Imagem PNG salva em violations/
      2. Log em arquivo via loguru (ppe_system.log)
      3. JSON estruturado para análise/métricas (violations.json)

    Nomenclatura de arquivo:
      sem_capacete_2024_05_20_143052.png
    """

    def __init__(self):
        self.violations_dir = VIOLATIONS_DIR
        self.logs_dir       = LOGS_DIR
        self.json_log_path  = self.logs_dir / "violations.json"

        # Configura loguru para arquivo
        log_file = self.logs_dir / "ppe_system.log"
        logger.add(
            str(log_file),
            rotation="10 MB",
            retention="30 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
            encoding="utf-8",
            enqueue=True,
        )

        # Cria JSON vazio se não existir
        if not self.json_log_path.exists():
            self.json_log_path.write_text("[]", encoding="utf-8")

        logger.info(f"📝 Logger iniciado | violations: {self.violations_dir}")

    def log_violation(self,
                      frame:    np.ndarray,
                      statuses: List[PersonStatus]) -> Optional[Path]:
        """
        Registra uma violação detectada no frame.

        Args:
            frame    : Frame BGR com a infração
            statuses : Lista de PersonStatus do motor de regras

        Returns:
            Path da imagem salva, ou None se sem violação.
        """
        violators = [s for s in statuses if not s.is_compliant]
        if not violators:
            return None

        timestamp  = datetime.now()
        ts_str     = timestamp.strftime("%Y_%m_%d_%H%M%S")
        ts_display = timestamp.strftime("%d/%m/%Y %H:%M:%S")

        # Agrega todos os tipos de violação
        all_violations: List[str] = []
        for status in violators:
            all_violations.extend(status.violations)
        unique_violations = list(set(all_violations))

        # Nome do arquivo
        tag            = "sem_capacete" if "SEM CAPACETE" in all_violations else "sem_epi"
        image_filename = f"{tag}_{ts_str}.png"
        image_path     = self.violations_dir / image_filename

        # Salva imagem
        saved = self._save_image(frame, image_path)

        # Log no console + arquivo
        n = len(violators)
        summary = f"{n} pessoa(s) | {', '.join(unique_violations)}"
        logger.warning(f"⚠️  VIOLAÇÃO | {ts_display} | {summary} | {image_filename}")

        # Registro JSON
        if saved:
            self._append_json({
                "timestamp":  timestamp.isoformat(),
                "image_file": image_filename,
                "violators":  n,
                "violations": unique_violations,
                "details": [
                    {
                        "person_bbox": s.person.bbox,
                        "violations":  s.violations,
                        "helmet_iou":  round(s.helmet_iou, 4),
                        "has_helmet":  s.has_helmet,
                        "has_vest":    s.has_vest,
                    }
                    for s in violators
                ]
            })

        return image_path if saved else None

    def get_today_count(self) -> int:
        """Número de violações registradas hoje (para Dashboard)."""
        try:
            records = json.loads(self.json_log_path.read_text(encoding="utf-8"))
            today   = datetime.now().date().isoformat()
            return sum(1 for r in records if r["timestamp"].startswith(today))
        except Exception:
            return 0

    def get_recent_violations(self, n: int = 10) -> list:
        """N violações mais recentes para o Dashboard."""
        try:
            records = json.loads(self.json_log_path.read_text(encoding="utf-8"))
            return sorted(records, key=lambda x: x["timestamp"], reverse=True)[:n]
        except Exception:
            return []

    def _save_image(self, frame: np.ndarray, path: Path) -> bool:
        try:
            cv2.imwrite(str(path), frame)
            return True
        except Exception as e:
            logger.error(f"Falha ao salvar imagem: {e}")
            return False

    def _append_json(self, record: dict) -> None:
        try:
            existing = json.loads(self.json_log_path.read_text(encoding="utf-8"))
            existing.append(record)
            self.json_log_path.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"Falha ao gravar JSON: {e}")
