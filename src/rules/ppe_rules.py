# ============================================================
#  src/rules/ppe_rules.py
#  Motor de Regras EPI — CORAÇÃO DO TCC
#  TCC: Monitoramento de EPI com Visão Computacional
# ============================================================
#
#  CONTRIBUIÇÃO CIENTÍFICA:
#  ────────────────────────
#  Esta lógica vai além de simples detecção de objetos.
#  Ela associa cada EPI à pessoa CORRETA usando análise
#  espacial de bounding boxes, suportando múltiplas
#  pessoas no mesmo frame de forma independente.
#
# ============================================================

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

# Garante que a raiz do projeto esteja no sys.path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import HEAD_REGION_RATIO, HELMET_IOU_THRESHOLD, REQUIRE_HELMET, REQUIRE_VEST
from src.ai.detector import Detection, FrameResult


@dataclass
class PersonStatus:
    """
    Status completo de EPI para uma pessoa detectada no frame.
    """
    person:     Detection
    has_helmet: bool       = False
    has_vest:   bool       = False
    has_boots:  bool       = False  # NOVO: Suporte a botas
    head_box:   List[int]  = field(default_factory=list)
    feet_box:   List[int]  = field(default_factory=list)  # NOVO: Região dos pés
    helmet_iou: float      = 0.0
    boots_iou:  float      = 0.0  # NOVO: IoU das botas
    violations: List[str]  = field(default_factory=list)

    @property
    def is_compliant(self) -> bool:
        return len(self.violations) == 0
    
    @property
    def person_bbox(self) -> List[int]:
        """Retorna bounding box da pessoa [x1, y1, x2, y2]."""
        return self.person.bbox

    def __str__(self) -> str:
        tag = "✅ CONFORME" if self.is_compliant else f"⚠️  {', '.join(self.violations)}"
        return f"Pessoa ({self.person.x1},{self.person.y1}) → {tag}"


class PPERulesEngine:
    """
    Motor de regras que associa EPIs detectados às pessoas corretas.

    METODOLOGIA (contribuição científica do TCC):
    ─────────────────────────────────────────────
    Para cada pessoa P detectada no frame:

    1. REGIÃO DA CABEÇA:
       Extrai os N% superiores da bounding box de P.
       Parâmetro: HEAD_REGION_RATIO (padrão: 30%)

    2. ASSOCIAÇÃO CAPACETE (IoU-based):
       Para cada capacete H detectado, calcula IoU entre H
       e a região da cabeça de P. Se IoU ≥ threshold → capacete OK.

    3. ASSOCIAÇÃO COLETE (centro-point):
       Verifica se o centro do colete está dentro da bbox de P.

    4. VIOLAÇÃO:
       Gera alerta apenas para pessoas sem os EPIs obrigatórios.

    VANTAGEM vs. detecção simples:
       Detecção simples: "há capacete no frame?" → falsos negativos
       Nossa abordagem: "ESTA pessoa tem capacete?" → preciso para N pessoas
    """

    def __init__(self,
                 require_helmet: bool = REQUIRE_HELMET,
                 require_vest:   bool = REQUIRE_VEST,
                 require_boots:  bool = False):  # NOVO: Suporte a botas
        self.require_helmet = require_helmet
        self.require_vest   = require_vest
        self.require_boots  = require_boots  # NOVO
        logger.info(
            f"🔧 Motor de Regras EPI | "
            f"Capacete: {'OBRIGATÓRIO' if require_helmet else 'opcional'} | "
            f"Colete: {'OBRIGATÓRIO' if require_vest else 'opcional'} | "
            f"Botas: {'OBRIGATÓRIO' if require_boots else 'opcional'}"  # NOVO
        )

    def evaluate(self, frame_result: FrameResult) -> List[PersonStatus]:
        """
        Avalia o status de EPI de cada pessoa no frame.

        Returns:
            Lista de PersonStatus, um por pessoa detectada.
        """
        statuses = []

        for person in frame_result.persons:
            status = PersonStatus(person=person)

            # 1. Calcula região da cabeça
            status.head_box = self._compute_head_region(person)

            # 2. Verifica capacete
            if frame_result.helmets:
                best_iou, _ = self._find_best_helmet(status.head_box, frame_result.helmets)
                status.helmet_iou = best_iou
                status.has_helmet = best_iou >= HELMET_IOU_THRESHOLD
                if status.has_helmet:
                    logger.debug(f"✅ Capacete associado ({person.x1},{person.y1}) IoU={best_iou:.3f}")
            else:
                status.has_helmet = False

            # 3. Verifica colete
            if frame_result.vests:
                status.has_vest = self._check_vest(person, frame_result.vests)
            else:
                status.has_vest = False

            # 4. Verifica botas (NOVO)
            # Calcula região dos pés (20% inferiores da bbox)
            status.feet_box = self._compute_feet_region(person)
            # TODO: Implementar detecção de botas quando modelo treinado disponível
            # Por enquanto, assume que não há botas detectadas
            status.has_boots = False

            # 5. Gera violações
            if self.require_helmet and not status.has_helmet:
                status.violations.append("SEM CAPACETE")
            if self.require_vest and not status.has_vest:
                status.violations.append("SEM COLETE")
            if self.require_boots and not status.has_boots:
                status.violations.append("SEM BOTAS")  # NOVO

            statuses.append(status)
            logger.debug(str(status))

        return statuses

    # ── Análise Espacial ──────────────────────────────────────

    def _compute_head_region(self, person: Detection) -> List[int]:
        """
        Extrai a região da cabeça como os N% superiores da bbox.

        Visualização:
          ┌─────────┐  ← y1
          │  CABEÇA │  ← HEAD_REGION_RATIO = 30%
          ├─────────┤  ← y1 + head_height
          │         │
          │  TORSO  │
          │         │
          └─────────┘  ← y2
        """
        head_height = int(person.height * HEAD_REGION_RATIO)
        return [person.x1, person.y1, person.x2, person.y1 + head_height]

    def _compute_feet_region(self, person: Detection) -> List[int]:
        """
        Extrai a região dos pés como os 20% inferiores da bbox.

        Visualização:
          ┌─────────┐  ← y1
          │  CABEÇA │
          ├─────────┤
          │  TORSO  │
          │         │
          ├─────────┤  ← y2 - feet_height
          │   PERNAS│  
          │  /BOTAS │  ← FEET_REGION_RATIO = 20%
          └─────────┘  ← y2
        """
        FEET_REGION_RATIO = 0.20
        feet_height = int(person.height * FEET_REGION_RATIO)
        return [person.x1, person.y2 - feet_height, person.x2, person.y2]

    def _compute_iou(self, box_a: List[int], box_b: List[int]) -> float:
        """
        Calcula IoU (Intersection over Union) entre duas bounding boxes.
        IoU = 0.0 → sem sobreposição | IoU = 1.0 → sobreposição perfeita
        """
        ix1 = max(box_a[0], box_b[0])
        iy1 = max(box_a[1], box_b[1])
        ix2 = min(box_a[2], box_b[2])
        iy2 = min(box_a[3], box_b[3])

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        intersection = iw * ih

        if intersection == 0:
            return 0.0

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union  = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0

    def _find_best_helmet(self,
                          head_box: List[int],
                          helmets: List[Detection]) -> Tuple[float, Optional[Detection]]:
        """Encontra o capacete com maior IoU em relação à cabeça."""
        best_iou    = 0.0
        best_helmet = None
        for helmet in helmets:
            iou = self._compute_iou(head_box, helmet.bbox)
            if iou > best_iou:
                best_iou    = iou
                best_helmet = helmet
        return best_iou, best_helmet

    def _check_vest(self, person: Detection, vests: List[Detection]) -> bool:
        """Verifica se o centro de algum colete está dentro da bbox da pessoa."""
        for vest in vests:
            if (person.x1 <= vest.center_x <= person.x2 and
                    person.y1 <= vest.center_y <= person.y2):
                return True
        return False


# ── Visualização ─────────────────────────────────────────────

def draw_ppe_status(frame: np.ndarray, statuses: List[PersonStatus]) -> np.ndarray:
    """
    Desenha o status de EPI no frame.

    Verde  → Conforme (todos os EPIs presentes)
    Vermelho → Infração detectada
    """
    import cv2

    for status in statuses:
        p     = status.person
        ok    = status.is_compliant
        color = (0, 200, 60) if ok else (0, 50, 220)
        thick = 2 if ok else 3

        # Bounding box da pessoa
        cv2.rectangle(frame, (p.x1, p.y1), (p.x2, p.y2), color, thick)

        # Tag de status
        tag        = "✓ CONFORME" if ok else "✗ " + " | ".join(status.violations)
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        (tw, th), _ = cv2.getTextSize(tag, font, font_scale, 2)

        bg_color = (0, 130, 40) if ok else (0, 30, 190)
        ty = max(p.y1 - 10, th + 10)

        cv2.rectangle(frame, (p.x1, ty - th - 8), (p.x1 + tw + 10, ty + 2), bg_color, -1)
        cv2.putText(frame, tag, (p.x1 + 5, ty - 2),
                    font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        # Região da cabeça (debug visual)
        if status.head_box:
            hx1, hy1, hx2, hy2 = status.head_box
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 165, 255), 1)

    return frame
