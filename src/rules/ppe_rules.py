import cv2
from typing import List
from src.ai.detector import FrameResult, Detection

class PersonStatus:
    """Armazena o estado de conformidade de cada pessoa detectada."""
    def __init__(self, person_det: Detection):
        self.person_bbox = person_det.bbox
        self.is_compliant = True
        self.violations = []
        self.found_elements = []  # Armazena quais EPIs foram validados

class PPERulesEngine:
    def __init__(self):
        # Configurações de obrigatoriedade (Podem vir do config.py)
        self.require_helmet = True
        self.require_vest = True
        self.require_boot = True
        
        # Margem de erro para considerar que um EPI pertence à pessoa (em pixels)
        self.pixel_margin = 15 

    def _is_inside(self, epi_bbox: List[int], person_bbox: List[int]) -> bool:
        """Verifica se o centro do EPI está dentro da caixa da pessoa."""
        # Centro do EPI
        ex = (epi_bbox[0] + epi_bbox[2]) // 2
        ey = (epi_bbox[1] + epi_bbox[3]) // 2
        
        # Coordenadas da Pessoa
        px1, py1, px2, py2 = person_bbox
        
        # Verifica inclusão com pequena margem de tolerância
        return (px1 - self.pixel_margin <= ex <= px2 + self.pixel_margin) and \
               (py1 - self.pixel_margin <= ey <= py2 + self.pixel_margin)

    def evaluate(self, result: FrameResult) -> List[PersonStatus]:
        """
        Aplica a lógica de engenharia de prompt:
        Para cada pessoa, verifica se os EPIs obrigatórios estão presentes em seu BBox.
        """
        statuses = []

        for person in result.persons:
            status = PersonStatus(person)
            
            # 1. Verificar Capacete
            has_helmet = any(self._is_inside(h.bbox, person.bbox) for h in result.helmets)
            if self.require_helmet and not has_helmet:
                status.is_compliant = False
                status.violations.append("Capacete Ausente")
            elif has_helmet:
                status.found_elements.append("Capacete")

            # 2. Verificar Colete
            has_vest = any(self._is_inside(v.bbox, person.bbox) for v in result.vests)
            if self.require_vest and not has_vest:
                status.is_compliant = False
                status.violations.append("Colete Ausente")
            elif has_vest:
                status.found_elements.append("Colete")

            # 3. Verificar Bota (Classe que adicionamos)
            has_boot = any(self._is_inside(b.bbox, person.bbox) for b in result.boots)
            if self.require_boot and not has_boot:
                status.is_compliant = False
                status.violations.append("Bota Ausente")
            elif has_boot:
                status.found_elements.append("Bota")

            statuses.append(status)

        return statuses

def draw_ppe_status(frame, statuses: List[PersonStatus]):
    """Desenha as caixas coloridas e as violações no frame final."""
    for status in statuses:
        x1, y1, x2, y2 = status.person_bbox
        
        # Verde para Seguro, Vermelho para Inseguro
        color = (0, 255, 0) if status.is_compliant else (0, 0, 255)
        thickness = 2
        
        # Desenha BBox da pessoa
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Header do status
        label = "CONFORME" if status.is_compliant else "VIOLACAO"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 7), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Listagem de violações abaixo da caixa
        if not status.is_compliant:
            y_offset = y2 + 20
            for violation in status.violations:
                # Sombra para leitura
                cv2.putText(frame, f"! {violation}", (x1 + 1, y_offset + 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                # Texto do Alerta
                cv2.putText(frame, f"! {violation}", (x1, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                y_offset += 20
                
    return frame