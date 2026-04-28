# src/rules/ppe_rules.py
import time
import config

class PPERulesEngine:
    def __init__(self):
        self.last_alerts = {}
        
        # NOVO: Dicionário para rastrear há quantos frames seguidos alguém está sem EPI
        # Como não temos um "Tracker" avançado (ex: DeepSORT), vamos usar uma heurística baseada 
        # em se existe QUALQUER violação na tela por X frames seguidos.
        self.consecutive_violations = 0
        self.FRAMES_TO_CONFIRM_VIOLATION = 10 # Só acusa violação se falhar por 10 frames seguidos

    def _boxes_intersect_with_margin(self, person_box, epi_box, margin_ratio=0.3):
        """
        Verifica interseção, mas expande a caixa da pessoa virtualmente para garantir
        que EPIs na extremidade (cabeça/pé) sejam detectados mesmo se o YOLO errar a caixa da pessoa.
        """
        px1, py1, px2, py2 = person_box
        ex1, ey1, ex2, ey2 = epi_box

        # Expande a caixa da pessoa verticalmente (30% para cima e para baixo)
        height = py2 - py1
        py1_expanded = py1 - (height * margin_ratio)
        py2_expanded = py2 + (height * margin_ratio)
        
        # Expande horizontalmente (20% para os lados)
        width = px2 - px1
        px1_expanded = px1 - (width * 0.2)
        px2_expanded = px2 + (width * 0.2)

        # Verifica interseção com a caixa expandida
        if px1_expanded > ex2 or px2_expanded < ex1 or py1_expanded > ey2 or py2_expanded < ey1:
            return False
        return True

    def check_violation(self, detections):
        persons = [d for d in detections if d['class_id'] == config.CLASS_PERSON]
        helmets = [d for d in detections if d['class_id'] == config.CLASS_HELMET]
        boots =   [d for d in detections if d['class_id'] == config.CLASS_BOOT]

        current_frame_violators = []

        for person in persons:
            has_helmet = False
            has_boot = False

            if config.REQUIRE_HELMET:
                for helmet in helmets:
                    # Usa a interseção com margem de segurança
                    if self._boxes_intersect_with_margin(person['bbox'], helmet['bbox']):
                        has_helmet = True
                        break

            if config.REQUIRE_BOOT:
                for boot in boots:
                    # Usa a interseção com margem de segurança
                    if self._boxes_intersect_with_margin(person['bbox'], boot['bbox']):
                        has_boot = True
                        break

            is_violating = False
            if config.REQUIRE_HELMET and not has_helmet:
                is_violating = True
            if config.REQUIRE_BOOT and not has_boot:
                is_violating = True

            if is_violating:
                current_frame_violators.append(person)

        # ==========================================
        # FILTRO TEMPORAL (DEBOUNCE)
        # ==========================================
        if current_frame_violators:
            # Se viu uma violação neste frame, aumenta o contador
            self.consecutive_violations += 1
        else:
            # Se está todo mundo OK neste frame, zera a contagem de violações
            self.consecutive_violations = 0

        # Só retorna a lista de infratores se a violação for consistente (durou X frames)
        if self.consecutive_violations >= self.FRAMES_TO_CONFIRM_VIOLATION:
            return current_frame_violators
        else:
            return [] # Ignora a violação temporariamente pois pode ser flickering do modelo

    def can_send_alert(self, cam_id):
        current_time = time.time()
        last_alert_time = self.last_alerts.get(cam_id, 0)
        
        if (current_time - last_alert_time) >= config.ALERT_COOLDOWN_SECONDS:
            self.last_alerts[cam_id] = current_time
            return True
            
        return False