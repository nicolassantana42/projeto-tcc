#!/usr/bin/env python3
# ============================================================
#  test_camera.py — Script de Teste e Diagnóstico de Câmeras
#  TCC: Monitoramento de EPI com Visão Computacional
#
#  Uso:
#    python test_camera.py                    # Testa câmera configurada
#    python test_camera.py --detect           # Detecta todas câmeras
#    python test_camera.py --index 1          # Testa câmera específica
#    python test_camera.py --rtsp URL         # Testa câmera IP
# ============================================================

import sys
import argparse
import cv2
from pathlib import Path

# Adiciona raiz ao path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.camera.capture import VideoCapture, CameraDetector
from loguru import logger


def test_specific_camera(source, duration=10):
    """
    Testa uma câmera específica e exibe preview.
    
    Args:
        source: Índice (int) ou URL (str) da câmera
        duration: Duração do teste em segundos
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"🎬 TESTANDO CÂMERA: {source}")
    logger.info(f"{'='*70}\n")
    
    cap = VideoCapture(camera_source=source, auto_detect=False)
    
    logger.info(f"⏱️  Capturando por {duration} segundos...")
    logger.info("   Pressione 'Q' ou 'ESC' para sair antes\n")
    
    frame_count = 0
    start_time = None
    
    try:
        for frame in cap.stream():
            import time
            if start_time is None:
                start_time = time.time()
            
            frame_count += 1
            
            # Adiciona informações no frame
            h, w = frame.shape[:2]
            info_text = f"Frame: {frame_count} | FPS: {cap.current_fps:.1f} | Res: {w}x{h}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(frame, "Pressione Q para sair", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            
            # Mostra o frame
            cv2.imshow(f"Teste de Camera - {source}", frame)
            
            # Log a cada 30 frames
            if frame_count % 30 == 0:
                logger.info(f"  📊 Frames: {frame_count} | FPS: {cap.current_fps:.1f}")
            
            # Verifica tecla
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # Q ou ESC
                logger.info("\n⏹️  Teste interrompido pelo usuário")
                break
            
            # Verifica tempo
            if time.time() - start_time >= duration:
                logger.info(f"\n⏱️  Tempo de teste ({duration}s) completado")
                break
                
    except KeyboardInterrupt:
        logger.info("\n⏹️  Teste interrompido (Ctrl+C)")
    except Exception as e:
        logger.error(f"\n❌ Erro durante teste: {e}")
    finally:
        cv2.destroyAllWindows()
    
    logger.info(f"\n{'='*70}")
    logger.success(f"✅ TESTE CONCLUÍDO")
    logger.info(f"  Total de frames capturados: {frame_count}")
    logger.info(f"  FPS médio: {cap.current_fps:.1f}")
    logger.info(f"{'='*70}\n")


def detect_all_cameras():
    """Detecta e lista todas as câmeras disponíveis."""
    logger.info(f"\n{'='*70}")
    logger.info("🔍 DETECTANDO TODAS AS CÂMERAS DISPONÍVEIS")
    logger.info(f"{'='*70}\n")
    
    available = CameraDetector.detect_available_cameras(max_test=10)
    
    if not available:
        logger.error("❌ Nenhuma câmera detectada!")
        logger.info("\n💡 DICAS:")
        logger.info("  1. Conecte uma webcam USB")
        logger.info("  2. Verifique se a câmera integrada está habilitada")
        logger.info("  3. Feche outros programas que possam estar usando a câmera")
        logger.info("  4. Para câmeras IP, use: python test_camera.py --rtsp URL")
    else:
        logger.success(f"\n✅ {len(available)} câmera(s) encontrada(s)!\n")
        
        # Testa cada câmera brevemente
        for idx in available:
            logger.info(f"\n📷 Testando câmera {idx}...")
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    logger.success(f"  ✅ Índice {idx}:")
                    logger.info(f"     Resolução: {w}x{h}")
                    logger.info(f"     FPS: {int(fps) if fps > 0 else 'N/A'}")
                cap.release()
        
        logger.info(f"\n{'='*70}")
        logger.info("💡 COMO USAR:")
        logger.info(f"  1. Escolha um índice (ex: {available[0]})")
        logger.info(f"  2. Configure no .env: CAMERA_INDEX={available[0]}")
        logger.info(f"  3. Ou teste com: python test_camera.py --index {available[0]}")
        logger.info(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Script de teste e diagnóstico de câmeras",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  Detectar todas as câmeras:
    python test_camera.py --detect

  Testar câmera específica por índice:
    python test_camera.py --index 1

  Testar câmera IP (Intelbras):
    python test_camera.py --rtsp "rtsp://admin:senha@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"

  Testar com duração customizada (20 segundos):
    python test_camera.py --index 0 --duration 20
        """
    )
    
    parser.add_argument(
        '--detect', 
        action='store_true',
        help='Detecta e lista todas as câmeras disponíveis'
    )
    
    parser.add_argument(
        '--index', 
        type=int,
        help='Testa câmera no índice especificado (ex: 0, 1, 2)'
    )
    
    parser.add_argument(
        '--rtsp', 
        type=str,
        help='Testa câmera IP via URL RTSP/HTTP'
    )
    
    parser.add_argument(
        '--duration', 
        type=int,
        default=10,
        help='Duração do teste em segundos (padrão: 10)'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "="*70)
    print("  🎥 TESTE E DIAGNÓSTICO DE CÂMERAS")
    print("  TCC - Sistema de Monitoramento de EPI")
    print("="*70 + "\n")
    
    # Executa ação apropriada
    if args.detect:
        detect_all_cameras()
    elif args.index is not None:
        test_specific_camera(args.index, args.duration)
    elif args.rtsp:
        test_specific_camera(args.rtsp, args.duration)
    else:
        # Modo padrão: usa configuração do .env
        from config import CAMERA_INDEX
        logger.info(f"📋 Usando câmera configurada no .env: {CAMERA_INDEX}")
        test_specific_camera(CAMERA_INDEX, args.duration)


if __name__ == "__main__":
    main()
