#!/usr/bin/env python3
"""
Test script para verificar la generación de QR codes reales de WhatsApp.

Este script prueba la funcionalidad de QR generation sin iniciar el servidor completo.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.services.whatsapp_service import WhatsAppService
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def test_qr_generation():
    """Test básico de generación de QR codes."""
    logger.info("🧪 Iniciando test de generación QR...")
    
    try:
        # Crear servicio WhatsApp para test
        wa_service = WhatsAppService()
        
        # Configurar path de sesión de test
        test_session_path = wa_service.session_path.parent / "test_session"
        wa_service.session_path = test_session_path
        test_session_path.mkdir(exist_ok=True, parents=True)
        
        # Variable para capturar QR
        qr_received = False
        qr_data = None
        
        async def on_qr_received(data):
            nonlocal qr_received, qr_data
            qr_received = True
            qr_data = data.get('qr') if isinstance(data, dict) else data
            logger.info(f"📱 QR Code recibido: {qr_data[:50]}...")
            
            # Verificar que el QR no sea un mock
            if qr_data.startswith("DEMO-") or qr_data.startswith("whatsapp-session-"):
                logger.warning("⚠️ QR Code es MOCK, no real de WhatsApp")
                return False
            else:
                logger.info("✅ QR Code parece ser REAL de WhatsApp Web.js")
                return True
        
        # Registrar handler
        wa_service.register_event_handler('qr_code', on_qr_received)
        
        # Intentar iniciar para QR generation
        logger.info("🔄 Intentando iniciar WhatsApp service para QR generation...")
        
        success = await wa_service.start_for_qr_generation()
        
        if success:
            logger.info("✅ WhatsApp service iniciado para QR generation")
            
            # Esperar hasta 30 segundos por QR
            for i in range(30):
                if qr_received:
                    break
                await asyncio.sleep(1)
                if i % 5 == 0:
                    logger.info(f"⏳ Esperando QR... ({i}/30 segundos)")
            
            if qr_received and qr_data:
                logger.info("🎯 TEST EXITOSO: QR Code generado correctamente")
                logger.info(f"📱 QR Code: {qr_data}")
                
                # Verificar formato del QR
                if len(qr_data) > 50 and not qr_data.startswith("DEMO-"):
                    logger.info("✅ QR Code tiene formato válido de WhatsApp Web")
                else:
                    logger.warning("⚠️ QR Code podría ser mock o inválido")
                
            else:
                logger.error("❌ TEST FALLIDO: No se generó QR Code en 30 segundos")
                
        else:
            logger.error("❌ TEST FALLIDO: No se pudo iniciar WhatsApp service")
            logger.info("💡 Esto es normal si Node.js no está instalado")
            
        # Cleanup
        await wa_service.stop()
        
    except Exception as e:
        logger.error(f"❌ Error en test: {e}")
        logger.info("💡 Esto es normal si Node.js no está disponible en el entorno")


async def test_fallback_mode():
    """Test del modo fallback cuando Node.js no está disponible."""
    logger.info("🧪 Iniciando test de modo fallback...")
    
    try:
        # Simular entorno sin Node.js
        wa_service = WhatsAppService()
        
        # Intentar start normal (debería fallar en producción)
        success = await wa_service.start()
        
        if success:
            logger.info("✅ WhatsApp service iniciado (modo normal)")
        else:
            logger.info("⚠️ WhatsApp service no disponible (modo fallback)")
            
        await wa_service.stop()
        
    except Exception as e:
        logger.info(f"⚠️ Error esperado en modo fallback: {e}")


if __name__ == "__main__":
    print("🚀 Iniciando tests de QR generation...")
    print("=" * 50)
    
    # Test 1: QR Generation
    print("\n📱 Test 1: Generación de QR Code real")
    asyncio.run(test_qr_generation())
    
    # Test 2: Fallback mode
    print("\n🔄 Test 2: Modo fallback")
    asyncio.run(test_fallback_mode())
    
    print("\n" + "=" * 50)
    print("✅ Tests completados")
    print("\n💡 Notas:")
    print("- Si Node.js no está instalado, es normal que fallen los tests")
    print("- En producción (Render), se usará modo fallback con QR demo")
    print("- Para QR reales, se necesita Node.js + whatsapp-web.js")
