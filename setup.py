#!/usr/bin/env python3
"""
Script de setup para WhatsApp Chatbot.

Automatiza la configuración inicial del proyecto para desarrollo local.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def check_python_version():
    """Verifica que Python sea 3.8+"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ es requerido")
        print(f"Versión actual: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")


def check_node_installed():
    """Verifica que Node.js esté instalado"""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js version: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Node.js no está instalado")
    print("Instala Node.js desde: https://nodejs.org/")
    return False


def check_redis_available():
    """Verifica que Redis esté disponible"""
    try:
        result = subprocess.run(['redis-cli', 'ping'], capture_output=True, text=True)
        if result.returncode == 0 and 'PONG' in result.stdout:
            print("✅ Redis está disponible")
            return True
    except FileNotFoundError:
        pass
    
    print("⚠️ Redis no está disponible localmente")
    print("Para desarrollo local, instala Redis:")
    print("- Ubuntu/Debian: sudo apt install redis-server")
    print("- macOS: brew install redis") 
    print("- Windows: Usa Docker o WSL")
    return False


def create_virtual_environment():
    """Crea entorno virtual Python"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment ya existe")
        return True
    
    try:
        print("📦 Creando virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        print("✅ Virtual environment creado")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error creando virtual environment")
        return False


def install_python_dependencies():
    """Instala dependencias Python"""
    try:
        print("📦 Instalando dependencias Python...")
        
        # Determinar comando pip según OS
        if os.name == 'nt':  # Windows
            pip_cmd = ['venv\\Scripts\\pip']
        else:  # Unix
            pip_cmd = ['venv/bin/pip']
        
        # Upgrade pip
        subprocess.run(pip_cmd + ['install', '--upgrade', 'pip'], check=True)
        
        # Instalar requirements
        subprocess.run(pip_cmd + ['install', '-r', 'requirements.txt'], check=True)
        
        print("✅ Dependencias Python instaladas")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias Python: {e}")
        return False


def install_node_dependencies():
    """Instala dependencias Node.js para WhatsApp"""
    session_path = Path("session")
    session_path.mkdir(exist_ok=True)
    
    try:
        print("📦 Instalando dependencias Node.js...")
        
        # Cambiar al directorio session
        original_cwd = os.getcwd()
        os.chdir(session_path)
        
        # Crear package.json si no existe
        package_json = {
            "name": "whatsapp-chatbot-session",
            "version": "1.0.0",
            "description": "WhatsApp Web.js dependencies",
            "dependencies": {
                "whatsapp-web.js": "^1.21.0",
                "qrcode-terminal": "^0.12.0"
            }
        }
        
        if not Path("package.json").exists():
            import json
            with open("package.json", "w") as f:
                json.dump(package_json, f, indent=2)
        
        # Instalar dependencias
        subprocess.run(['npm', 'install'], check=True)
        
        os.chdir(original_cwd)
        print("✅ Dependencias Node.js instaladas")
        return True
        
    except subprocess.CalledProcessError as e:
        os.chdir(original_cwd)
        print(f"❌ Error instalando dependencias Node.js: {e}")
        return False


def create_env_file():
    """Crea archivo .env desde .env.example"""
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ Archivo .env ya existe")
        return True
    
    if not env_example.exists():
        print("❌ .env.example no encontrado")
        return False
    
    try:
        shutil.copy(env_example, env_file)
        print("✅ Archivo .env creado desde .env.example")
        print("⚠️  IMPORTANTE: Edita .env con tus valores reales:")
        print("   - OPENAI_API_KEY: Tu API key de OpenAI")
        print("   - REDIS_URL: URL de tu Redis local")
        print("   - Emails de departamentos")
        return True
    except Exception as e:
        print(f"❌ Error creando .env: {e}")
        return False


def run_initial_tests():
    """Ejecuta tests básicos para verificar setup"""
    try:
        print("🧪 Ejecutando tests básicos...")
        
        # Determinar comando pytest según OS
        if os.name == 'nt':  # Windows
            pytest_cmd = ['venv\\Scripts\\pytest']
        else:  # Unix
            pytest_cmd = ['venv/bin/pytest']
        
        # Ejecutar solo tests unitarios rápidos
        result = subprocess.run(
            pytest_cmd + ['tests/', '-v', '--tb=short', '-x'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Tests básicos pasaron")
            return True
        else:
            print("⚠️ Algunos tests fallaron (normal en setup inicial)")
            print("Verifica tu configuración .env")
            return False
            
    except subprocess.CalledProcessError:
        print("⚠️ No se pudieron ejecutar tests (pytest no encontrado)")
        return False


def print_next_steps():
    """Imprime próximos pasos después del setup"""
    print("\n" + "="*60)
    print("🎉 Setup completado!")
    print("="*60)
    print()
    print("Próximos pasos:")
    print()
    print("1. 📝 Edita el archivo .env con tus valores reales:")
    print("   - OPENAI_API_KEY=sk-tu-api-key-aquí")
    print("   - REDIS_URL=redis://localhost:6379")
    print("   - Configurar emails de departamentos")
    print()
    print("2. 🚀 Para desarrollo local:")
    
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix  
        print("   source venv/bin/activate")
    
    print("   python main.py")
    print()
    print("3. 🧪 Para ejecutar tests:")
    print("   pytest tests/ -v")
    print()
    print("4. 📱 Para conectar WhatsApp:")
    print("   - En primera ejecución aparecerá un QR code")
    print("   - Escanéalo con WhatsApp para autenticar")
    print("   - La sesión se guardará automáticamente")
    print()
    print("5. 🌐 Endpoints disponibles:")
    print("   - http://localhost:8000/health - Health check")
    print("   - http://localhost:8000/stats - Estadísticas")
    print("   - http://localhost:8000/ - Info general")
    print()
    print("¡Listo para comenzar! 🚀")
    print()


def main():
    """Función principal de setup"""
    print("🤖 WhatsApp Chatbot Setup")
    print("="*40)
    print()
    
    # Verificaciones previas
    check_python_version()
    
    node_available = check_node_installed()
    redis_available = check_redis_available()
    
    if not node_available:
        print("\n❌ Setup cannot continue without Node.js")
        sys.exit(1)
    
    # Setup steps
    steps = [
        ("Crear virtual environment", create_virtual_environment),
        ("Instalar dependencias Python", install_python_dependencies),
        ("Instalar dependencias Node.js", install_node_dependencies),
        ("Crear archivo .env", create_env_file),
        ("Ejecutar tests básicos", run_initial_tests)
    ]
    
    success_count = 0
    
    for step_name, step_func in steps:
        print(f"\n🔄 {step_name}...")
        if step_func():
            success_count += 1
        else:
            print(f"⚠️ {step_name} falló, pero continuando...")
    
    print(f"\n📊 Setup completado: {success_count}/{len(steps)} pasos exitosos")
    
    if success_count >= 4:  # Al menos los pasos críticos
        print_next_steps()
    else:
        print("\n❌ Setup tuvo demasiados errores. Revisa los mensajes arriba.")
        sys.exit(1)


if __name__ == "__main__":
    main()