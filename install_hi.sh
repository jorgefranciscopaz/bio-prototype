#!/bin/bash

echo "===================================="
echo "  Instalador del Proyecto HI (H!)"
echo "===================================="

# -------------------------------
# Verificar Python
# -------------------------------
if ! command -v python &> /dev/null; then
    echo "❌ Python no está instalado o no está en el PATH"
    exit 1
fi

PY_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

if [[ "$PY_VERSION" != "3.10" ]]; then
    echo "❌ Python $PY_VERSION detectado"
    echo "👉 Este proyecto requiere Python 3.10.x"
    exit 1
fi

echo "✅ Python 3.10 detectado"

# -------------------------------
# Crear entorno virtual
# -------------------------------
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python -m venv venv
else
    echo "📦 Entorno virtual existente"
fi

# -------------------------------
# Activar entorno
# -------------------------------
source venv/bin/activate

# -------------------------------
# Actualizar pip
# -------------------------------
python -m pip install --upgrade pip

# -------------------------------
# Instalar dependencias
# -------------------------------
echo "⬇️ Instalando dependencias..."
pip install -r requirements.txt

# -------------------------------
# Verificación rápida
# -------------------------------
python - << END
import cv2, mediapipe, sklearn, firebase_admin
print("✅ Dependencias críticas cargadas correctamente")
END

echo "===================================="
echo " ✅ Instalación completada con éxito"
echo " 👉 Activa el entorno con:"
echo "    source venv/bin/activate"
echo "===================================="
