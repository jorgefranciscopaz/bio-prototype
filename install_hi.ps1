Write-Host "===================================="
Write-Host "  Instalador del Proyecto HI (H!)"
Write-Host "===================================="

# -------------------------------
# Verificar Python
# -------------------------------
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Python no está instalado o no está en el PATH"
    exit
}

$pyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"

if ($pyVersion -ne "3.10") {
    Write-Host "❌ Python $pyVersion detectado"
    Write-Host "👉 Este proyecto requiere Python 3.10.x"
    exit
}

Write-Host "✅ Python 3.10 detectado"

# -------------------------------
# Crear entorno virtual
# -------------------------------
if (-not (Test-Path "venv")) {
    Write-Host "📦 Creando entorno virtual..."
    python -m venv venv
} else {
    Write-Host "📦 Entorno virtual existente"
}

# -------------------------------
# Activar entorno
# -------------------------------
.\venv\Scripts\Activate.ps1

# -------------------------------
# Actualizar pip
# -------------------------------
python -m pip install --upgrade pip

# -------------------------------
# Crear requirements.txt
# -------------------------------
@"
opencv-python==4.9.0.80
mediapipe==0.10.9
numpy==1.26.4
pandas==2.2.2
scipy==1.11.4
scikit-learn==1.4.2
joblib==1.4.2
firebase-admin==6.5.0
tqdm==4.66.4
protobuf==3.20.3
matplotlib==3.8.4
seaborn==0.13.2
"@ | Out-File requirements.txt -Encoding UTF8

# -------------------------------
# Instalar dependencias
# -------------------------------
Write-Host "⬇️ Instalando dependencias..."
pip install -r requirements.txt

# -------------------------------
# Verificación rápida
# -------------------------------
python -c "import cv2, mediapipe, sklearn, firebase_admin; print('✅ Dependencias críticas cargadas correctamente')"

Write-Host "===================================="
Write-Host " ✅ Instalación completada con éxito"
Write-Host " 👉 Activa el entorno con:"
Write-Host "    .\venv\Scripts\Activate.ps1"
Write-Host "===================================="