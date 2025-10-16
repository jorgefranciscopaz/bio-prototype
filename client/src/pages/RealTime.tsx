import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { ArrowLeft, CameraOff } from "lucide-react";
import { useNavigate } from "react-router-dom";

const RealTime = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [prediccion, setPrediccion] = useState<string>("Sin detección");
  const [confianza, setConfianza] = useState<number>(0);
  const [activo, setActivo] = useState(false);
  const [modelo, setModelo] = useState<string | null>(null);
  const navigate = useNavigate();

  // === Cargar modelo seleccionado ===
  useEffect(() => {
    const modeloGuardado = localStorage.getItem("modeloSeleccionado");
    if (modeloGuardado) {
      setModelo(modeloGuardado);
    } else {
      alert("⚠️ No hay modelo seleccionado. Vuelve al menú principal.");
      navigate("/");
    }
  }, [navigate]);

  // === Activar cámara ===
  const iniciarCamara = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" }, // Cámara frontal
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error("❌ Error al acceder a la cámara:", error);
    }
  };

  // === Enviar frame al backend (manteniendo orientación real) ===
  const enviarFrame = async () => {
    if (!videoRef.current) return;

    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext("2d");

    // 👉 NO invertimos el canvas, mostramos la cámara tal como se ve en la realidad
    ctx?.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

    const frame = canvas.toDataURL("image/jpeg");

    try {
      const res = await fetch("http://localhost:5000/predict_realtime", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: frame, modelo }),
      });
      const data = await res.json();
      if (data.prediccion) {
        setPrediccion(data.prediccion);
        setConfianza((data.confianza * 100).toFixed(1) as unknown as number);
      }
    } catch (error) {
      console.error("❌ Error en predicción:", error);
    }
  };

  // === Bucle de predicción ===
  useEffect(() => {
    let intervalo: NodeJS.Timeout;
    if (activo) {
      iniciarCamara();
      intervalo = setInterval(enviarFrame, 250);
    }
    return () => clearInterval(intervalo);
  }, [activo]);

  // === Detener cámara ===
  const detenerCamara = () => {
    setActivo(false);
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((t) => t.stop());
    }
  };

  return (
    <div className="relative min-h-screen bg-background text-white flex flex-col items-center justify-center">
      {/* Botón volver */}
      <Button
        onClick={() => {
          detenerCamara();
          navigate("/");
        }}
        className="absolute top-6 left-6 bg-gray-800 hover:bg-gray-700"
      >
        <ArrowLeft className="mr-2 h-4 w-4" /> Volver
      </Button>

      {/* Título */}
      <h1 className="text-3xl font-bold mb-4 text-center flex items-center gap-2">
        📡 Detección en Tiempo Real
      </h1>

      {/* Contenedor de cámara */}
      <div className="relative w-[90%] max-w-5xl aspect-video bg-black rounded-2xl border border-gray-700 overflow-hidden flex items-center justify-center shadow-lg">
        {!activo && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-400 space-y-3">
            <CameraOff className="w-16 h-16 text-gray-500" />
            <p className="text-sm">Cámara desactivada</p>
          </div>
        )}
        {/* 👇 Vista real (no espejo) */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`transition-all duration-500 object-cover w-full h-full ${
            activo ? "opacity-100" : "opacity-0"
          }`}
          style={{
            transform: "scaleX(-1)", // Esto corrige la orientación visual de la cámara frontal
          }}
        />
      </div>

      {/* Botones de control */}
      <div className="mt-8">
        {!activo ? (
          <Button
            onClick={() => setActivo(true)}
            className="bg-accent text-accent-foreground px-6 py-3 text-lg rounded-xl hover:bg-accent/90"
          >
            Activar Cámara y Detectar
          </Button>
        ) : (
          <Button
            onClick={detenerCamara}
            className="bg-red-600 hover:bg-red-700 px-6 py-3 text-lg rounded-xl"
          >
            Detener Detección
          </Button>
        )}
      </div>

      {/* Resultado de predicción */}
      <div className="mt-10 text-center">
        <p className="text-lg text-accent-foreground">
          <span className="text-accent font-semibold">Letra detectada:</span>{" "}
          <span className="text-3xl font-bold text-primary">{prediccion}</span>
        </p>
        <p className="text-sm mt-2 text-muted-foreground">
          Confianza: {confianza}%
        </p>
      </div>
    </div>
  );
};

export default RealTime;
