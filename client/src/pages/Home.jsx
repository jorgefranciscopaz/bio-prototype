import { useState, useEffect } from "react";
import CameraFeed from "../components/CameraFeed";
import ToggleAIButton from "../components/ToggleAIButton";
import PredictionBox from "../components/PredictionBox";


export default function Home() {
  const [isAIActive, setIsAIActive] = useState(false);
  const [prediction, setPrediction] = useState("");
  const [status, setStatus] = useState("IA desactivada");

  useEffect(() => {
    let interval;

    const fetchPrediction = async () => {
      try {
        const response = await fetch("http://localhost:5000/predict");
        if (!response.ok) throw new Error("Error en la conexión con el backend");
        const data = await response.json();
        setPrediction(data.prediccion || "Sin detección");
      } catch (error) {
        console.error("❌ Error al obtener predicción:", error);
        setPrediction("Error de conexión");
      }
    };

    if (isAIActive) {
      setStatus("IA activada - analizando gestos...");
      fetchPrediction(); // Primera llamada inmediata
      interval = setInterval(fetchPrediction, 1000); // Llamadas cada 1s
    } else {
      setStatus("IA desactivada");
      clearInterval(interval);
      setPrediction("");
    }

    return () => clearInterval(interval);
  }, [isAIActive]);

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col items-center justify-center space-y-6 text-white px-4">
      {/* Cámara */}
      <CameraFeed isActive={isAIActive} />

      {/* Botón de encendido/apagado de IA */}
      <ToggleAIButton
        isActive={isAIActive}
        onToggle={() => setIsAIActive(!isAIActive)}
      />

      {/* Estado de conexión / predicción */}
      <div className="text-sm text-gray-400">{status}</div>

      {/* Resultado de predicción */}
      <PredictionBox prediction={prediction} />
    </div>
  );
}
