import { useRef, useState, useEffect } from "react";
import CameraFeed from "../components/CameraFeed";
import ToggleAIButton from "../components/ToggleAIButton";
import PredictionBox from "../components/PredictionBox";

export default function Home() {
  const [isAIActive, setIsAIActive] = useState(false);
  const [prediction, setPrediction] = useState("");
  const [status, setStatus] = useState("IA desactivada");
  const videoRef = useRef(null); // 👈 ESTE ref

  useEffect(() => {
    let interval;
    const sendFrameToAPI = async () => {
      if (!videoRef.current) return;
      const canvas = document.createElement("canvas");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      const imageBase64 = canvas.toDataURL("image/jpeg");
      try {
        const res = await fetch("http://localhost:5000/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: imageBase64 }),
        });
        const data = await res.json();
        setPrediction(data.prediccion || "Sin detección");
      } catch (err) {
        console.error(err);
        setPrediction("Error de conexión");
      }
    };

    if (isAIActive) {
      setStatus("IA activada - analizando...");
      interval = setInterval(sendFrameToAPI, 1000);
    } else {
      setStatus("IA desactivada");
      clearInterval(interval);
      setPrediction("");
    }

    return () => clearInterval(interval);
  }, [isAIActive]);

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col items-center justify-center text-white space-y-4">
      <CameraFeed ref={videoRef} isActive={isAIActive} />
      <ToggleAIButton isActive={isAIActive} onToggle={() => setIsAIActive(!isAIActive)} />
      <p className="text-gray-400">{status}</p>
      <PredictionBox prediction={prediction} />
    </div>
  );
}
