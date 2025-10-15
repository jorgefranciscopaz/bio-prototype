import { useRef, useState, useEffect } from "react";
import VideoInterface from "../components/VideoInterface";
import ControlBar from "../components/ControlBar";
import PredictionBox from "../components/PredictionBox";
import "../styles.css"; // incluye el CSS del espejo

export default function Home() {
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isStaticMode, setIsStaticMode] = useState(false);
  const [isGestureMode, setIsGestureMode] = useState(false);
  const [prediction, setPrediction] = useState("");
  const videoRef = useRef(null);

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

    if (isCameraOn && (isStaticMode || isGestureMode)) {
      interval = setInterval(sendFrameToAPI, 1000);
    } else {
      clearInterval(interval);
      setPrediction("");
    }

    return () => clearInterval(interval);
  }, [isCameraOn, isStaticMode, isGestureMode]);

  const handleCameraToggle = () => {
    setIsCameraOn(!isCameraOn);
    if (isCameraOn) {
      setIsStaticMode(false);
      setIsGestureMode(false);
    }
  };

  return (
    <div>
      <header>
        <div style={{ maxWidth: "1100px", margin: "auto", padding: "0 24px" }}>
          <h1>H! Sign Language Platform</h1>
          <p>Detección en tiempo real de lenguaje de señas con IA</p>
        </div>
      </header>

      <main>
        {/* 📸 El video se muestra en espejo, pero el backend procesa la imagen normal */}
        <VideoInterface isCameraOn={isCameraOn} ref={videoRef} />
        <PredictionBox prediction={prediction} />
        <ControlBar
          isCameraOn={isCameraOn}
          onCameraToggle={handleCameraToggle}
          isStaticMode={isStaticMode}
          onStaticModeToggle={() => setIsStaticMode(!isStaticMode)}
          isGestureMode={isGestureMode}
          onGestureModeToggle={() => setIsGestureMode(!isGestureMode)}
        />
      </main>
    </div>
  );
}
