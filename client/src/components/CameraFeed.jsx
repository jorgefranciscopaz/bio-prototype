import { useEffect, useRef } from "react";

export default function CameraFeed({ isActive }) {
  const videoRef = useRef(null);

  useEffect(() => {
    let stream;
    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch (err) {
        console.error("Error al acceder a la cámara:", err);
      }
    };

    if (isActive) startCamera();
    else if (videoRef.current?.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }

    return () => {
      if (stream) stream.getTracks().forEach(track => track.stop());
    };
  }, [isActive]);

  return (
    <div className="w-full flex justify-center">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="rounded-2xl shadow-lg w-96 border border-gray-700"
      />
    </div>
  );
}
