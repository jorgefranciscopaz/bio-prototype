import { forwardRef, useEffect } from "react";

const CameraFeed = forwardRef(({ isActive }, ref) => {
  useEffect(() => {
    let stream;

    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (ref.current) {
          ref.current.srcObject = stream;
          console.log("✅ Cámara iniciada correctamente");
        }
      } catch (err) {
        console.error("❌ Error al acceder a la cámara:", err);
      }
    };

    if (isActive) {
      startCamera();
    } else {
      if (ref.current && ref.current.srcObject) {
        ref.current.srcObject.getTracks().forEach(track => track.stop());
        ref.current.srcObject = null;
      }
    }

    return () => {
      if (stream) stream.getTracks().forEach(track => track.stop());
    };
  }, [isActive, ref]);

  return (
    <video
      ref={ref}
      autoPlay
      playsInline
      muted
      className="rounded-2xl shadow-lg w-[480px] h-[360px] border border-gray-700 object-cover"
    />
  );
});

export default CameraFeed;
