import { forwardRef, useEffect } from "react";

const CameraFeed = forwardRef(({ isActive }, ref) => {
  useEffect(() => {
    if (!isActive) return;
    const video = ref.current;
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
      } catch (error) {
        console.error("Error accediendo a la cámara:", error);
      }
    };
    startCamera();

    return () => {
      if (video.srcObject) video.srcObject.getTracks().forEach((t) => t.stop());
    };
  }, [isActive]);

  return <video ref={ref} autoPlay playsInline width="640" height="480" />;
});

export default CameraFeed;
