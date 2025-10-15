import { forwardRef } from "react";
import CameraFeed from "./CameraFeed";

const VideoInterface = forwardRef(({ isCameraOn }, ref) => {
  return (
    <div className="video-box">
      {isCameraOn ? (
        <CameraFeed ref={ref} isActive={true} />
      ) : (
        <div>
          <p style={{ fontSize: "1.2rem", color: "#ccc" }}>📷 Cámara apagada</p>
          <p style={{ color: "#777" }}>
            Enciende la cámara para iniciar la detección
          </p>
        </div>
      )}
    </div>
  );
});

export default VideoInterface;
