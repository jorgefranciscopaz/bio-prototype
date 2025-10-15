const ControlBar = ({
  isCameraOn,
  onCameraToggle,
  isStaticMode,
  onStaticModeToggle,
  isGestureMode,
  onGestureModeToggle,
}) => {
  return (
    <div className="control-bar">
      <button
        onClick={onCameraToggle}
        className={isCameraOn ? "btn-camera-on" : "btn-camera-off"}
      >
        {isCameraOn ? "Apagar Cámara" : "Encender Cámara"}
      </button>

      <button
        onClick={onStaticModeToggle}
        disabled={!isCameraOn}
        className={`btn-mode ${isStaticMode ? "btn-mode-active" : ""}`}
      >
        {isStaticMode ? "Modo Estático Activo" : "Activar Estático"}
      </button>

      <button
        onClick={onGestureModeToggle}
        disabled={!isCameraOn}
        className={`btn-mode ${isGestureMode ? "btn-mode-active" : ""}`}
      >
        {isGestureMode ? "Modo Dinámico Activo" : "Activar Dinámico"}
      </button>
    </div>
  );
};

export default ControlBar;
