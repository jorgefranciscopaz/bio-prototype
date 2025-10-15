export default function PredictionBox({ prediction }) {
  return (
    <div
      style={{
        marginTop: "16px",
        textAlign: "center",
        fontSize: "1.5rem",
        fontWeight: "bold",
        color: "#ffb347",
      }}
    >
      {prediction || "Sin predicción"}
    </div>
  );
}
