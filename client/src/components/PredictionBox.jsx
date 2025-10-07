export default function PredictionBox({ prediction }) {
  return (
    <div className="mt-4 p-4 bg-white/10 text-white rounded-xl shadow-lg text-center">
      <h3 className="text-lg font-bold">Predicción actual:</h3>
      <p className="text-2xl mt-2">{prediction || "Esperando gesto..."}</p>
    </div>
  );
}
