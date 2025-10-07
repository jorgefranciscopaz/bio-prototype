export default function ToggleAIButton({ isActive, onToggle }) {
  return (
    <button
      onClick={onToggle}
      className={`px-6 py-3 rounded-xl font-semibold transition-all ${
        isActive ? "bg-red-600 hover:bg-red-700" : "bg-green-600 hover:bg-green-700"
      } text-white`}
    >
      {isActive ? "Desactivar IA" : "Activar IA"}
    </button>
  );
}
