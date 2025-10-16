import { useEffect, useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Sparkles, Trophy, Zap, X } from "lucide-react";
import { useNavigate } from "react-router-dom";
import monkeyPeek from "@/assets/monkey-peek.png";

const Index = () => {
  const navigate = useNavigate();
  const [modelos, setModelos] = useState<string[]>([]);
  const [modeloSeleccionado, setModeloSeleccionado] = useState<string | null>(null);
  const [dropdownAbierto, setDropdownAbierto] = useState(false);

  // === Estados del modal de entrenamiento ===
  const [modalAbierto, setModalAbierto] = useState(false);
  const [estado, setEstado] = useState<"naming" | "recolectando" | "entrenando" | "finalizado" | null>(null);
  const [nombreModelo, setNombreModelo] = useState("");
  const [letraActual, setLetraActual] = useState("A");
  const [muestras, setMuestras] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);

  // === Cargar modelos desde Flask ===
  useEffect(() => {
    fetch("http://localhost:5000/modelos_personalizados")
      .then((res) => res.json())
      .then((data) => {
        // ✅ Mostrar solo los archivos que terminan en "_modelo.pkl"
        const modelosFiltrados = (data.modelos || [])
          .filter((m: string) => m.endsWith("_modelo.pkl"))
          .map((m: string) => m.replace("_modelo.pkl", "")); // limpia nombre visible
        setModelos(modelosFiltrados);
      })
      .catch((err) => console.error("Error al cargar modelos:", err));
  }, []);

  // === Activar cámara ===
  const iniciarCamara = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error("❌ Error al acceder a la cámara:", error);
    }
  };

  // === Iniciar flujo de creación ===
  const abrirModalEntrenamiento = () => {
    setModalAbierto(true);
    setEstado("naming");
  };

  // === Enviar nombre de modelo al backend ===
  const confirmarNombreModelo = async () => {
    if (!nombreModelo.trim()) {
      alert("Por favor ingresa un nombre para tu modelo.");
      return;
    }

    try {
      await fetch("http://localhost:5000/definir_modelo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nombre_modelo: nombreModelo }),
      });

      setEstado("recolectando");
      await iniciarCamara();

      alert("📸 La recolección está corriendo en Python.\nCuando termine, continúa con el entrenamiento.");

      setEstado("entrenando");
      const res = await fetch("http://localhost:5000/entrenar_modelo_personalizado", { method: "POST" });
      const data = await res.json();

      console.log("✅ Modelo entrenado:", data);
      setEstado("finalizado");

      // 🔄 Actualizar lista de modelos al finalizar
      const modelosActualizados = await fetch("http://localhost:5000/modelos_personalizados").then((r) => r.json());
      const modelosFiltrados = (modelosActualizados.modelos || [])
        .filter((m: string) => m.endsWith("_modelo.pkl"))
        .map((m: string) => m.replace("_modelo.pkl", ""));
      setModelos(modelosFiltrados);
    } catch (error) {
      console.error("❌ Error durante la creación del modelo:", error);
    }
  };

  // === Cerrar modal ===
  const cerrarModal = () => {
    setModalAbierto(false);
    setEstado(null);
    setNombreModelo("");
    setMuestras(0);
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((t) => t.stop());
    }
  };

  // === Seleccionar modelo existente ===
  const handleSeleccionarModelo = async (nombre: string) => {
    const modeloArchivo = `${nombre}_modelo.pkl`; // se guarda el archivo real
    setModeloSeleccionado(nombre);
    setDropdownAbierto(false);
    localStorage.setItem("modeloSeleccionado", modeloArchivo);

    try {
      const res = await fetch(`http://localhost:5000/usar_modelo/${modeloArchivo}`);
      const data = await res.json();
      alert(`✅ Modelo "${nombre}" cargado correctamente`);
      console.log("Modelo activado:", data);
    } catch (error) {
      console.error("❌ Error al activar modelo:", error);
    }
  };

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Fondos decorativos */}
      <div className="absolute inset-0 bg-gradient-to-br from-accent/5 via-transparent to-primary/5" />
      <div className="absolute top-32 right-32 w-80 h-80 bg-accent/10 rounded-full blur-[120px]" />
      <div className="absolute bottom-32 right-1/3 w-96 h-96 bg-primary/5 rounded-full blur-[100px]" />

      {/* Monito decorativo */}
      <div className="hidden lg:block fixed left-0 bottom-0 top-0 z-20 pointer-events-none w-1/3 max-w-md">
        <img
          src={monkeyPeek}
          alt="Monito amigable saludando"
          className="h-full w-full animate-fade-in"
          style={{ objectFit: "cover", objectPosition: "left center" }}
        />
      </div>

      {/* Main content */}
      <div className="relative z-10 min-h-screen">
        <div className="container mx-auto px-4 py-8 lg:py-16 min-h-screen flex items-center">
          <div className="w-full lg:ml-auto lg:w-2/3 lg:pl-8">
            <div className="max-w-3xl mx-auto lg:mx-0 space-y-8 lg:space-y-10">
              {/* Header */}
              <div className="space-y-4 text-center lg:text-left animate-fade-in">
                <h1 className="text-4xl md:text-5xl lg:text-6xl xl:text-7xl font-bold leading-tight text-foreground">
                  ¡Aprende lenguaje de señas!
                </h1>
                <p className="text-lg md:text-xl lg:text-2xl text-muted-foreground">
                  Elige tu camino de aprendizaje
                </p>
              </div>

              {/* Cards */}
              <div className="space-y-4 lg:space-y-5 animate-fade-in" style={{ animationDelay: "0.1s" }}>
                {/* Card 1 - Personalizado */}
                <Card className="group w-full opacity-80 pointer-events-none">
                  <CardHeader className="p-4 md:p-6 relative">
                    <div className="flex items-start gap-3 md:gap-5">
                      <div className="p-3 md:p-4 rounded-2xl bg-primary/10 shrink-0">
                        <Sparkles className="w-5 h-5 md:w-7 md:h-7 text-primary" strokeWidth={2.5} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-bold text-accent uppercase tracking-widest mb-1 md:mb-2">
                          Personalizado
                        </div>
                        <CardTitle className="text-xl md:text-2xl lg:text-3xl mb-1 md:mb-2 text-primary/60">
                          Crear modelo a medida
                        </CardTitle>
                        <CardDescription className="text-sm md:text-base text-muted-foreground">
                          🧪 En fase de desarrollo
                        </CardDescription>
                      </div>
                    </div>
                    <div className="absolute top-3 right-3 bg-yellow-500 text-black text-xs font-bold px-3 py-1 rounded-full shadow-md">
                      En desarrollo
                    </div>
                  </CardHeader>
                </Card>

                {/* Card 2 - Modelos */}
                <Card className="group w-full p-4 md:p-6 relative">
                  <div className="flex items-start gap-3 md:gap-5">
                    <div className="p-3 md:p-4 rounded-2xl bg-secondary/20 shrink-0">
                      <Trophy className="w-5 h-5 md:w-7 md:h-7 text-primary" strokeWidth={2.5} />
                    </div>
                    <div className="flex-1">
                      <div className="text-xs font-bold text-primary uppercase tracking-widest mb-1 md:mb-2">
                        Modelos
                      </div>
                      <CardTitle className="text-xl md:text-2xl lg:text-3xl mb-2 text-primary">
                        Elegir modelo existente
                      </CardTitle>
                      <CardDescription className="text-sm md:text-base mb-3">
                        Haz clic para mostrar los modelos disponibles
                      </CardDescription>

                      <Button
                        onClick={() => setDropdownAbierto(!dropdownAbierto)}
                        className="bg-primary/10 hover:bg-primary/20 text-primary rounded-lg"
                      >
                        {modeloSeleccionado
                          ? `Modelo seleccionado: ${modeloSeleccionado}`
                          : "Elegir modelo existente"}
                      </Button>

                      {dropdownAbierto && (
                        <div className="absolute mt-2 bg-gray-900/90 border border-gray-700 rounded-xl shadow-lg w-full max-h-64 overflow-y-auto z-50 p-2">
                          {modelos.length > 0 ? (
                            modelos.map((modelo, idx) => (
                              <div
                                key={idx}
                                onClick={() => handleSeleccionarModelo(modelo)}
                                className="px-4 py-2 rounded-lg text-gray-100 hover:bg-primary/30 cursor-pointer"
                              >
                                {modelo}
                              </div>
                            ))
                          ) : (
                            <p className="text-sm text-muted-foreground px-4 py-2">
                              No hay modelos personalizados aún.
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </Card>

                {/* Card 3 - En Vivo */}
                <Card
                  onClick={() => {
                    if (!modeloSeleccionado) {
                      alert("Por favor selecciona un modelo antes de continuar.");
                      return;
                    }
                    navigate("/realtime", { state: { modeloSeleccionado } });
                  }}
                  className="group cursor-pointer w-full border-accent/40 bg-gradient-to-br from-accent/10 via-accent/5 to-transparent hover:border-accent/60 shadow-lg"
                >
                  <CardHeader className="p-4 md:p-6">
                    <div className="flex items-start gap-3 md:gap-5">
                      <div className="p-3 md:p-4 rounded-2xl bg-accent/30 group-hover:bg-accent/40 transition-all duration-300 group-hover:scale-110 shadow-md shrink-0">
                        <Zap className="w-5 h-5 md:w-7 md:h-7 text-accent-foreground" strokeWidth={2.5} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-bold text-accent uppercase tracking-widest mb-1 md:mb-2">
                          En Vivo
                        </div>
                        <CardTitle className="text-xl md:text-2xl lg:text-3xl mb-1 md:mb-2 group-hover:text-accent transition-colors">
                          Detección en tiempo real
                        </CardTitle>
                        <CardDescription className="text-sm md:text-base">
                          Practica con tu cámara ahora mismo
                        </CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                </Card>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Modal */}
      {modalAbierto && (
        <div className="fixed inset-0 bg-black/60 flex justify-center items-center z-50">
          <div className="bg-gray-900 text-white rounded-2xl p-6 w-[90%] max-w-lg relative border border-gray-700">
            <button className="absolute top-4 right-4 text-gray-400 hover:text-white" onClick={cerrarModal}>
              <X className="w-5 h-5" />
            </button>

            {estado === "naming" && (
              <div className="text-center space-y-4">
                <h2 className="text-xl font-semibold">🧩 Nombra tu modelo personalizado</h2>
                <input
                  type="text"
                  value={nombreModelo}
                  onChange={(e) => setNombreModelo(e.target.value)}
                  placeholder="Ejemplo: modelo_jorge"
                  className="w-full px-4 py-2 rounded-lg text-black"
                />
                <Button onClick={confirmarNombreModelo} className="w-full mt-2">
                  Comenzar entrenamiento
                </Button>
              </div>
            )}

            {estado === "recolectando" && (
              <div className="space-y-4 text-center">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  className="rounded-lg w-full h-64 object-cover border border-gray-700"
                  style={{ transform: "scaleX(1)" }}
                />
                <p className="text-lg">
                  Letra actual: <span className="font-bold text-accent">{letraActual}</span>
                </p>
                <p>Muestras capturadas: {muestras}/100</p>
              </div>
            )}

            {estado === "entrenando" && (
              <div className="flex flex-col items-center justify-center p-10">
                <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-accent" />
                <p className="mt-4 text-accent">Entrenando modelo personalizado...</p>
              </div>
            )}

            {estado === "finalizado" && (
              <div className="flex flex-col items-center p-6 space-y-3">
                <p className="text-lg text-green-400 font-semibold">✅ Entrenamiento completado con éxito</p>
                <Button onClick={cerrarModal}>Cerrar</Button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Index;
