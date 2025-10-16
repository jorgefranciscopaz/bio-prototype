import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Sparkles, Trophy, Zap } from "lucide-react";
import monkeyPeek from "@/assets/monkey-peek.png";

const Index = () => {
  const [modelos, setModelos] = useState<string[]>([]);
  const [modeloSeleccionado, setModeloSeleccionado] = useState<string | null>(null);
  const [dropdownAbierto, setDropdownAbierto] = useState(false);

  // === Cargar modelos desde Flask ===
  useEffect(() => {
    fetch("http://localhost:5000/modelos_personalizados")
      .then((res) => res.json())
      .then((data) => {
        // Solo modelos .pkl válidos (ya filtrados en backend, pero reforzamos aquí)
        const modelosFiltrados = (data.modelos || []).filter((m: string) => m.endsWith("_modelo.pkl"));
        setModelos(modelosFiltrados);
      })
      .catch((err) => console.error("Error al cargar modelos:", err));
  }, []);

  // === Seleccionar modelo ===
  const handleSeleccionarModelo = async (nombre: string) => {
    setModeloSeleccionado(nombre);
    setDropdownAbierto(false); // cerrar dropdown

    try {
      const res = await fetch(`http://localhost:5000/usar_modelo/${nombre}`);
      const data = await res.json();
      console.log("✅ Modelo activado:", data);
      alert(`✅ Modelo "${nombre}" cargado correctamente`);
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
                <Card className="group cursor-pointer w-full">
                  <CardHeader className="p-4 md:p-6">
                    <div className="flex items-start gap-3 md:gap-5">
                      <div className="p-3 md:p-4 rounded-2xl bg-primary/10 group-hover:bg-primary/15 transition-all duration-300 group-hover:scale-110 shrink-0">
                        <Sparkles className="w-5 h-5 md:w-7 md:h-7 text-primary" strokeWidth={2.5} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-bold text-accent uppercase tracking-widest mb-1 md:mb-2">
                          Personalizado
                        </div>
                        <CardTitle className="text-xl md:text-2xl lg:text-3xl mb-1 md:mb-2 group-hover:text-primary transition-colors">
                          Crear modelo a medida
                        </CardTitle>
                        <CardDescription className="text-sm md:text-base">
                          Entrena tu propio modelo de reconocimiento
                        </CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                </Card>

                {/* Card 2 - Modelos */}
                <Card className="group w-full p-4 md:p-6 relative">
                  <div className="flex items-start gap-3 md:gap-5">
                    <div className="p-3 md:p-4 rounded-2xl bg-secondary/20 group-hover:bg-secondary/30 transition-all duration-300 shrink-0">
                      <Trophy className="w-5 h-5 md:w-7 md:h-7 text-primary" strokeWidth={2.5} />
                    </div>

                    <div className="flex-1">
                      <div className="text-xs font-bold text-primary uppercase tracking-widest mb-1 md:mb-2">
                        Modelos
                      </div>
                      <CardTitle className="text-xl md:text-2xl lg:text-3xl mb-2 group-hover:text-primary transition-colors">
                        Elegir modelo existente
                      </CardTitle>
                      <CardDescription className="text-sm md:text-base mb-3">
                        Haz clic para mostrar los modelos disponibles
                      </CardDescription>

                      {/* Botón Dropdown */}
                      <Button
                        onClick={() => setDropdownAbierto(!dropdownAbierto)}
                        className="bg-primary/10 hover:bg-primary/20 text-primary rounded-lg"
                      >
                        {modeloSeleccionado
                          ? `Modelo seleccionado: ${modeloSeleccionado.replace("_modelo.pkl", "")}`
                          : "Elegir modelo existente"}
                      </Button>

                      {/* Dropdown Manual */}
                      {dropdownAbierto && (
                        <div className="absolute mt-2 bg-gray-900/90 border border-gray-700 rounded-xl shadow-lg w-full max-h-64 overflow-y-auto z-50 p-2">
                          {modelos.length > 0 ? (
                            modelos.map((modelo, idx) => (
                              <div
                                key={idx}
                                onClick={() => handleSeleccionarModelo(modelo)}
                                className="px-4 py-2 rounded-lg text-gray-100 hover:bg-primary/30 cursor-pointer"
                              >
                                {modelo.replace("_modelo.pkl", "")}
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
                <Card className="group cursor-pointer w-full border-accent/40 bg-gradient-to-br from-accent/10 via-accent/5 to-transparent hover:border-accent/60 shadow-lg">
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
    </div>
  );
};

export default Index;
