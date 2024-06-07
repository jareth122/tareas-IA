
# Reporte del Juego con Phaser.js y Red Neuronal

## Introducción
 Es un juego desarrollado con la librería Phaser.js, que se ejecuta en un lienzo HTML5. Este juego presenta un entorno donde un jugador debe evitar colisiones con naves y balas controladas por una red neuronal. A lo largo del código, se definen y manejan varias entidades del juego, como el jugador, naves, balas, y controles de juego. También se incluye la integración de una red neuronal para el modo de juego automático, lo que añade complejidad y dinamismo al comportamiento del jugador.

## Desarrollo
El desarrollo del juego se divide en varias etapas, comenzando con la configuración y carga de recursos, la creación de entidades del juego, y finalmente, la lógica de juego en tiempo real. A continuación se describen estos componentes en detalle:

### Dimensiones y Variables del Juego
- Se definen las dimensiones del lienzo del juego y se declaran múltiples variables para los objetos del juego como el jugador, fondo, balas y naves.
- Variables para controlar la física de las balas y el estado del jugador, así como las teclas de control y el menú de pausa.

### Configuración de Phaser
- Se inicia el juego utilizando Phaser, configurando las dimensiones y los métodos de ciclo de vida del juego (preload, create, update, render).
- En preload, se cargan los recursos gráficos como imágenes y sprites.

### Creación de Entidades del Juego
- En create, se configura la física del juego, el fondo, las naves, balas y el jugador.
- Se habilita la física para estos objetos para manejar colisiones y movimientos.
- Se configuran las animaciones del jugador y se añaden eventos de control para las teclas de salto, avanzar y retroceder.

### Red Neuronal
- Se inicializa una red neuronal utilizando la librería Synaptic, con capas definidas para entrenar y predecir el comportamiento del jugador en modo automático.
- Funciones para entrenar la red neuronal (`entrenarRedNeuronal`), procesar datos de entrada (`datosDeEntrenamiento`), y entrenar específicamente para los saltos (`entrenamientoSalto`).

### Control y Lógica del Juego
- Funciones para manejar el estado del juego (`actualizar`), incluyendo el movimiento del jugador y la actualización de las balas.
- Mecanismos para controlar manualmente el jugador mediante las teclas y automáticamente mediante la red neuronal.
- Funciones para pausar el juego y manejar la pausa, incluyendo la lógica para reiniciar las variables del juego y el entrenamiento de la red neuronal.

### Manejo de Colisiones y Velocidades
- Funciones para manejar colisiones (`colisionH`), disparar balas con velocidades aleatorias (`dispararBala1`, `dispararBala2`, `dispararBala3`), y resetear las posiciones de las balas y el jugador.
- Función `velocidadAleatoria` para generar velocidades dentro de un rango especificado.

## Conclusión
El código presenta un completo juego de plataforma en 2D utilizando Phaser.js, donde el jugador debe esquivar balas y naves controladas por una red neuronal. Se han implementado funciones tanto para el control manual como automático del jugador, añadiendo complejidad con la integración de una red neuronal para el aprendizaje y predicción del movimiento del jugador.

A través de una buena organización del código y la correcta implementación de las funciones de control y actualización, el juego demuestra ser un interesante ejemplo de la combinación de técnicas de desarrollo de juegos y aprendizaje automático. Las futuras mejoras podrían incluir una mayor optimización de la red neuronal, ampliación de los niveles de juego y una mejor interfaz de usuario.
