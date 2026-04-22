
1. Contexto


Se va a realizar un proyecto de fin de master. En este proyecto se va a abordar la resolución de ARC-AGI-1 desde un punto de vista optimizado a recursos de computación limitados.

Titulo provisional del proyecto: Optimización de modelos de lenguaje compactos para la síntesis de programas orientada a la resolución de ARC-AGI-1

1.1. Descripción y justificación del proyecto

La hipótesis de escala está llegando a su límite en las tareas de razonamiento abstracto. Los modelos extensos de lenguaje (LLM) como GPT-4o, tienen dificultades para superar el 50% de precisión en ARC-AGI sin un “Scaffolding” o ajuste fino. Esto sugiere una discrepancia fundamental: ARC requiere una deducción algorítmica precisa y la generalización a partir de unos pocos ejemplos, mientras que los transformadores favorecen la memorización probabilística de las estadísticas superficiales. La resolución del problema ARC se considera ampliamente como un indicador para alcanzar la Inteligencia Artificial General (AGI). El trabajo es relevante por que propone una vía eficiente desde el punto de vista de la capacidad de computación, con modelos de lenguaje pequeños (< 3B parámetros) explorando un paradigma determinístico de la sístesis de programa al contrario de los modelos de lenguaje grandes (LLM) con modelos probabilísticos de predicción de próximo token.

1.2. Objetivos hipotéticos

El objetivo principal es desarrollar y explorar las capacidades de la síntesis de programas en entornos restringidos. Se pretende demostrar que en ámbitos regidos por una lógica estricta, la limitación de la capacidad del modelo (mediante SLM) obliga al sistema de aprendizaje a descubrir algoritmos robustos como reglas de actualización recursivas.

Si se limita la capacidad del modelo, siendo más compacto, se reducir tendencia a sobre-ajustar o memorizar. Esta restricción fuerza a aprender reglas que se pueden aplicar en puzzles en vez de heurísticas frágiles. Se busca el desarrollo de reglas de transición.

1.3. Primeros enfoques para solucionarlo

El trabajo de desarrollará mediante prácticas de desarrollo ágiles dividido en objetivos. Estos objetivos se van a dividir en bloques incrementales y se desarrollarán de manera independiente.

Se ha considera explorar un enfoque basado en arquitectura TRM ó modelos sin ninguna pre-entrenamiento CompressARC. Experimentar las siguientes técnicas:


    Bucles de refinamiento, tendencia actual para verificar la consistencia del modelo.

    "Test-time compute scaling"

    "Test-time trainning"

    "Fine-tunning" ó refinamiento. Ajuste fino en inferencia.

    Aproximación hacia un razonamiento iterativo con soluciones provicionales que se actualizan (TRM).

    "Early-stopping". Parada de iteraciones antes de que deje de aprender adecuadamente.

    "Deep supervision / multi-step improvement"

    "voto/top‑k" para resolver diferentes variantes y elegir las predicciones más consistentes.


1.4. Capacidad de cómputo disponible

Sistema operativo: Ubuntu 24.04
Procesador: Intel i9 12900k
Memoria RAM: 96 GB DDR5
Targeta gráfica: AMD Radeon 9070 XT 16GB
ROCM: 7.2.2

1.5. Objetivo principal

El objetivo principal es el diseño y desarrollo de un modelo compacto para maximizar la puntuación en ARC-AGI-1 con el nivel de puntuación disponible. La idea es que al final del proyecto se realice una comparación entre la puntuación y el cómputo utilizado. Se busca una comparativa en el que se premie la eficiencia. Con el cómputo disponible no es posible alcanzar la máxima puntuación en ARC-AGI-1, el objetivo lógico debe ser una lucha llevada a la eficiencia.