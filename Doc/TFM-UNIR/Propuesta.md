# Propuesta

[ X ] Tipo 1. Piloto experimental
[ ] Tipo 2. Desarrollo Software
[ ] Tipo 3. Comparativa de soluciones

## Título provisional
Optimización de modelos de lenguaje compactos para la síntesis de programas orientada a la resolución de ARC-AGI

## Descripción y justificación del trabajo a desarrollar
La hipótesis de escala está llegando a su límite en las tareas de razonamiento abstracto. Los modelos extensos de lenguaje (LLM) como GPT-4o, tienen dificultades para superar el 50% de precisión en ARC-AGI sin un “Scaffolding” o ajuste fino. Esto sugiere una discrepancia fundamental: ARC requiere una deducción algorítmica precisa y la generalización a partir de unos pocos ejemplos, mientras que los transformadores favorecen la memorización probabilística de las estadísticas superficiales. La resolución del problema ARC se considera ampliamente como un indicador para alcanzar la Inteligencia Artificial General (AGI). El trabajo es relevante por que propone una vía eficiente desde el punto de vista de la capacidad de computación, con modelos de lenguaje pequeños (< 3B parámetros) explorando un paradigma determinístico de la sístesis de programa al contrario de los modelos de lenguaje grandes (LLM) con modelos probabilísticos de predicción de próximo token.

## Objetivos e impacto que se espera conseguir con el desarrollo del trabajo

El objetivo principal es desarrollar y explorar las capacidades de la síntesis de programas en entornos restringidos. Se pretende demostrar que en ámbitos regidos por una lógica estricta, la limitación de la capacidad del modelo (mediante SLM) obliga al sistema de aprendizaje a descubrir algoritmos robustos como reglas de actualización recursivas.

Si se limita la capacidad del modelo, siendo más compacto, se reducesu tendecia a sobreajustar o memorizar. Esta restricción fuerza a aprender reglas que se pueden aplicar en puzles en vez de heurísticas frágiles. Se busca el desarrollo de reglas de transición.

## Metodología, tecnologías o técnicas previstas para abordar el desarrollo del trabajo

El trabajo de desarrollará mediante prácticas de desarrollo ágiles dividido en objetivos. Estos objetivos se van a dividir en bloques incrementales y se desarrollarán de manera independiente.

Se ha considera explorar un enfoque basado en arquitectura TRM ó modelos sin ninguna pre-entrenamiento CompressARC. Experimentar las siguientes técnicas:

- Bucles de refinamiento, tendencia actual para verificar la consistencia del modelo.
- "Test-time compute scaling"
- "Test-time trainning"
- "Fine-tunning" ó refinamiento. Ajuste fino en inferencia.
- Aproximación hacia un razonamiento iterativo con soluciones provicionales que se actualizan (TRM).
- "Early-stopping". Parada de iteraciones antes de que deje de aprender adecuadamente.
- "Deep supervision / multi-step improvement"
- "voto/top‑k" para resolver diferentes variantes y elegir las predicciones más consistentes.


Franzen, D., Disselhoff, J., & Hartmann, D. (2024). The LLM ARChitect: Solving ARC-AGI Is A Matter of Perspective.

Chollet, F., Knoop, M., Kamradt, G., & Landers, B. (2024). Arc prize 2024: Technical report. arXiv preprint arXiv:2412.04604.

Chollet, F., Knoop, M., Kamradt, G., & Landers, B. (2026, January 15). ARC Prize 2025: Technical report. arXiv.org. https://arxiv.org/abs/2601.10904

Roye-Azar, A., Vargas-Naranjo, S., Ghai, D., Balamurugan, N., & Amir, R. (2025). Tiny Recursive Models on ARC-AGI-1: Inductive Biases, Identity Conditioning, and Test-Time Compute. arXiv preprint arXiv:2512.11847.

Wang, G., Li, J., Sun, Y., Chen, X., Liu, C., Wu, Y., ... & Yadkori, Y. A. (2025). Hierarchical Reasoning Model. arXiv preprint arXiv:2506.21734.

Cole, J., & Osman, M. (2025). Don't throw the baby out with the bathwater: How and why deep learning for ARC. arXiv preprint arXiv:2506.14276.

Liao, I., & Gu, A. (2025). ARC-AGI without pretraining. arXiv preprint arXiv:2512.06104.