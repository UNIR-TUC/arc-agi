# Diario de desarrollo

## 2026-08-16 - Entender CompressARC: del preprocesamiento al aprendizaje

### Objetivo de la sesion

Entender paso a paso el flujo de debug de `analyze_example.py` para una tarea ARC concreta, especialmente `007bbfb7`: como se representa `self.problem`, como se generan los `logits`, como se calcula la perdida y como aprende el modelo.

### Flujo observado

`analyze_example.py`:

1. Carga la tarea `007bbfb7` del split `training`.
2. `preprocessing.preprocess_tasks(...)` crea un objeto `Task`.
3. Se construye un `ARCCompressor`, un optimizador Adam y un `Logger`.
4. Se ejecutan 2000 pasos de entrenamiento.
5. Cada 50 pasos se guarda un plot con cuatro candidatos: `sample`, `sample average`, `guess 1` y `guess 2`.

### Preprocesamiento de `self.problem`

En `Task._create_problem_tensor`, primero se reserva un tensor NumPy con forma:

```text
(ejemplos, colores, filas, columnas, modo)
```

Para `007bbfb7`:

```text
shape inicial = (6, 5, 9, 9, 2)
```

Interpretacion de los ejes:

- `6`: cinco ejemplos de entrenamiento y un ejemplo de test.
- `5`: colores presentes, incluidos en `task.colors = [0, 2, 4, 6, 7]`.
- `9, 9`: tamano maximo reservado para filas y columnas.
- `2`: `input` y `output`.

La dimension `color` usa una codificacion one-hot. Si una celda tiene el indice interno `2`, su vector puede ser:

```text
[0, 0, 1, 0, 0]
```

La linea:

```python
self.problem = torch.from_numpy(np.argmax(self.problem, axis=1)).to(torch.get_default_device())
```

hace tres transformaciones:

1. `argmax(..., axis=1)` busca la capa de color activa y la convierte en un unico indice.
2. `torch.from_numpy(...)` convierte el array NumPy en tensor PyTorch.
3. `.to(...)` mueve el tensor al dispositivo por defecto, normalmente la GPU en este script.

La forma pasa de:

```text
(6, 5, 9, 9, 2) -> (6, 9, 9, 2)
```

Despues de la conversion, una celda se consulta como:

```python
self.problem[ejemplo, fila, columna, modo]
```

y contiene el indice interno del color correcto. El color ARC original se recupera con `self.colors[indice]`.

Las salidas de test no se rellenan durante la construccion de `self.problem`, porque son desconocidas.

### Vista de debug

Se anadio el metodo `Task.debug_problem_dataframe(...)` en `preprocessing.py`.

Para ver todo el tensor en formato largo desde la Debug Console:

```python
problem_df = self.debug_problem_dataframe(full_problem=True)
problem_df
```

Antes de `argmax`, la tabla conserva:

```text
example, color_index, color, row, column, mode, value
```

Despues de `argmax`, la tabla conserva las coordenadas disponibles del tensor convertido:

```text
example, row, column, mode, color_index, color, value
```

El metodo no sustituye `self.problem`; solo crea una vista temporal.

### Datos concretos de `007bbfb7`

- `n_train = 5`.
- `n_test = 1`.
- Todos los inputs tienen forma `3x3`.
- Todos los outputs de entrenamiento tienen forma `9x9`.
- El output del test se predice con forma `9x9` por las heuristicas de formas.
- Colores ARC: `0, 2, 4, 6, 7`.
- `task.problem` despues de la conversion: `(6, 9, 9, 2)`.

### Que datos se comparan durante el entrenamiento

En cada paso, `model.forward()` produce `logits`, mascaras y costes KL. Despues `train.take_step()` extrae:

```python
logits_slice = logits[example_num, :, :, :, in_out_mode]
problem_slice = task.problem[example_num, :, :, in_out_mode]
```

El entrenamiento compara:

- input y output de los cinco ejemplos de `train`;
- input del ejemplo de `test`;
- nunca el output del ejemplo de `test`.

La exclusion esta controlada por:

```python
if example_num >= task.n_train and in_out_mode == 1:
    continue
```

Nominalmente hay `459` celdas objetivo visibles:

```text
5 * (3*3 + 9*9) + 3*3 = 459
```

Para los inputs de `3x3`, el codigo puede evaluar varios recortes posibles dentro del espacio `9x9`, usando `x_mask` y `y_mask`. Para los outputs de `9x9`, la cuadrícula ocupa todo el espacio.

El output verdadero del test, aunque existe en `training_solutions.json` para poder evaluar el split de entrenamiento, no se usa para calcular la perdida durante el aprendizaje. Se usa despues para comprobar si `guess 1` o `guess 2` acertaron.

### Logits

Los `logits` son puntuaciones sin normalizar que el modelo produce para cada color posible:

```text
logits[ejemplo, color, fila, columna, modo]
```

No son colores ni probabilidades. El color con el logit mayor es la prediccion actual.

En `train.py` se anade una columna de logits cero para representar el color negro:

```python
logits = torch.cat([torch.zeros_like(logits[:, :1, :, :]), logits], dim=1)
```

La `cross_entropy` compara todos los logits de una celda con el indice correcto almacenado en `self.problem`.

### Perdida, reconstruccion y KL

La perdida total es:

```python
loss = total_KL + 10 * reconstruction_error
```

- `reconstruction_error`: mide cuanto se parecen las predicciones a los colores y tamanos conocidos.
- `total_KL`: mide cuanta informacion especifica usa el codigo latente respecto a una distribucion de referencia.

El error de reconstruccion usa `cross_entropy`. Para una celda, la idea es:

```text
error = -log(probabilidad asignada al color correcto)
```

Una probabilidad alta para el color correcto produce poco error; una probabilidad baja produce mucho error.

KL significa divergencia de Kullback-Leibler. Aqui compara el posterior aprendido del latente con un prior normal de referencia:

- KL pequena: el latente se parece al ruido generico y transporta poca informacion especial.
- KL grande: el latente se aleja del prior para codificar detalles de esta tarea.

La arquitectura intenta reconstruir bien las demostraciones usando una representacion relativamente compacta.

### Como aprende el modelo

Cada iteracion sigue este ciclo:

```text
model.forward()
-> logits y mascaras
-> cross_entropy y reconstruction_error
-> total_KL + 10 * reconstruction_error
-> loss.backward()
-> optimizer.step()
-> siguiente iteracion
```

`loss.backward()` calcula el gradiente de la perdida respecto a los pesos y latentes. `optimizer.step()` usa esos gradientes para actualizarlos. El modelo no decide manualmente que parametro corregir; PyTorch calcula la influencia de cada parametro mediante diferenciacion automatica.

El output de test no tiene una respuesta conocida durante este ciclo. El modelo aprende con las partes visibles y aplica el comportamiento aprendido al input de test.

### Plots cada 50 pasos

`visualization.plot_solution(...)` dibuja cuatro columnas:

- `sample`: softmax de los logits actuales del test en ese paso. Es una prediccion continua y puede ser difusa.
- `sample average`: media movil exponencial de logits y mascaras, con `ema_decay = 0.97`. Suaviza el ruido entre pasos.
- `guess 1`: mejor solucion discreta acumulada por el `Logger`.
- `guess 2`: segunda mejor solucion discreta distinta.

Para `guess 1` y `guess 2`, el logger aplica `argmax`, escoge el recorte con las mascaras, convierte los indices internos a colores ARC y acumula puntuaciones por hash mediante `np.logaddexp`.

Por eso `guess 1` y `guess 2` no son simplemente las dos predicciones del paso actual: son los dos candidatos con mayor evidencia acumulada hasta ese momento.

### Puntos de debug utiles

- Breakpoint en la linea de `np.argmax` de `preprocessing.py`: observar la representacion one-hot antes de eliminar el eje `color`.
- Breakpoint en `train.py` justo despues de `model.forward()`: inspeccionar `logits`, `x_mask`, `y_mask` y `KL_amounts`.
- Breakpoint en `train.py` en `cross_entropy(...)`: comparar `logits_crops` con `target_batch`.
- Breakpoint despues de `loss.backward()`: inspeccionar gradientes.
- Breakpoint en `optimizer.step()`: observar el momento en que Adam actualiza los parametros.

### Estado y proxima sesion

- Se entendio el flujo conceptual desde JSON hasta etiquetas, logits, perdida y actualizacion.
- Se creo una vista provisional en DataFrame para inspeccionar `self.problem` completo.
- La ejecucion de `analyze_example.py` termino con codigo de salida no cero en los intentos observados; la causa concreta queda pendiente de investigar.
- Siguiente paso sugerido: detenerse en una iteracion temprana de `train.take_step()` y comparar una celda concreta de `logits_crops` con su valor en `target_batch`, observando tambien `loss`, `loss.grad_fn` y los gradientes de un parametro.
