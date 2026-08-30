# Diario de desarrollo

## 2026-08-26 - Entender el multitensor: mascaras, dimensiones y analogia con Pandas

### Objetivo de la sesion

Entender desde cero que representa el multitensor de CompressARC, por que usa
cinco indicadores binarios, como esos indicadores se combinan con los tamanos
reales de una tarea y como visualizar sus hojas mediante estructuras familiares
de Pandas.

La confusion inicial procedia de interpretar una expresion como:

```text
[1, 0, 1, 1, 0]
```

como si fuera la forma de un tensor o una codificacion one-hot de sus datos. No
es ninguna de las dos cosas. Es una **mascara de presencia de ejes**.

### Idea principal

CompressARC reconoce cinco tipos de indices:

```text
[examples, colors, directions, x, y]
```

Para cada tipo de indice decide si una representacion depende o no de el:

```text
0 = el eje no aparece en esta hoja
1 = el eje aparece en esta hoja
```

Por ejemplo:

```text
[1, 0, 1, 1, 0]
```

significa que la hoja distingue `examples`, `directions` y `x`, pero no
distingue `colors` ni `y`.

Los bits no son datos de la tarea y tampoco indican tamanos. El primer `1` no
significa que exista un solo ejemplo: significa que se debe incluir el eje de
ejemplos, sea cual sea su longitud real.

Una forma compacta de resumir la estructura es:

```python
MultiTensor = dict[mascara_binaria, tensor]
```

La implementacion utiliza una lista anidada de profundidad cinco en lugar de un
diccionario, pero la interpretacion anterior es equivalente y mas sencilla de
visualizar.

### Por que hay cinco indicadores binarios

Cada indicador corresponde a una pregunta independiente:

| Posicion | Eje | Pregunta |
| --- | --- | --- |
| 0 | `examples` | La informacion cambia entre ejemplos? |
| 1 | `colors` | La informacion cambia entre colores? |
| 2 | `directions` | La informacion cambia entre direcciones? |
| 3 | `x` | La informacion cambia entre filas? |
| 4 | `y` | La informacion cambia entre columnas? |

Cada respuesta solo puede ser si o no. Por eso cada posicion es binaria. Con
cinco decisiones existen:

$$
2^5 = 32
$$

combinaciones posibles. Cada combinacion equivale a elegir un subconjunto de
los cinco ejes:

```text
{colors}                         -> [0, 1, 0, 0, 0]
{examples, colors}               -> [1, 1, 0, 0, 0]
{examples, x, y}                 -> [1, 0, 0, 1, 1]
{examples, colors, x, y}         -> [1, 1, 0, 1, 1]
{examples, colors, dir, x, y}    -> [1, 1, 1, 1, 1]
```

El objetivo es disponer simultaneamente de representaciones con distintos
niveles de detalle. Una propiedad general de un color no necesita repetirse
para cada pixel. Una propiedad espacial, en cambio, si necesita los ejes `x`
e `y`.

### Mascara, tamanos y datos son tres cosas distintas

Para una tarea hipotetica se pueden tener los tamanos reales:

```python
dim_lengths = [
    2,  # examples
    2,  # colors no negros
    8,  # directions
    3,  # x: filas
    3,  # y: columnas
]
```

La mascara selecciona algunos de esos tamanos. El metodo real
`MultiTensorSystem.shape()` hace conceptualmente esto:

```python
def shape(dims, dim_lengths, channel_size):
    result = []

    for is_present, real_length in zip(dims, dim_lengths):
        if is_present:
            result.append(real_length)

    result.append(channel_size)
    return result
```

Ejemplos:

```python
shape([0, 1, 0, 0, 0], dim_lengths, 16)
# [2, 16]
# [color, channel]

shape([1, 1, 0, 0, 0], dim_lengths, 16)
# [2, 2, 16]
# [example, color, channel]

shape([1, 0, 1, 1, 0], dim_lengths, 8)
# [2, 8, 3, 8]
# [example, direction, x, channel]

shape([1, 1, 1, 1, 1], dim_lengths, 8)
# [2, 2, 8, 3, 3, 8]
# [example, color, direction, x, y, channel]
```

Cuando un bit vale cero, no aparece un eje de longitud cero ni de longitud uno:
el eje se omite por completo.

El ultimo eje, `channel`, no forma parte de la mascara. Contiene rasgos internos
aprendidos por la red. En el residual stream actual tiene longitud 16 para las
hojas sin `directions` y longitud 8 para las hojas con `directions`. El latente
inicial utiliza 4 canales antes de proyectarse al residual stream.

### Como leer un tensor completo

La forma:

```text
[2, 2, 8, 3, 3, 8]
```

se interpreta como:

```text
[example, color, direction, x, y, channel]
```

Contiene:

$$
2 \times 2 \times 8 \times 3 \times 3 \times 8 = 2304
$$

valores numericos. Una consulta como:

```python
features = tensor[0, 1, 3, 2, 0, :]
```

selecciona todos los canales del primer ejemplo, segundo color, cuarta
direccion, tercera fila y primera columna. El resultado es un vector de ocho
rasgos aprendidos, no un color ni una direccion:

```text
[0.31, -0.52, 1.17, 0.04, -0.81, 0.22, 0.95, -0.13]
```

### Analogias con Pandas

La equivalencia mental mas util es:

```text
Hoja tensorial  ~= DataFrame con MultiIndex
Multitensor     ~= diccionario de 18 DataFrames
Mascara         ~= columnas elegidas para formar el MultiIndex
Canales         ~= columnas de caracteristicas numericas
```

Una cuadrícula puede ponerse primero en formato largo:

```python
import pandas as pd

grids = [
    [[0, 2, 0], [0, 0, 1], [2, 0, 0]],
    [[0, 1, 0], [2, 0, 0], [0, 0, 2]],
]

rows = []
for example, grid in enumerate(grids):
    for x, row in enumerate(grid):
        for y, color in enumerate(row):
            rows.append({
                "example": example,
                "x": x,
                "y": y,
                "color": color,
            })

pixels = pd.DataFrame(rows).set_index(["example", "x", "y"])
```

El resultado conceptual es:

```text
                 color
example x y
0       0 0          0
          1          2
          2          0
        1 0          0
...
1       2 2          2
```

En un tensor se escribiria `tensor[example, x, y]`; con el `MultiIndex` de
Pandas se escribiria `pixels.loc[(example, x, y)]`.

### La mascara como eleccion de un MultiIndex

La mascara puede traducirse a nombres de columnas:

```python
AXES = ["example", "color", "direction", "x", "y"]

def selected_axes(mask):
    return [
        axis
        for axis, included in zip(AXES, mask)
        if included
    ]

selected_axes([1, 0, 1, 1, 0])
# ["example", "direction", "x"]
```

En terminos familiares de Pandas:

| Mascara | Vista equivalente |
| --- | --- |
| `[0, 1, 0, 0, 0]` | `groupby(["color"])` |
| `[1, 1, 0, 0, 0]` | `groupby(["example", "color"])` |
| `[1, 0, 0, 1, 0]` | `groupby(["example", "x"])` |
| `[1, 0, 0, 1, 1]` | `groupby(["example", "x", "y"])` |
| `[1, 1, 0, 1, 1]` | `groupby(["example", "color", "x", "y"])` |
| `[1, 1, 1, 1, 1]` | `groupby` por todos los ejes |

Esta es una analogia estructural. Las hojas no son resultados estadisticos
precalculados por un `groupby`: contienen representaciones neuronales que se
inicializan, transforman y aprenden durante el entrenamiento.

### Construir una hoja como DataFrame

Para la mascara `[1, 0, 1, 1, 0]`:

```python
import numpy as np
import pandas as pd

sizes = {
    "example": 2,
    "color": 2,
    "direction": 8,
    "x": 3,
    "y": 3,
}

mask = [1, 0, 1, 1, 0]
axes = selected_axes(mask)

index = pd.MultiIndex.from_product(
    [range(sizes[axis]) for axis in axes],
    names=axes,
)

leaf = pd.DataFrame(
    np.random.randn(len(index), 8),
    index=index,
    columns=[f"channel_{i}" for i in range(8)],
)
```

El resultado tiene esta estructura:

```text
                     channel_0  channel_1  ...  channel_7
example direction x
0       0         0       0.31      -0.22  ...       0.14
                  1      -0.18       0.47  ...       0.83
                  2       1.03      -0.38  ...      -0.11
        1         0       0.61       0.25  ...       0.44
...
1       7         2      -0.53       0.73  ...       0.19
```

En Pandas su forma es `(48, 8)` porque aplana los indices compuestos en filas:

$$
2 \times 8 \times 3 = 48
$$

En PyTorch la misma informacion conserva sus ejes separados y tiene forma
`[2, 8, 3, 8]`.

### El multitensor completo expresado con Pandas

Conceptualmente se podria construir asi:

```python
multitensor = {}

for mask in valid_masks:
    axes = selected_axes(mask)
    channels = 8 if mask[2] else 16
    index = pd.MultiIndex.from_product(
        [range(sizes[axis]) for axis in axes],
        names=axes,
    )

    multitensor[tuple(mask)] = pd.DataFrame(
        np.random.randn(len(index), channels),
        index=index,
        columns=[f"channel_{i}" for i in range(channels)],
    )
```

Consultas equivalentes:

```python
multitensor[(0, 1, 0, 0, 0)]
# DataFrame indexado por color

multitensor[(1, 0, 0, 1, 1)]
# DataFrame indexado por example, x, y

multitensor[(1, 1, 1, 1, 1)]
# DataFrame indexado por todos los ejes
```

### Las 32 combinaciones y sus reglas de validez

El codigo conserva una combinacion si cumple estas reglas:

1. Si incluye `x` o `y`, tambien debe incluir `examples`, porque una fila o una
   columna pertenecen a una cuadrícula concreta.
2. Debe incluir al menos uno entre `colors`, `directions`, `x` e `y`. Una hoja
   que solo distinguiera ejemplos se descarta por decision de diseno.

Para los tamanos `E=2`, `C=2`, `D=8`, `X=3`, `Y=3`, las combinaciones son:

| Mascara `ECDXY` | Estado | Forma residual o motivo |
| --- | --- | --- |
| `00000` | Invalida | No contiene ningun eje informativo |
| `00001` | Invalida | `Y` requiere `E` |
| `00010` | Invalida | `X` requiere `E` |
| `00011` | Invalida | `X/Y` requieren `E` |
| `00100` | Valida | `[D, canal] = [8, 8]` |
| `00101` | Invalida | `Y` requiere `E` |
| `00110` | Invalida | `X` requiere `E` |
| `00111` | Invalida | `X/Y` requieren `E` |
| `01000` | Valida | `[C, canal] = [2, 16]` |
| `01001` | Invalida | `Y` requiere `E` |
| `01010` | Invalida | `X` requiere `E` |
| `01011` | Invalida | `X/Y` requieren `E` |
| `01100` | Valida | `[C, D, canal] = [2, 8, 8]` |
| `01101` | Invalida | `Y` requiere `E` |
| `01110` | Invalida | `X` requiere `E` |
| `01111` | Invalida | `X/Y` requieren `E` |
| `10000` | Invalida | Solo contiene `E` |
| `10001` | Valida | `[E, Y, canal] = [2, 3, 16]` |
| `10010` | Valida | `[E, X, canal] = [2, 3, 16]` |
| `10011` | Valida | `[E, X, Y, canal] = [2, 3, 3, 16]` |
| `10100` | Valida | `[E, D, canal] = [2, 8, 8]` |
| `10101` | Valida | `[E, D, Y, canal] = [2, 8, 3, 8]` |
| `10110` | Valida | `[E, D, X, canal] = [2, 8, 3, 8]` |
| `10111` | Valida | `[E, D, X, Y, canal] = [2, 8, 3, 3, 8]` |
| `11000` | Valida | `[E, C, canal] = [2, 2, 16]` |
| `11001` | Valida | `[E, C, Y, canal] = [2, 2, 3, 16]` |
| `11010` | Valida | `[E, C, X, canal] = [2, 2, 3, 16]` |
| `11011` | Valida | `[E, C, X, Y, canal] = [2, 2, 3, 3, 16]` |
| `11100` | Valida | `[E, C, D, canal] = [2, 2, 8, 8]` |
| `11101` | Valida | `[E, C, D, Y, canal] = [2, 2, 8, 3, 8]` |
| `11110` | Valida | `[E, C, D, X, canal] = [2, 2, 8, 3, 8]` |
| `11111` | Valida | `[E, C, D, X, Y, canal] = [2, 2, 8, 3, 3, 8]` |

Cuando `E=0`, `X` e `Y` tambien deben ser cero. Solo quedan las tres
combinaciones no vacias de `C` y `D`. Cuando `E=1`, son validas todas las
combinaciones de los otros cuatro indicadores salvo la completamente vacia:

$$
3 + (2^4 - 1) = 3 + 15 = 18
$$

### `share_up` visto como un `merge`

Supongamos una hoja general por color:

```text
color  feature
0           10
1           20
```

y una tabla detallada por ejemplo, color y posicion. En Pandas se propagaria
la propiedad general mediante:

```python
detailed = detailed.merge(
    color_features,
    on="color",
    how="left",
)
```

El valor 10 se repite para todas las filas detalladas del color 0 y el valor 20
para todas las del color 1. Esto se parece a `share_up`: una hoja pequena como
`[color, channel]` transmite informacion a hojas mas detalladas como
`[example, color, x, y, channel]`.

La implementacion PyTorch usa `unsqueeze` y broadcasting. No crea fisicamente
un DataFrame con filas duplicadas.

### `share_down` visto como un `groupby`

Si se parte de:

```text
example  color  x  y  feature
0        0      0  0      0.2
0        0      0  1      0.4
0        1      1  0      0.8
1        1      2  2      1.0
```

se puede obtener un resumen por color mediante:

```python
summary = detailed.groupby("color")["feature"].mean()
```

Resultado:

```text
color
0    0.3
1    0.9
```

Esto se parece a `share_down`: una hoja detallada elimina algunos ejes mediante
promedios y comunica el resumen a una hoja menor.

```text
share_up    ~= merge/broadcast hacia mas claves
share_down  ~= groupby/mean hacia menos claves
```

### Relacion con `task.problem`

Conviene separar cuatro objetos:

```text
dims              -> mascara que selecciona ejes
dim_lengths       -> tamanos reales de esos ejes
task.problem      -> indices de los colores observados en las cuadrículas
MultiTensor       -> representaciones internas aprendidas
```

Durante el preprocesamiento, `task.problem` si usa temporalmente una dimension
one-hot de colores:

```text
[example, color, x, y, mode]
```

Despues se aplica `argmax` sobre `color` y queda:

```text
[example, x, y, mode]
```

Ese one-hot temporal no es la mascara `dims`. Son mecanismos diferentes:

- El one-hot representa que color real ocupa una celda.
- `dims` describe que ejes existen en una hoja del multitensor.

### Dos significados diferentes de mascara

En el repositorio aparece la palabra mascara con dos sentidos:

```text
dims = [1, 0, 1, 1, 0]
```

es una mascara estructural de presencia de ejes.

```text
task.masks[example, x, y, mode]
```

es una mascara espacial que indica si una celda pertenece a la cuadrícula real
o al padding reservado para igualar tamanos.

La primera organiza las hojas del multitensor. La segunda evita tratar el
relleno como si fueran datos reales.

### Inventario de estructuras tensoriales y pesos del programa

Esta seccion recorre las estructuras importantes en el mismo orden en que
aparecen durante una iteracion:

```text
Task
  -> MultiTensorSystem
  -> parametros de ARCCompressor
  -> latente z y residual x
  -> logits y mascaras predichas
  -> perdida
  -> gradientes y Adam
  -> Logger y seleccion pass@2
```

No se enumeran contadores de bucles como `example_num` o `layer_num`, salvo
cuando ayudan a interpretar un indice. El objetivo es inventariar las
estructuras que almacenan datos, parametros, activaciones o resultados.

#### Cuatro familias que no deben mezclarse

| Familia | Ejemplos | Persiste entre iteraciones | La optimiza Adam |
| --- | --- | --- | --- |
| Datos observados | `task.problem`, `task.masks` | Si, no cambian | No |
| Parametros | `mean`, matrices `weight`, `bias` | Si, cambian | Si |
| Activaciones | `z`, `x`, `logits`, `ce` | No, se recalculan | No directamente |
| Estado de seguimiento | EMA, curvas, soluciones | Si | No |

Una activacion puede tener gradientes dentro del grafo de PyTorch, pero eso no
la convierte en un parametro. Adam actualiza los tensores hoja incluidos en
`model.weights_list`; las activaciones se vuelven a calcular despues.

#### 1. Datos y metadatos de `Task`

Estas estructuras se crean en `preprocessing.Task` antes de construir el
modelo.

| Nombre | Tipo | Forma o estructura | Contenido y uso | Entrenable |
| --- | --- | --- | --- | --- |
| `task.unprocessed_problem` | `dict` de listas | JSON original | Pares `train` y entradas `test` sin convertir | No |
| `task.n_train` | `int` | Escalar | Numero de demostraciones completas | No |
| `task.n_test` | `int` | Escalar | Numero de ejemplos de test | No |
| `task.n_examples` | `int` | `n_train + n_test` | Longitud real del eje `examples` | No |
| `task.shapes` | lista anidada | `[E][2][2]` | Por ejemplo: forma de input/output y longitudes x/y | No |
| `task.in_out_same_size` | `bool` | Escalar | Indica si cada demostracion conserva su forma | No |
| `task.all_in_same_size` | `bool` | Escalar | Indica si todos los inputs tienen la misma forma | No |
| `task.all_out_same_size` | `bool` | Escalar | Indica si todos los outputs tienen la misma forma | No |
| `task.colors` | `list[int]` | `[C+1]` | Traduce indice interno a etiqueta ARC; incluye negro | No |
| `task.n_colors` | `int` | Escalar | Numero de colores no negros; longitud del eje `colors` | No |
| `task.n_x`, `task.n_y` | `int` | Escalares | Maximo numero de filas y columnas reservado | No |

`E`, `C`, `X` e `Y` se usaran a partir de ahora como abreviaturas de
`n_examples`, `n_colors`, `n_x` y `n_y`. `D` siempre vale 8.

##### `task.problem` antes y despues de `argmax`

Durante su construccion, `task.problem` es un array NumPy one-hot:

```text
tipo  = numpy.ndarray
forma = [E, C+1, X, Y, 2]
ejes  = [example, color, x, y, input/output]
```

El ultimo eje de longitud 2 usa:

```text
0 = input
1 = output
```

Despues de `np.argmax(..., axis=1)`, el eje one-hot de color desaparece:

```text
tipo  = torch.Tensor de enteros
forma = [E, X, Y, 2]
valor = indice interno del color correcto
```

Este tensor es el objetivo que usa la `cross_entropy`. La salida de los
ejemplos de test no se rellena y se excluye expresamente de la perdida.

##### `task.masks`

```text
tipo  = torch.Tensor
forma = [E, X, Y, 2]
valor = 1 para una celda real; 0 para padding
```

Tiene la misma forma que `task.problem`, pero no contiene colores. Se usa para
evitar que las operaciones direccionales procesen padding y para respetar las
formas reales de cada cuadrícula.

##### `task.solution` y `task.solution_hash`

`task.solution` solo existe cuando el split dispone de respuestas de
referencia. Tras eliminar su eje one-hot tiene forma `[n_test, X, Y]` y contiene
indices internos de color. No entra en la perdida; sirve para evaluar.
`task.solution_hash` es un entero que permite comparar la respuesta verdadera
con las candidatas sin pasarla al modelo.

#### 2. Estructuras de `MultiTensorSystem`

| Nombre | Tipo | Contenido | Papel |
| --- | --- | --- | --- |
| `multitensor_system.dim_lengths` | `list[int]` | `[E, C, 8, X, Y]` | Tamanos reales de los cinco ejes |
| `dims` | `list[int]` | Cinco bits `ECDXY` | Selecciona los ejes presentes en una hoja |
| `MultiTensor.data` | Lista anidada | Arbol binario de profundidad 5 | Almacena hasta 32 hojas indexables por `dims` |
| `MultiTensor[dims]` | Cualquier objeto | Una hoja | Puede ser tensor, pesos, posterior u otra lista |

`MultiTensor` no obliga a que todas sus hojas contengan el mismo tipo. La
estructura exterior solo proporciona el indice `dims`. Por eso existen, entre
otros:

```text
MultiTensor[Tensor]
MultiTensor[[weight, bias]]
MultiTensor[[mean, local_capacity_adjustment]]
MultiTensor[8 x 8 pares de pesos]
```

El decorador `@multify` recorre las 18 mascaras validas. Para cada una extrae
la hoja correspondiente de sus argumentos, ejecuta la funcion y guarda el
resultado bajo la misma mascara.

#### 3. Bloques basicos usados para construir pesos

Todos los parametros siguientes se crean con `requires_grad=True`.

##### Tensor de ceros entrenable

```python
zeros = torch.zeros(shape, requires_grad=True)
```

Se utiliza para ajustes de capacidad y capacidades objetivo. Que empiece en
cero no significa que permanezca en cero: Adam puede modificarlo.

##### Transformacion lineal

`Initializer.initialize_linear()` devuelve una lista de dos tensores:

```text
linear = [weight, bias]

weight.shape = [n_in, n_out]
bias.shape   = [n_out]
```

La operacion `affine` transforma solo el ultimo eje:

```python
output = torch.matmul(input, weight)
output = output + bias  # solamente cuando use_bias=True
```

En analogia con Pandas, cada fila conserva su `MultiIndex`, mientras sus
columnas numericas pasan de `n_in` a `n_out`.

##### Bloque residual

`Initializer.initialize_residual()` devuelve dos transformaciones lineales:

```text
residual_weights = [
    [weight_down, bias_down],
    [weight_up,   bias_up],
]
```

Sus formas son:

```text
weight_down = [channel_dim, n_in]
bias_down   = [n_in]
weight_up   = [n_out, channel_dim]
bias_up     = [channel_dim]
```

El recorrido es:

```text
x [..., channel_dim]
  -> proyeccion down [..., n_in]
  -> operacion propia [..., n_out]
  -> proyeccion up [..., channel_dim]
  -> sumar al x original [..., channel_dim]
```

Por eso el ancho temporal de una operacion puede ser 2, 4 o 16, pero el
residual `x` siempre vuelve a 16 canales si no contiene `directions`, o a 8 si
si las contiene.

##### Posterior latente

Cada hoja de `multiposteriors` contiene:

```text
posterior = [mean, local_capacity_adjustment]
```

Ambos tensores tienen:

```text
forma = shape(dims, decoding_dim=4)
```

`mean` representa la señal aprendida antes de mezclarla con ruido.
`local_capacity_adjustment` reparte la capacidad de informacion dentro de la
hoja. Ambos son parametros entrenables y especificos de la tarea.

##### Matriz 8 por 8 de comunicacion direccional

Cada hoja de `direction_share_weights` contiene:

```text
direction_maps[direccion_destino][direccion_origen] = [weight, bias]
```

Hay $8 \times 8 = 64$ transformaciones lineales por hoja. Cada matriz `weight`
tiene forma `[channel_dim, channel_dim]`. La simetrizacion hace que muchas
posiciones reutilicen el mismo objeto de pesos.

#### 4. Parametros almacenados en `ARCCompressor`

##### Decodificacion VAE-like

| Atributo | Estructura por hoja `dims` | Forma | Uso |
| --- | --- | --- | --- |
| `model.multiposteriors` | `[mean, local_capacity_adjustment]` | Ambos `shape(dims, 4)` | Parametrizar la muestra latente `z` |
| `model.decode_weights` | `[weight, bias]` | `[4, channel_dim]`, `[channel_dim]` | Proyectar `z` al residual `x` |
| `model.target_capacities` | `Tensor` | `[4]` | Capacidad global por canal latente de esa hoja |

`target_capacities` no contiene una capacidad por pixel. Su forma `[4]` se
propaga por broadcasting sobre los ejes presentes en `mean`.

##### Pesos de las cuatro capas

Cada atributo de la tabla es una lista de longitud `n_layers = 4`. Cada
elemento de esa lista es un `MultiTensor` con 18 hojas.

| Atributo por capa | Estructura de cada hoja | Ancho temporal | Funcion |
| --- | --- | --- | --- |
| `share_up_weights[layer]` | Bloque residual | 16 -> 16 | Llevar vistas generales hacia vistas detalladas |
| `softmax_weights[layer]` | Bloque residual | 2 -> `2*(2^k-1)` | Comparar valores sobre subconjuntos de ejes |
| `cummax_weights[layer]` | Bloque residual | 4 -> 4 | Propagar maximos en ocho direcciones |
| `shift_weights[layer]` | Bloque residual | 4 -> 4 | Desplazar informacion un pixel |
| `direction_share_weights[layer]` | Matriz 8 por 8 de lineales | `channel_dim` -> `channel_dim` | Comunicar direcciones |
| `nonlinear_weights[layer]` | Bloque residual | 16 -> 16 | Aplicar SiLU |
| `share_down_weights[layer]` | Bloque residual | 8 -> 8 | Resumir vistas detalladas en vistas generales |

En la fila de `softmax`, $k$ es el numero de ejes activos entre `colors`,
`directions`, `x` e `y`. `examples` nunca participa en su softmax.

`cummax` y `shift` solo ejecutan su operacion espacial para las hojas:

```text
[1, 1, 1, 1, 1]
[1, 0, 1, 1, 1]
```

En las demas hojas devuelven el residual sin aplicar esa operacion.

##### Cabezas finales

| Atributo | Estructura | Formas | Hoja consumida | Resultado |
| --- | --- | --- | --- | --- |
| `model.head_weights` | `[weight, bias]` | `[16,2]`, `[2]` | `[1,1,0,1,1]` | Logits de color para input/output |
| `model.mask_weights` | `[weight, bias]` | `[16,2]`, `[2]` | `[1,0,0,1,0]` y `[1,0,0,0,1]` | Logits de filas y columnas validas |

Las dos columnas de `head_weights[0]` se inicializan iguales, pero el tensor
completo sigue siendo entrenable. En `forward()`, la matriz se aplica sin su
sesgo convencional y despues se suma expresamente `100 * head_weights[1]`.

El mismo `mask_weights` se reutiliza para `x` e `y`; eso ayuda a tratar filas y
columnas de forma simetrica.

##### `model.weights_list`

```text
tipo      = list[torch.Tensor]
contenido = todos los tensores creados por Initializer
consumidor = torch.optim.Adam
```

No es una copia de los pesos: contiene referencias a los tensores originales.
Tras la simetrizacion, algunas hojas apuntan al mismo tensor. Ademas, los
tensores creados inicialmente y sustituidos por esas referencias pueden seguir
en `weights_list` aunque ya no participen en el `forward`; tendran
`grad is None` y Adam no podra cambiar la salida a traves de ellos.

#### 5. Variables de `channel_layer()` y `decode_latents()`

Para cada una de las 18 hojas se crean estas variables:

| Nombre | Tipo y forma | Significado |
| --- | --- | --- |
| `mean` | Parametro `shape(dims,4)` | Mensaje latente aprendido |
| `local_capacity_adjustment` | Parametro `shape(dims,4)` | Ajuste local del presupuesto de informacion |
| `target_capacity` | Parametro `[4]` | Control global por canal latente |
| `desired_global_capacity` | Activacion `[4]` | Capacidad positiva transformada exponencialmente |
| `output_scaling` | Activacion `[4]` | Escala final dependiente de la capacidad |
| `desired_local_capacity` | Activacion `shape(dims,4)` | Capacidad repartida por elemento |
| `noise_std`, `noise_var` | Activaciones `shape(dims,4)` | Cantidad de ruido del posterior |
| `signal_std`, `signal_var` | Activaciones `shape(dims,4)` | Cantidad de señal aprendida |
| `normalized_mean` | Activacion `shape(dims,4)` | `mean` centrada y normalizada |
| `z` | Activacion `shape(dims,4)` | Muestra: señal aprendida mas ruido aleatorio |
| `KL` | Activacion `shape(dims,4)` | Coste de informacion por elemento latente |
| `x` | Activacion `shape(dims,channel_dim)` | `z` proyectado al residual de 16 u 8 canales |

`KL` conserva el eje de cuatro canales latentes. Mas tarde se suman todos sus
elementos; no es todavia un escalar dentro de `channel_layer()`.

`decode_latents()` agrupa los resultados en:

```text
x           = MultiTensor[Tensor], 18 hojas residuales
KL_amounts  = list[Tensor], 18 tensores KL con forma por hoja
KL_names    = list[str], 18 mascaras convertidas a texto
```

#### 6. El residual `x` durante `forward()`

`x` es la variable principal del razonamiento. Sigue siendo un `MultiTensor` de
18 hojas durante las cuatro capas.

```text
hoja sin directions: shape(dims, 16)
hoja con directions: shape(dims, 8)
```

Las operaciones internas usan una variable temporal `z`, pero no es el mismo
`z` del VAE:

```python
z = affine(x, residual_weights[0])
z = operation(z)
z = affine(z, residual_weights[1])
x = x + z
```

En este contexto, `z` solo significa "propuesta temporal de cambio residual".
Es una reutilizacion local del nombre.

Otras estructuras temporales relevantes son:

| Variable | Donde aparece | Uso |
| --- | --- | --- |
| `lower_xs` | `share_up` | Hojas menos detalladas preparadas con `unsqueeze` |
| `higher_xs` | `share_down` | Hojas mas detalladas reducidas con `mean` |
| `softmaxxes` | `softmax` | Resultados para cada subconjunto de ejes, luego concatenados |
| `masks` local | Capa direccional | `task.masks` reordenada para hacer broadcasting con una hoja |
| `x_list`, `z_list` | `direction_share` | Ocho cortes, uno por direccion |

#### 7. Salidas de `ARCCompressor.forward()`

| Nombre | Forma al salir de `forward()` | Contenido |
| --- | --- | --- |
| `output` | `[E,C,X,Y,2]` | Logits para colores no negros y modos input/output |
| `x_mask` | `[E,X,2]` | Puntuacion por fila para input/output |
| `y_mask` | `[E,Y,2]` | Puntuacion por columna para input/output |
| `KL_amounts` | Lista de 18 tensores | Costes KL por hoja |
| `KL_names` | Lista de 18 textos | Mascara asociada a cada coste KL |

Aqui `output` todavia no contiene el negro. El nombre cambia a `logits` al
entrar en `train.take_step()`, donde se antepone un plano de ceros:

```python
logits = torch.cat(
    [torch.zeros_like(output[:, :1, :, :]), output],
    dim=1,
)
```

Desde ese momento:

```text
logits.shape = [E,C+1,X,Y,2]
```

El cero fijo es el logit de referencia del negro.

`x_mask` e `y_mask` tampoco son las mascaras binarias de `task.masks`. Son
logits aprendidos que puntuan que filas y columnas deben quedar dentro del
recorte. `postprocess_mask()` les suma modificadores de 0 o -1000 para prohibir
indices fuera de los limites posibles.

#### 8. Tensores temporales de la reconstruccion

Para cada ejemplo visible y cada modo input/output se crean:

| Nombre | Forma | Uso |
| --- | --- | --- |
| `logits_slice` | `[C+1,X,Y]` | Logits de un ejemplo y modo concretos |
| `problem_slice` | `[X,Y]` | Indice correcto de color por celda |
| `output_shape` | Lista `[out_x,out_y]` | Tamano real del objetivo |
| `x_logprobs` | `[X-out_x+1]` | Score de cada desplazamiento vertical |
| `y_logprobs` | `[Y-out_y+1]` | Score de cada desplazamiento horizontal |
| `x_log_partition`, `y_log_partition` | Escalares | Normalizadores de los desplazamientos |
| `target_crop` | `[out_x,out_y]` | Zona real del objetivo |
| `logits_crops` | `[n_offsets,C+1,out_x,out_y]` | Todos los recortes candidatos en batch |
| `target_batch` | `[n_offsets,out_x,out_y]` | Objetivo repetido para cada candidato |
| `ce` | `[n_offsets,out_x,out_y]` | Cross-entropy por candidato y pixel |
| `ce_sum` | `[n_x_offsets,n_y_offsets]` | Error total de colores por recorte |
| `logprobs` | `[n_x_offsets,n_y_offsets]` | Evidencia conjunta de recorte y colores |
| `logprob` | Escalar | Evidencia agregada sobre recortes |

La perdida combina:

```text
total_KL            = suma de los 18 tensores de KL
reconstruction_error = suma de -logprob sobre datos visibles
loss                 = total_KL + 10 * reconstruction_error
```

Los tres terminan siendo tensores escalares de PyTorch para conservar el grafo
de diferenciacion.

#### 9. Gradientes y estado de Adam

Despues de:

```python
loss.backward()
```

cada parametro utilizado en el grafo recibe otro tensor en su atributo `.grad`
con la misma forma:

```text
weight.shape == weight.grad.shape
mean.shape   == mean.grad.shape
```

`optimizer.step()` mantiene para cada parametro activo dos tensores internos,
habitualmente llamados primer y segundo momento:

```text
exp_avg     = media movil del gradiente
exp_avg_sq  = media movil del gradiente al cuadrado
```

Ambos tienen la misma forma que el parametro. No pertenecen a
`model.weights_list`, pero forman parte del estado del optimizador y explican
por que Adam consume memoria adicional.

`optimizer.zero_grad()` limpia los gradientes. No borra los pesos ni los
momentos de Adam.

#### 10. Tensores y estructuras del `Logger`

El logger recibe activaciones con `.detach()`: conserva sus valores, pero corta
su conexion con el grafo de gradientes.

##### Curvas de entrenamiento

| Nombre | Estructura | Contenido |
| --- | --- | --- |
| `KL_curves` | `dict[str,list]` | Una curva de escalares por mascara `dims` |
| `total_KL_curve` | `list[Tensor]` | KL total por iteracion |
| `reconstruction_error_curve` | `list[Tensor]` | Error de reconstruccion por iteracion |
| `loss_curve` | `list[Tensor]` | Perdida total por iteracion |

`materialize_curves()` apila cada lista y la convierte de GPU a listas de
numeros Python.

##### Muestra actual y media exponencial

| Nombre | Forma | Significado |
| --- | --- | --- |
| `current_logits` | `[n_test,C+1,X,Y]` | Logits actuales del modo output de test |
| `current_x_mask` | `[n_test,X]` | Scores actuales de filas |
| `current_y_mask` | `[n_test,Y]` | Scores actuales de columnas |
| `ema_logits` | `[n_test,C+1,X,Y]` | Media exponencial de logits |
| `ema_x_mask` | `[n_test,X]` | Media exponencial de scores x |
| `ema_y_mask` | `[n_test,Y]` | Media exponencial de scores y |

La actualizacion usa:

$$
\operatorname{EMA}_{t}=0.97\operatorname{EMA}_{t-1}+0.03\operatorname{actual}_{t}
$$

##### Soluciones discretas

| Nombre | Tipo | Uso |
| --- | --- | --- |
| `colors` local | Tensor `[n_test,X,Y]` | `argmax` de color por pixel |
| `uncertainties` | Tensor `[n_test,X,Y]` | `logsumexp(logits) - max(logits)` |
| `solution_slices` | Lista y luego tupla | Cuadrículas recortadas y recoloreadas |
| `solution_hashes_count` | `dict[hash,float]` | Evidencia acumulada por candidata |
| `solution_most_frequent` | Tupla de cuadrículas | Primer intento pass@2 |
| `solution_second_most_frequent` | Tupla de cuadrículas | Segundo intento pass@2 |
| `solution_contributions_log` | Lista por iteracion | Hash y score de muestra actual y EMA |
| `solution_picks_history` | Lista por iteracion | Historial de los dos hashes elegidos |

#### 11. Ejemplo completo con la tarea `007bbfb7`

Para esta tarea:

```text
E = 6 ejemplos: 5 train + 1 test
C = 4 colores no negros
D = 8 direcciones
X = 9 filas
Y = 9 columnas
task.colors = [0, 2, 4, 6, 7]
```

Las formas principales son:

```text
task.problem                         [6, 9, 9, 2]
task.masks                           [6, 9, 9, 2]

mean de dims [1,1,0,1,1]            [6, 4, 9, 9, 4]
x de dims [1,1,0,1,1]               [6, 4, 9, 9, 16]

mean de dims [1,1,1,1,1]            [6, 4, 8, 9, 9, 4]
x de dims [1,1,1,1,1]               [6, 4, 8, 9, 9, 8]

target_capacity de cualquier hoja   [4]
output al salir de forward           [6, 4, 9, 9, 2]
logits despues de anadir negro       [6, 5, 9, 9, 2]
x_mask                               [6, 9, 2]
y_mask                               [6, 9, 2]
current_logits del test              [1, 5, 9, 9]
```

La hoja `[1,1,0,1,1]` es especialmente importante porque la cabeza de color la
convierte en `output`. La hoja `[1,1,1,1,1]` contiene ademas las ocho
direcciones y participa en `cummax` y `shift`.

#### 12. Receta de inspeccion en el debugger

Para reconocer las estructuras sin modificarlas:

```python
# Metadatos de la tarea
task.multitensor_system.dim_lengths
list(task.multitensor_system)

# Posterior de una hoja
dims = [1, 1, 0, 1, 1]
mean, local_adjustment = model.multiposteriors[dims]
mean.shape
local_adjustment.shape

# Pesos de decodificacion de esa hoja
decode_weight, decode_bias = model.decode_weights[dims]
decode_weight.shape
decode_bias.shape

# Primer bloque residual de softmax
down_linear, up_linear = model.softmax_weights[0][dims]
down_linear[0].shape   # weight down
down_linear[1].shape   # bias down
up_linear[0].shape     # weight up
up_linear[1].shape     # bias up

# Estado del optimizador para un parametro despues del primer step
optimizer.state[decode_weight].keys()
optimizer.state[decode_weight]["exp_avg"].shape
optimizer.state[decode_weight]["exp_avg_sq"].shape
```

La pregunta que conviene hacer siempre al encontrar una variable es:

```text
Es dato fijo, parametro entrenable, activacion temporal o estado de seguimiento?
Que representa cada eje de su shape?
```

### Que se aprendio

- El multitensor no es un unico tensor de cinco dimensiones: es una coleccion
  de 18 tensores con combinaciones diferentes de cinco ejes conceptuales.
- Los cinco bits son interruptores estructurales, no longitudes ni datos.
- Los tamanos reales se guardan por separado en `dim_lengths` y se insertan al
  construir la forma de cada hoja.
- Cada hoja puede imaginarse como un `DataFrame` cuyo `MultiIndex` contiene los
  ejes seleccionados por la mascara y cuyas columnas son los canales aprendidos.
- `share_up` se parece a propagar datos mediante `merge`; `share_down`, a
  resumirlos mediante `groupby().mean()`.
- Las 18 hojas no son estadisticas precalculadas de la cuadrícula. Sus valores
  son representaciones neuronales que se ajustan mediante gradientes.
- La mascara `dims` no debe confundirse con el one-hot temporal de colores ni
  con las mascaras espaciales de padding.

### Puntos de debug utiles

- Inspeccionar `task.multitensor_system.dim_lengths` para ver los cinco tamanos
  reales de una tarea.
- Evaluar `list(task.multitensor_system)` para obtener las 18 mascaras validas.
- Evaluar `task.multitensor_system.shape(dims, 4)` para observar la forma de un
  posterior latente concreto.
- Detenerse en `Initializer.initialize_posterior()` para comparar `dims` con la
  forma real de `mean`.
- Detenerse en `layers.share_direction()` para observar como `share_up` inserta
  ejes con `unsqueeze` y como `share_down` los agrega con `mean`.

### Estado y proxima sesion

- Se separaron conceptualmente mascaras, longitudes, datos observados y
  representaciones aprendidas.
- Se obtuvo una traduccion completa del multitensor a estructuras de Pandas.
- Se enumeraron las 32 combinaciones y se justificaron las 18 validas.
- Siguiente paso sugerido: inspeccionar una tarea real en el debugger y
  convertir dos hojas concretas, por ejemplo `[0, 1, 0, 0, 0]` y
  `[1, 1, 0, 1, 1]`, a DataFrames temporales para observar sus formas y valores
  antes y despues de `share_up`.