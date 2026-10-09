# Rakuten — clasificación multimodal de productos
## Guía del proyecto: objetivos, decisiones y demo

*Versión en inglés: `docs/guia-proyecto-en.md` — contenido idéntico.*

---

## Cómo leer esta guía

Está escrita para alguien que conoce los conceptos de MLOps por encima y
quiere entender **qué hace cada herramienta en general** y, por separado,
**qué hace exactamente en este proyecto**. Son dos cosas distintas y no
siempre coinciden: Airflow puede hacer muchísimo más de lo que hace aquí, y
esa diferencia es en sí misma una decisión que hay que saber defender.

La Parte 1 sitúa el problema. La Parte 2 recoge las decisiones estratégicas,
que es donde se juega la nota. La Parte 3 explica las herramientas una a una.
La Parte 4 recorre el sistema entero dos veces, siguiendo un dato concreto.
Las Partes 5 y 6 son operativas: los comandos de la demo y el significado de
cada panel. La Parte 7 anticipa preguntas y la Parte 8 enumera lo que el
sistema **no** hace.

Si solo tienes diez minutos antes de la defensa: Parte 2, Parte 5 y Parte 6.

---

# Parte 1 — El proyecto

## 1.1 El problema

Rakuten es un marketplace. Cuando un vendedor publica un producto, alguien
tiene que decidir en qué categoría va. El catálogo tiene **27 categorías**
(los `prdtypecode`), y clasificar a mano decenas de miles de productos no
escala.

El dato de cada producto tiene tres partes:

- **designation** — el título, siempre presente.
- **description** — texto libre, frecuentemente vacío.
- **imagen** — una foto del producto; hay 84.916 en el conjunto de
  entrenamiento.

Es un problema **multimodal**: la respuesta está repartida entre el texto y
la imagen, y ninguno de los dos basta por sí solo. Un cable y un cargador se
parecen mucho en la foto y se distinguen en el título; dos libros con títulos
genéricos se distinguen por la portada.

## 1.2 Qué pedía el enunciado

El ejercicio no es "entrenar el mejor modelo". Es **poner un modelo en
producción y mantenerlo vivo**. Los objetivos, tal y como estaban formulados:

1. **Recoger los datos y guardarlos en una base de datos SQL (o NoSQL) en
   local**, con un script de Python que se ejecute una vez.
2. **Versionar los datos** con DVC, sin Git, y guardar sus hashes en MLflow.
3. **Servir el modelo** detrás de una API.
4. **Registrar los entrenamientos** en MLflow: parámetros, métricas y
   artefactos.
5. **Comparar versiones** y promover la mejor a producción.
6. **Monitorizar** el servicio y detectar *drift* en los datos.
7. **Construir dashboards** en Grafana: uno de drift de datos, otro de salud
   de la API.
8. **Orquestar** el ciclo con Airflow.
9. **Contenerizar** todo y tener CI.

Nótese lo que **no** pide: no pide la mayor exactitud posible. Eso cambia por
completo cuál es el modelo correcto, y es la primera decisión estratégica.

## 1.3 Vista de pájaro

Siete servicios, cada uno en su contenedor:

```
                      ┌──────────────────────────────────────┐
   Datos              │            Monitorización            │
  ┌──────────────┐    │  ┌─────────────┐    ┌─────────────┐  │
  │ CSVs Rakuten │    │  │  Evidently  │    │ Prometheus  │  │
  │ + 84.916 img │    │  │ POST /run   │    │   :9090     │  │
  └──────┬───────┘    │  │   :8100     │    └──────┬──────┘  │
         │ importación│  └──────┬──────┘           │         │
         │  una vez   │         │                  ▼         │
         ▼            │         │           ┌─────────────┐  │
  ┌──────────────┐    │         │           │   Grafana   │  │
  │    SQLite    │◄───┼─────────┘           │ 2 dashboards│  │
  │  products    │    │  resumen            │    :3000    │  │
  │  predictions │    └─────────────────────┴──────┬──────┘  │
  │  drift_reports│                                │         │
  └──────┬───────┘                                 └─────────┘
         │  ▲                      Orquestación
         │  │ guarda cada    ┌──────────────────────┐
         │  │ predicción     │       Airflow        │
         │  │                │  rakuten_training    │
         │  │                │  rakuten_drift_check │
         │  │                │        :8080         │
         │  │                └───┬──────────────┬───┘
         │  │      cada hora ────┘              │ entrena + recarga
         │  │      POST /run                    ▼
         ▼  │                            ┌──────────────┐
  ┌──────────────┐    Servicio           │   FastAPI    │
  │ training.py  │  ┌─────────────┐      │  /predict/   │
  │ LSTM + VGG16 │  │  Streamlit  │─────▶│  /training/  │
  │   + blend    │  │    :8501    │      │  /metrics    │
  └──────┬───────┘  └─────────────┘      │    :8000     │
         │                               └──────▲───────┘
         │ registra      ┌──────────────┐       │
         └──────────────▶│    MLflow    │───────┘
                         │  tracking +  │ champion
                         │   Registry   │
                         │    :5000     │
                         └──────────────┘
```

| Servicio | Puerto | Papel |
|---|---|---|
| `api` | 8000 | FastAPI: sirve predicciones, lanza entrenamientos, expone métricas |
| `streamlit` | 8501 | Interfaz de demostración |
| `mlflow` | 5000 | Seguimiento de experimentos y Model Registry |
| `drift` | 8100 | Servicio de Evidently; calcula el drift cuando se le pide |
| `airflow` | 8080 | Orquestador: dos DAGs |
| `prometheus` | 9090 | Recolecta métricas de la API |
| `grafana` | 3000 | Dashboards y alertas |

## 1.4 El modelo, en dos párrafos

Dos ramas que se entrenan por separado y se combinan al final:

- **Rama de texto** — una red LSTM sobre `designation + description`. El
  texto se tokeniza (se convierte cada palabra en un número según un
  vocabulario aprendido) y la LSTM lee esa secuencia en orden.
- **Rama de imagen** — **VGG16**, una red convolucional preentrenada sobre
  ImageNet. Se reutiliza lo que ya sabe ver (bordes, texturas, formas) y solo
  se entrena la capa final para las 27 categorías. Esto se llama *transfer
  learning*, y es lo que permite entrenar con 84.916 imágenes en vez de los
  millones que haría falta desde cero.

Cada rama produce 27 probabilidades. La predicción final es una **media
ponderada** de las dos: `w_texto × P_texto + w_imagen × P_imagen`. Esos dos
pesos no se eligen a mano, se buscan sobre un conjunto reservado para ello.
El resultado pesa unos 150 MB y responde en menos de un segundo en CPU.

---

# Parte 2 — Las decisiones estratégicas

Esta es la parte que diferencia un proyecto aprobado de uno bueno. Cada
decisión se presenta con lo que cuesta, no solo con lo que gana: una decisión
sin coste reconocido suele ser una decisión que no se ha pensado.

## 2.1 Siete contenedores en vez de uno

**La decisión.** Cada servicio corre en su propia imagen, con su propio
entorno de Python, y se comunican por HTTP y un volumen compartido.

**Por qué.** No es decoración arquitectónica: es que **no caben juntos**.

- TensorFlow 2.13 exige `typing_extensions < 4.6` y `protobuf < 5`.
- El servidor de MLflow necesita SQLAlchemy y Alembic, que exigen
  `typing_extensions >= 4.6`.
- Evidently arrastra Litestar y sus propios extras de Uvicorn.

Son requisitos mutuamente incompatibles. Un único entorno de Python con las
tres cosas sencillamente no se instala. Aislar cada conjunto de dependencias
en su propia imagen es lo que hace el stack **instalable**, no lo que lo hace
elegante.

**Qué cuesta.** Siete contenedores en una máquina de 2 núcleos compiten por
CPU. Durante esta sesión aparecieron dos fallos que son consecuencia directa
de eso: el webserver de Airflow moría por un *timeout* de gunicorn de 120 s
durante su primer arranque, y la descarga de artefactos de MLflow fallaba a
los 30 s exactos por el *timeout* por defecto de otro gunicorn distinto.
Ambos se corrigieron subiendo los timeouts a 300 s. La lección: cuando se
reparte poca CPU entre muchos procesos, los fallos aparecen como timeouts, no
como errores de lógica.

Y la lección tuvo una segunda parte. Con un reentrenamiento corriendo dentro
del contenedor `api`, los 300 s **tampoco bastaron**: el scheduler registró
`Heartbeat recovered after 41.33 seconds` y el webserver salió con
`No response from gunicorn master within 300 seconds`. No era un fallo de
Airflow ni falta de memoria (`OOMKilled` era falso) — el supervisor se rindió
esperando a un proceso maestro que solo estaba haciendo cola por CPU. Ahora
son 900 s y un único *worker*, porque con `SequentialExecutor` y un solo
espectador el segundo no aportaba nada y duplicaba el trabajo de arranque
sobre el recurso escaso.

Y como red de seguridad, los siete servicios llevan `restart: unless-stopped`.
Un contenedor que muere por un pico de carga vuelve solo, en vez de quedarse
apagado hasta que alguien mire `docker compose ps`. Que es exactamente como
se descubrió este fallo: Airflow llevaba cuatro horas parado sin que nada lo
dijera.

**Y la causa de raíz del arranque lento de Airflow era otra.** Su base de
metadatos es SQLite y vivía dentro del contenedor, no en un volumen. Cada
`docker compose up --build` recrea el contenedor y la destruye, así que
Airflow reinicializaba *todo* en cada construcción: migraciones, varios
cientos de filas de permisos de Flask-AppBuilder (el muro de `Added
Permission ...` en los logs) y, lo que de verdad duele en una demo, **el
historial de ejecuciones de los DAGs y su estado de pausa**. Los minutos de
arranque no eran inherentes a Airflow: era reconstruirlo entero cada vez.

Ahora la base vive en un volumen con nombre (`airflow-db`), apuntada por
`AIRFLOW__DATABASE__SQL_ALCHEMY_CONN`. El directorio se crea en el Dockerfile
con el propietario correcto, porque Docker siembra un volumen vacío desde la
imagen conservando la propiedad; creado al montar, pertenecería a `root` y
Airflow no podría escribir. El primer arranque sigue pagando la
inicialización una vez; los siguientes, no.

## 2.2 El modelo ligero en vez del más preciso

**La decisión.** Se sirve el modelo LSTM + VGG16 (~150 MB), no el *ensemble*
de un compañero con F1 ponderado de 0,9151.

**Por qué.** Los dos modelos se juzgan en ejes distintos. El del compañero
son siete *encoders* afinados, unos 4,5 GB de pesos, varios segundos por
predicción en CPU. Para un proyecto evaluado por criterios de MLOps —tamaño
de imagen, coste de versionado, latencia de servicio, una demo que tiene que
funcionar en directo— el modelo ligero es la opción defendible.

**Qué cuesta.** Exactitud. Y hay que decirlo sin rodeos cuando pregunten.

**El matiz que lo convierte en una buena respuesta**: el sitio correcto para
el modelo más fuerte es el **Model Registry, como versión competidora**. Esa
es precisamente la comparación para la que existe un registry. No se ha
descartado el modelo; se ha colocado donde el sistema sabe evaluarlo.

## 2.3 El Model Registry como única fuente de verdad

**La decisión.** Lo que la API sirve es lo que el registry ha bendecido, no
un fichero que alguien dejó en disco.

**Por qué.** Es la diferencia entre "tenemos un modelo" y "sabemos qué modelo
tenemos". Cada entrenamiento se registra como una **versión nueva**, se
compara con el campeón reinante sobre F1 ponderado en datos reservados, y
**solo se promueve si gana**. Si pierde, se queda como *challenger* y
producción no se toca.

```
entrenar ──▶ registrar versión ──▶ ¿supera al campeón en
                                    ensemble_test_weighted_f1?
                                         │
                            ┌────────────┴────────────┐
                           sí                         no
                            │                          │
                      etiqueta: champion        etiqueta: challenger
                      la API la descarga        producción intacta
```

**Qué cuesta.** Una dependencia de red en el arranque de la API. Si MLflow no
responde, hay que decidir qué hacer — y eso lleva a la decisión siguiente.

## 2.4 La degradación es deliberada

**La decisión.** Si MLflow está caído, la API **arranca igualmente**, carga el
modelo del directorio local y lo declara: `model_source: local-directory` en
`/health`. El entrenamiento también continúa sin registrar nada.

**Por qué.** Un servidor de seguimiento muerto no debe costar una ejecución de
varias horas, ni tumbar el servicio de predicciones. La observabilidad debe
degradarse; no debe llevarse la producción por delante.

**El detalle que lo hace honesto**: el camino degradado está **etiquetado de
forma distinta**. El panel de Grafana muestra "local fallback" en vez de un
número de versión. Esta misma tarde eso sirvió para detectar un fallo real —
la API llevaba horas sirviendo desde disco sin que nadie lo supiera, porque la
descarga del campeón fallaba por el timeout de gunicorn. Un *fallback*
silencioso habría escondido el problema; uno etiquetado lo delató.

## 2.5 Airflow orquesta y no calcula

**La decisión.** Los DAGs no importan TensorFlow, ni Evidently, ni el código
de entrenamiento. Cada tarea es una **llamada HTTP** a un servicio.

**Por qué.** Si Airflow importara el código de entrenamiento, habría que
instalar TensorFlow en su imagen — recreando exactamente el conflicto de
dependencias que obligó a partir el stack (§2.1). Y lo haría en silencio, al
construir la imagen, semanas después de que nadie recuerde por qué.

**Qué cuesta.** Dos consecuencias concretas, ambas visibles en el código:

- El sensor que espera a que termine un entrenamiento usa
  `mode="reschedule"`, que libera la ranura de ejecución entre comprobaciones.
  En modo `poke` ocuparía la única ranura del `SequentialExecutor` durante
  horas y nada más podría correr.
- El DAG de drift dispara el de entrenamiento con
  `wait_for_completion=False`: lo lanza y sigue, para que el chequeo horario
  no se atasque detrás de una ejecución de varias horas.

**Está protegido por tests.** `tests/test_dags.py` falla si un DAG importa
algo pesado, si el `Dockerfile.airflow` instala TensorFlow o Evidently, o si
un DAG llama a un endpoint que no existe.

## 2.6 Un solo disparador automático de reentrenamiento

**La decisión.** Airflow es el único camino automático al reentrenamiento. Las
alertas de Grafana informan y no actúan.

**Por qué.** Esta decisión se tomó *durante* el proyecto, corrigiendo un diseño
anterior. Grafana tenía un *webhook* apuntando a `/training/` que reentrenaba
cuando saltaba la alerta de drift — razonable cuando no había orquestador.
Al llegar Airflow, pasó a ser un **segundo disparador independiente** para la
misma acción, con otro horario, y que se activaba estuvieran los DAGs
pausados o no.

El endpoint `/training/` serializa las ejecuciones tras un *lock* (devuelve
409 si ya hay una en marcha), así que nunca chocaban. Pero eso **escondía** el
problema en vez de resolverlo: cuando arrancaba una ejecución, nada registraba
cuál de los dos la había lanzado.

> Dos disparadores automáticos para una misma acción no son redundancia:
> son ambigüedad.

**El detalle técnico que importa**: el *provisioning* de alerting de Grafana
es **aditivo**. Quitar el contact point del YAML no lo borra de la base de
datos de Grafana, que vive en un volumen que sobrevive a `docker compose
down`. Hizo falta un borrado explícito:

```yaml
deleteContactPoints:
  - orgId: 1
    uid: rakuten-retrain
resetPolicies:
  - 1
```

Verificado contra una instancia real: el contact point desapareció y las dos
reglas de alerta sobrevivieron.

## 2.7 SQLite

**La decisión.** SQLite para los datos del proyecto y para la metadata de
Airflow.

**Por qué.** El enunciado pide "una base de datos SQL (o NoSQL) en local". La
propia redacción —"SQL *o* NoSQL"— indica que no importa el motor, sino que
exista almacenamiento estructurado y persistente poblado por un script que
corre una vez. SQLite **es** una base de datos SQL completa: mismo lenguaje,
mismas garantías ACID, sin la complejidad de un servidor cliente-servidor.

Además encaja con el perfil de uso: un despliegue de un solo nodo y una base
que se lee mucho más de lo que se escribe.

**Qué cuesta.** No hay concurrencia de escritura real. La capa de acceso está
aislada en `db.py`, así que migrar a Postgres es un cambio de cadena de
conexión, no una reescritura.

**Ojo con una confusión frecuente**: Airflow avisa en su interfaz de que no se
use SQLite como metadata DB en producción. Eso se refiere a **su propia
contabilidad interna** (DagRuns, TaskInstances, usuarios), que no tiene nada
que ver con los datos de Rakuten. Migrar Airflow a Postgres no tocaría ni una
fila de la tabla `products`.

## 2.8 SequentialExecutor

**La decisión.** Airflow corre con `SequentialExecutor`: una tarea a la vez,
sin flota de *workers*.

**Por qué.** Es el tamaño correcto para un despliegue de un solo nodo con dos
DAGs y ejecuciones esporádicas. Un `CeleryExecutor` con workers separados
sería sobre-ingeniería en una máquina de 2 núcleos que ya corre siete
contenedores.

**Qué cuesta.** Cero paralelismo. Y es exactamente la razón por la que el
sensor usa `mode="reschedule"` (§2.5): con una sola ranura, un sensor que la
ocupe bloquea el sistema entero.

**La frase para la defensa**: pasar a `LocalExecutor` y Postgres es
*configuración, no rediseño*.

## 2.9 DVC sin Git

**La decisión.** DVC inicializado con `dvc init --no-scm`, y los hashes de los
datos registrados como *tags* de MLflow.

**Por qué.** Una ejecución de MLflow ya registra los parámetros del código y
las métricas resultantes, pero **no qué datos las produjeron**. Dos
ejecuciones con parámetros idénticos y métricas distintas son inexplicables
sin ese dato. Con el hash en la ejecución, cualquier discrepancia se puede
atribuir al código o a los datos.

**El detalle elegante**: `src/data_version.py` **lee a mano** los ficheros
`.dvc` (que son YAML con el md5, el tamaño y la ruta) en vez de importar la
librería `dvc`. Así el entorno de entrenamiento nunca necesita DVC instalado.
Es el mismo patrón que con MLflow: la herramienta pesada vive en su propio
sitio, y quien la consume solo lee el artefacto que produce.

**Qué cuesta el `--no-scm`.** Es lo que pide el enunciado, pero tiene un
precio operativo que conviene conocer porque no está documentado en ningún
sitio obvio: **DVC no escribe ninguno de los `.gitignore` que normalmente
escribiría**. Con Git, `dvc init` protege por su cuenta el caché, los datos
rastreados y el fichero de credenciales. Sin Git no protege nada, y hay tres
cosas que quedan expuestas a un `git add -A`: el caché (`.dvc/cache/`), los
ficheros temporales, y `.dvc/config.local` — que contiene el **token del
remoto en texto plano**. Las tres están ahora en el `.gitignore` del
repositorio, escritas a mano.

**Y una trampa de git que aparece al querer versionar los punteros.** Los
ficheros `.dvc` viven bajo `data/`, que estaba excluido entero. El patrón
obvio para re-incluirlos **no funciona**:

```
/data/
!data/preprocessed/*.dvc     <-- nunca llega a aplicarse
```

Git no puede re-incluir un fichero cuyo directorio padre está excluido, y
falla **en silencio**: no hay error, los ficheros simplemente no aparecen.
Hay que re-incluir cada nivel al bajar:

```
/data/*
!/data/preprocessed/
/data/preprocessed/*
!/data/preprocessed/*.dvc
```

## 2.9b El remoto: por qué DagsHub y no Google Drive

**La decisión.** Los datos se empujan a un remoto de DVC alojado en DagsHub,
con autenticación básica por token.

**Por qué no Drive, que era la opción obvia.** Google **bloquea la aplicación
OAuth por defecto de DVC**. No es un aviso que se pueda saltar con
"configuración avanzada": es un bloqueo duro, porque la app usa *restricted
scopes* (acceso completo al Drive del usuario) y DVC no ha conseguido pasar
la verificación de Google. Su propio mantenedor describe la situación como
"atascada en el limbo", sin arreglo ni fecha. La app necesita permisos
amplios porque no puede saber de antemano a qué carpeta vas a apuntar.

La única vía con Drive sería montar un proyecto propio en Google Cloud con
credenciales OAuth propias — media hora de trabajo, y con los tokens
caducando a los 7 días mientras la app esté en modo de pruebas.

**Por qué DagsHub resuelve el mismo problema mejor**: autenticación por
token en vez de OAuth, así que no hay pantalla de consentimiento ni
aplicación que un tercero pueda bloquear; 100 GB gratuitos por repositorio;
y endpoints estándar que DVC trata como cualquier otro remoto.

**Y dentro de DagsHub, el endpoint S3 y no el HTTP.** DagsHub expone el mismo
almacén por dos vías: `…/<repo>.dvc`, que DVC trata como un remoto HTTP, y
`…/<repo>.s3`, que trata como un remoto S3. Para tres CSV es indiferente.
Para 79.200 imágenes no lo es en absoluto, y el motivo está en una fase que
no se ve: antes de subir nada, DVC comprueba **qué objetos ya existen** en el
remoto, para no reenviar lo que esté puesto. Un remoto HTTP no tiene
operación de listado en bloque, así que esa comprobación se convierte en una
petición por objeto — 79.200 peticiones antes de empezar a subir. La barra se
queda en `0/?`, sin denominador porque no puede saber cuántas quedan, durante
horas. El endpoint S3 lista paginado: unas decenas de llamadas para el mismo
inventario.

El cambio es una línea de configuración más el paquete `dvc-s3` (que `dvc`
no instala por defecto: sin él, `dvc push` falla con `s3 is supported, but
requires 'dvc-s3' to be installed`). Y tiene una trampa: `dvc remote add`
sobre un remoto que ya existe reescribe la URL y **deja intactas las demás
claves**, que eran legales para un remoto HTTP e ilegales para uno S3. A
partir de ahí todos los comandos de DVC fallan con `extra keys not allowed @
data['remote']['origin']['auth']`, incluido el `dvc remote remove` con el que
se intentaría arreglar — porque DVC valida el fichero entero antes de
ejecutar nada. La salida es borrar `.dvc/config.local` y reescribirlo con
`dvc remote modify --local`.

**El detalle que separa configuración de secreto.** DVC parte la
configuración en dos ficheros a propósito: `.dvc/config` lleva la URL del
remoto y **sí se versiona**, para que quien clone sepa de dónde bajar los
datos; `.dvc/config.local` lleva las credenciales y **nunca se versiona**.
El comando que las escribe lo dice en su propio nombre: `dvc remote modify
--local`.

## 2.10 Los DAGs nacen pausados

**La decisión.** `AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: "True"`.

**Por qué.** Arrancar el stack no debe lanzar un entrenamiento de varias
horas por sorpresa. Quien quiera que corran, los despausa.

**Qué cuesta.** Hay que acordarse de despausarlos antes de la demo. Está en
la lista de comprobación de la Parte 5.

---

# Parte 3 — Las piezas, una a una

Cada herramienta en dos planos: qué es en general, y qué hace aquí.

## 3.1 Docker y Docker Compose

**Qué es.** Docker empaqueta una aplicación con todo su entorno —sistema
operativo mínimo, librerías, dependencias— en una **imagen** que corre igual
en cualquier máquina. Compose describe un conjunto de contenedores y cómo se
relacionan, en un solo fichero YAML.

**Qué hace aquí.** `docker-compose.yml` define los siete servicios, sus
puertos, sus variables de entorno, sus volúmenes y sus dependencias de
arranque. Es lo que convierte "instálate estas tres pilas de dependencias
incompatibles" en `docker compose up -d`.

Detalles que conviene conocer:

- **Volúmenes montados**: `./data`, `./models` y `./logs` se montan desde el
  host. Los ~2,4 GB de imágenes JPEG no tienen por qué estar dentro de una
  imagen de Docker. Consecuencia práctica: lo que escribas en `data/` en tu
  máquina lo ve el contenedor al instante, sin reconstruir nada.
- **`depends_on` con `condition: service_healthy`**: la API no arranca hasta
  que MLflow responde a su *healthcheck*.
- **Qué se copia y qué se monta**: `src/` se **copia** al construir la imagen.
  Por eso, un cambio en `src/drift_service.py` exige
  `docker compose up -d --build drift`, mientras que un cambio en un dashboard
  de Grafana se recoge solo.

## 3.2 FastAPI

**Qué es.** Un framework de Python para construir APIs HTTP. Su rasgo
distintivo es que usa las anotaciones de tipos de Python para validar las
peticiones automáticamente y generar documentación interactiva sin escribirla.

**Qué hace aquí.** Es el servicio que sirve el modelo. Endpoints:

| Endpoint | Método | Qué hace |
|---|---|---|
| `/predict/` | POST | Clasifica filas de un CSV; guarda cada predicción en SQLite |
| `/training/` | POST | Lanza un entrenamiento en segundo plano; devuelve 202 al instante |
| `/training/status` | GET | Estado de la ejecución en curso |
| `/model/reload` | POST | Vuelve a resolver el campeón del registry y lo carga |
| `/health` | GET | Vivo, modelo cargado, versión y origen |
| `/model-info` | GET | Detalle del modelo servido |
| `/metrics` | GET | Métricas en formato Prometheus |

Dos decisiones de diseño visibles aquí:

- **`/training/` devuelve 202, no 200.** Una ejecución completa tarda horas;
  una petición HTTP no debe quedarse abierta tanto tiempo. El trabajo ocurre
  en un hilo de fondo y quien llama consulta `/training/status`.
- **`/model/reload` existe por un motivo concreto.** La API recoge un campeón
  nuevo automáticamente al arrancar y tras un entrenamiento que *ella* ejecutó.
  Ninguno de los dos cubre una ejecución lanzada fuera del servicio — alguien
  corriendo `training.py` a mano, o un compañero promoviendo una versión desde
  la interfaz de MLflow. Sin este endpoint, la única forma de recogerlo sería
  reiniciar el contenedor, que recarga VGG16 desde cero y tumba el servicio
  mientras tanto.

**Dónde verlo.** <http://localhost:8000/docs> — documentación interactiva
generada automáticamente.

## 3.3 SQLite

**Qué es.** Una base de datos SQL que vive en un único fichero, sin servidor.
La librería se enlaza dentro del propio programa.

**Qué hace aquí.** Tres tablas en `data/rakuten.db`:

- **`products`** — los datos de Rakuten, cargados una vez por
  `src/data/import_to_db.py`. Es el **lado de referencia** de la detección de
  drift: contra esto se compara el tráfico actual.
- **`predictions`** — cada predicción servida, con su texto, su clase
  predicha, su confianza y la versión del modelo. Es el **lado actual** de la
  comparación.
- **`drift_reports`** — el resumen de cada chequeo de drift.

El detalle que explica el diseño: **`/predict/` guarda cada predicción**. Sin
eso no habría nada contra lo que comparar, y la detección de drift sería
imposible. El guardado está envuelto en un `try/except` deliberado: un fallo
al escribir una fila de monitorización nunca debe convertir una predicción
correcta en una petición fallida.

## 3.4 DVC

**Qué es.** *Data Version Control*. Git versiona código, pero no funciona bien
con ficheros de gigabytes. DVC resuelve eso guardando en el repositorio un
**puntero** (un fichero `.dvc` pequeño, con el hash md5, el tamaño y la ruta)
mientras el dato real vive en otro sitio.

**Qué hace aquí.** Inicializado sin Git (`dvc init --no-scm`), como pide el
enunciado, así que los ficheros `.dvc` son el registro en sí. `training.py`
lee esos hashes mediante `src/data_version.py` y los registra como *tags* de
la ejecución de MLflow, con el prefijo `data.` (por ejemplo
`data.X_train_update.csv`).

Lo rastreado son los tres CSV y el directorio de imágenes de entrenamiento.
El remoto está en DagsHub, de modo que quien clone el repositorio obtiene los
punteros con el código y recupera los datos con `dvc pull`.

**Para qué sirve en la práctica.** Para responder a "¿por qué esta ejecución
dio 0,71 y aquella 0,68 con los mismos parámetros?". Si los hashes coinciden,
la diferencia está en el código o en la aleatoriedad; si no coinciden, está
en los datos.

**Un detalle de instalación que cuesta una tarde si no se sabe.** DVC 3.55.2
declara su dependencia de `pathspec` **sin tope superior**, así que `pip`
resuelve a `pathspec` 1.x — que eliminó el símbolo privado `_DIR_MARK` que
importa el módulo de ignores de DVC. El resultado es que **todos** los
comandos de DVC fallan idénticos, antes de hacer nada:

```
ERROR: unexpected error - cannot import name '_DIR_MARK'
from 'pathspec.patterns.gitwildmatch'
```

Se arregla fijando `pathspec==0.12.1`.

El segundo caso es más instructivo, porque el arreglo correcto cambió con el
entorno. El extra de Drive arrastra un `pyOpenSSL` de 2022 que choca con las
versiones modernas de `cryptography` y falla con `module 'lib' has no
attribute 'GEN_EMAIL'`. La respuesta inmediata es fijar `pyopenssl==24.2.1`,
que exige `cryptography < 44`. Pero al instalar `dvc-s3` en Python 3.14, pip
resuelve `asyncssh` a una versión que exige `cryptography >= 48.0.1`, y las
dos restricciones no tienen solución común — ninguna versión publicada de
`pyOpenSSL` acepta `cryptography` 50.

La salida no es un pin mejor: es que **nada del entorno depende de
`pyOpenSSL`**. El árbol inverso de dependencias lo deja a la vista — estaba
ahí como residuo del intento con Drive, que se descartó. Se desinstala y el
conflicto desaparece. La lección general: antes de buscar la combinación de
versiones que concilia dos restricciones, conviene preguntarse si el paquete
en conflicto hace falta.

Los pines y esta nota están en `scripts/setup_dvc.sh`.

**Y DVC vive en su propio entorno**, nunca junto al de entrenamiento: dvc 3.x
sube `typing_extensions` por encima del `< 4.6` que exige TensorFlow 2.13. Es
el mismo principio que separa los siete contenedores (§2.1), aplicado al
instalar una herramienta local.

## 3.5 TensorFlow / Keras

**Qué es.** La librería con la que se definen y entrenan las redes neuronales.
Keras es su interfaz de alto nivel.

**Qué hace aquí.** Construye y entrena las dos ramas del modelo, y ejecuta las
predicciones. `src/training.py` orquesta el proceso completo: entrena la rama
de texto, entrena la rama de imagen, busca los pesos de la mezcla, evalúa
sobre datos reservados, y registra todo en MLflow.

**Un apunte de rendimiento relevante para la demo**: cada predicción implica
una pasada hacia adelante de VGG16 en CPU. `/predict/` es **lento por
naturaleza** — no es un problema que arreglar, es el coste de clasificar
imágenes sin GPU. Hay que decirlo antes de que alguien lo pregunte.

## 3.6 MLflow

**Qué es.** Dos cosas que conviene no mezclar:

1. **Tracking** — un cuaderno de laboratorio. Cada ejecución registra sus
   parámetros, sus métricas y sus artefactos, y quedan comparables entre sí.
2. **Model Registry** — un catálogo de modelos con versiones y etiquetas, que
   responde a "¿cuál es el modelo que está en producción ahora mismo?".

**Qué hace aquí.** Ambas cosas.

- Cada entrenamiento crea una ejecución en el experimento
  `rakuten-classification`, con los hiperparámetros, las métricas y un
  **paquete de artefactos autocontenido** (pesos, tokenizador, mapeo de
  clases, pesos de la mezcla).
- Esa ejecución se registra como una versión nueva del modelo
  `rakuten-fusion`.
- `promote_if_better()` la compara con el campeón sobre
  `ensemble_test_weighted_f1`. Solo si gana **estrictamente**, se le pone la
  etiqueta `champion`; si no, se queda como `challenger`.
- La API descarga los artefactos del campeón al arrancar.

**Dos detalles operativos.**

- **`--serve-artifacts`**: el servidor de MLflow hace de proxy para los
  artefactos, de modo que los clientes no necesitan acceso directo al
  almacenamiento. El coste: los ficheros grandes pasan por ese proxy, y por
  eso hizo falta `--gunicorn-opts "--timeout 300"`. Con el valor por defecto
  de 30 s, la descarga de los pesos de VGG16 moría a mitad y la API caía
  silenciosamente al directorio local.
- **`mlflow-skinny` vs el servidor completo**: el cliente ligero se puede
  instalar junto a TensorFlow; el servidor completo no (es la pila de
  SQLAlchemy/Alembic la que inicia el conflicto).

**Dónde verlo.** <http://localhost:5000> — experimentos, ejecuciones,
métricas comparadas y el registro de modelos con sus etiquetas.

## 3.7 Evidently

**Qué es.** Una librería para detectar *drift*: el fenómeno por el cual los
datos que llegan a un modelo en producción dejan de parecerse a los datos con
los que se entrenó. Compara dos conjuntos —referencia y actual— columna a
columna, con tests estadísticos, y dice cuáles se han movido.

**Qué hace aquí.** Vive en su propio servicio (`src/drift_service.py`,
puerto 8100) con un único endpoint de trabajo: `POST /run`.

Compara tres columnas:

| Columna | Qué compara | Qué significa que se mueva |
|---|---|---|
| `text_length` | longitud del texto | los productos que llegan tienen descripciones de otra longitud |
| `word_count` | número de palabras | lo mismo, desde otro ángulo |
| `prdtypecode` | etiquetas **reales** en referencia vs **predichas** en actual | la mezcla de salidas se ha movido |

**Una distinción que hay que decir en voz alta antes de que la pregunten**:
las dos primeras son *drift de entrada* y la tercera es *drift de predicción*.
Son medidas distintas. Y el **drift de verdad fundamental (*ground truth*) no
es medible aquí**, porque las predicciones en producción no tienen etiqueta
real. Nadie nos dice si acertamos.

El umbral de acción es **0,5**: si más de la mitad de las columnas
monitorizadas han derivado, se pide reentrenar.

## 3.8 Prometheus

**Qué es.** Una base de datos de series temporales que funciona por
**scraping**: en vez de que las aplicaciones le envíen datos, Prometheus va
periódicamente a un endpoint HTTP de cada aplicación y se lleva lo que
encuentra. Guarda cada valor con su marca de tiempo y permite consultarlo con
un lenguaje propio, **PromQL**.

**Qué hace aquí.** Raspa `/metrics` de la API. Retención: 15 días.

Las métricas que la API expone:

| Métrica | Tipo | Qué mide |
|---|---|---|
| `rakuten_http_requests_total` | Counter | peticiones, por ruta y código de estado |
| `rakuten_http_request_duration_seconds` | Histogram | latencia por ruta |
| `rakuten_predictions_total` | Counter | predicciones, por clase predicha |
| `rakuten_prediction_confidence` | Histogram | confianza de cada predicción |
| `rakuten_model_version` | Gauge | versión servida (0 = fallback local) |
| `rakuten_model_loaded` | Gauge | 1 si hay modelo cargado |
| `rakuten_training_in_progress` | Gauge | 1 durante un entrenamiento |
| `rakuten_training_runs_total` | Counter | ejecuciones terminadas, por resultado |
| `rakuten_drift_detected` | Gauge | último veredicto de Evidently |
| `rakuten_drift_share_of_drifted_columns` | Gauge | proporción de columnas derivadas |
| `rakuten_drift_report_age_seconds` | Gauge | antigüedad del último informe |
| `rakuten_predictions_stored_total` | Gauge | filas en la tabla `predictions` |

**Un detalle que explica comportamientos confusos en la demo**: los
**Counters y Histograms viven en la memoria del proceso** y se ponen a cero
cuando el contenedor se reinicia. Los **Gauges que se leen de SQLite**
(`rakuten_predictions_stored_total`, los de drift) **sobreviven**, porque se
recalculan desde la base de datos en cada *scrape*. Si tras un reinicio ves
"0 peticiones" pero "97 predicciones almacenadas", no es una incoherencia: son
dos mecanismos distintos.

## 3.9 Grafana

**Qué es.** La capa de visualización. Se conecta a fuentes de datos
(Prometheus, aquí) y dibuja paneles. También evalúa reglas de alerta.

**Qué hace aquí.** Dos dashboards y dos alertas, todo **provisionado desde
ficheros**: existen en el momento en que arranca el stack, nadie los dibuja a
mano. La Parte 6 los recorre panel por panel.

**Un matiz importante tras el cambio de §2.6**: las alertas **informan y no
actúan**. Se evalúan, se ven en la interfaz, y no llaman a nada.

## 3.10 Airflow

**Qué es en general.** Un orquestador de flujos de trabajo. Se describen
tareas y sus dependencias en un grafo dirigido acíclico (**DAG**), y Airflow
se encarga de ejecutarlas en el orden correcto, en el momento correcto,
reintentando lo que falla y dejando registro de todo.

Lo que aporta frente a un `cron`:

- **Dependencias explícitas** — "esto solo si aquello terminó bien".
- **Reintentos con política** — número y espaciado configurables.
- **Visibilidad** — una interfaz donde se ve qué corrió, cuándo, cuánto tardó
  y por qué falló, con los logs de cada tarea.
- **Backfill** — ejecutar el pasado si hiciera falta.
- **Disparo entre DAGs** — un flujo puede lanzar otro.

**Qué hace aquí.** Dos DAGs, y hacen deliberadamente poco: cada tarea es una
llamada HTTP (§2.5).

### `rakuten_training` — domingos a las 03:00, y cuando el drift lo pida

```
start_training ──▶ wait_for_training ──▶ report_registry_decision ──▶ reload_champion
POST /training/    GET /training/status   GET /training/status         POST /model/reload
202, no espera     sensor, reschedule     lee el veredicto             la API sirve
                                          del registry                 el campeón nuevo
```

### `rakuten_drift_check` — cada hora

```
run_drift_check ──▶ drift_above_threshold ──▶ trigger_training
POST /run           ShortCircuitOperator       TriggerDagRunOperator
en el servicio      para aquí si no se         wait_for_completion=False
de drift            supera el 0,5
```

Los tres operadores que aparecen, explicados:

- **`PythonOperator`** — ejecuta una función de Python. Aquí, siempre una que
  hace una petición HTTP.
- **`PythonSensor`** — espera a que una condición se cumpla, comprobándola
  cada cierto tiempo. En `mode="reschedule"` libera la ranura entre
  comprobaciones en vez de bloquearla.
- **`ShortCircuitOperator`** — si su función devuelve falso, salta todo lo que
  venga después. Es el "si no hay drift, no hagas nada".
- **`TriggerDagRunOperator`** — lanza otro DAG.

**Dónde verlo.** <http://localhost:8080> (admin / admin). Recuerda que nacen
pausados.

## 3.11 Streamlit

**Qué es.** Una librería que convierte un script de Python en una aplicación
web, sin escribir HTML ni JavaScript.

**Qué hace aquí.** La interfaz de demostración, con cuatro secciones:

1. **Overview & architecture** — el diagrama, cómo llega un modelo a
   producción, y la tabla de los cuatro fallos encontrados en el repositorio
   de partida.
2. **Live prediction** — clasificar filas y ver el resultado con su
   `model_version`.
3. **Model & registry** — versión servida, contador de predicciones, y un
   botón para lanzar un entrenamiento.
4. **Monitoring** — el último informe de drift y las métricas operativas.

Importante: **Streamlit no carga el modelo**. Llama a la API por HTTP, como
cualquier otro cliente. Es coherente con el principio de §2.5.

## 3.12 GitHub Actions

**Qué es.** La integración continua de GitHub: cada push ejecuta
automáticamente lo que se le indique.

**Qué hace aquí.** Dos trabajos:

- **`lint-and-test`** — linter y la batería completa de tests.
- **`build-images`** — construye las cuatro imágenes propias (`api`,
  `streamlit`, `drift`, `airflow`) en paralelo, para comprobar que los
  Dockerfiles siguen siendo válidos.

**Qué protegen los tests.** 56 en total. Más allá de lo obvio, fijan
decisiones que de otro modo se desharían en silencio:

- Que las rutas de entrenamiento y de servicio no vuelvan a divergir (los
  cuatro fallos del repositorio original).
- Que ningún DAG importe TensorFlow, Evidently o el código de entrenamiento.
- Que ningún contact point de Grafana llame al endpoint de entrenamiento.
- Que `POST /run` acepte cuerpo vacío.

Ese último merece un comentario, porque ilustra la filosofía: `DriftRequest`
declara sus tres campos con valor por defecto, pero FastAPI hace obligatorio
el cuerpo cuando el parámetro es un modelo Pydantic sin default propio. El
resultado era un 422 ante el comando más obvio del mundo. El DAG nunca lo
pisaba, porque siempre manda los tres campos — así que el único sitio donde
iba a aparecer era un terminal, en directo, el día de la defensa.

---

# Parte 4 — Cómo circula todo

Dos recorridos completos siguiendo un dato concreto.

## 4.1 El viaje de una predicción

1. Alguien hace `POST /predict/` con una ruta a un CSV, una ruta a las
   imágenes y un límite de filas.
2. La API comprueba que hay modelo cargado. Si no, 503.
3. Valida el límite contra `RAKUTEN_MAX_PREDICT_ROWS` (200). Si se pasa, 422.
4. Lee el CSV **desde dentro del contenedor**. Este punto es crítico y
   explica un fallo real: la ruta debe estar en un directorio montado
   (`data/`). Un fichero en el directorio temporal del sistema del host es
   invisible para el contenedor, y produce un 404.
5. Para cada fila: compone el texto, lo tokeniza, carga la imagen, la
   preprocesa, ejecuta ambas ramas, combina con los pesos de la mezcla.
6. Incrementa `rakuten_predictions_total` por clase y observa la confianza.
7. **Guarda cada predicción en SQLite**, con sus rasgos de texto calculados.
   Envuelto en `try/except`: un fallo aquí no rompe la respuesta.
8. Devuelve las predicciones junto con `model_version` y `model_source`.

Mientras tanto, un *middleware* ha contado la petición y medido su latencia.
Prometheus se llevará ambas cosas en su siguiente *scrape*.

## 4.2 El ciclo completo: del drift al modelo nuevo

```
 1. Tráfico real llega a /predict/
         │
         ▼
 2. Cada predicción se guarda en la tabla `predictions`
         │
         ▼
 3. Cada hora, rakuten_drift_check hace POST /run en el servicio de drift
         │
         ▼
 4. Evidently compara:
      referencia = muestra aleatoria de `products` (datos de entrenamiento)
      actual     = las N predicciones más recientes
    sobre text_length, word_count y prdtypecode
         │
         ▼
 5. Escribe un informe HTML, registra las métricas en MLflow,
    guarda el resumen en SQLite
         │
         ▼
 6. ShortCircuitOperator: ¿proporción de columnas derivadas >= 0,5?
         │
    ┌────┴────┐
   no         sí
    │          │
  para    7. TriggerDagRunOperator lanza rakuten_training
               │
               ▼
          8. POST /training/ → 202 inmediato
               │
               ▼
          9. El sensor espera (en modo reschedule, sin bloquear)
               │
               ▼
         10. training.py entrena, registra la ejecución en MLflow
             con los hashes de DVC, y registra una versión nueva
               │
               ▼
         11. promote_if_better: ¿supera al campeón en F1 ponderado?
               │
          ┌────┴────┐
         no         sí
          │          │
    challenger   champion
    producción       │
    intacta          ▼
                12. POST /model/reload: la API descarga los artefactos
                    nuevos y los sirve
                         │
                         ▼
                13. rakuten_model_version cambia en Grafana
```

**La frase que resume la Parte 4**: ningún paso de este ciclo es una persona.

---

# Parte 5 — La demo

## 5.1 Antes de empezar

Con tiempo, no sobre la marcha:

- [ ] `docker compose up -d` — y esperar a que los siete estén `healthy`.
      Compruébalo con `docker compose ps`. La API tarda **unos ocho minutos**
      en abrir el puerto: carga las dos ramas de Keras antes de escuchar.
      Hasta entonces el chequeo se rechaza al instante («Could not connect»,
      `after 0 ms`), que parece una caída y no lo es. No midas antes de
      tiempo, y levanta el stack con tiempo de sobra antes de la defensa.
- [ ] Ejecutar desde una ventana de `cmd` o PowerShell propia, **no** desde
      el terminal integrado de un editor. Compose dibuja su progreso con
      códigos que mueven el cursor, y hay terminales que no repintan la
      línea final: el comando ha terminado y parece colgado. Si no hay
      alternativa, `set COMPOSE_PROGRESS=plain` desactiva ese dibujado.
- [ ] Saber que `docker compose up -d` **no devuelve el control enseguida**.
      Los servicios con `condition: service_healthy` hacen que Compose espere
      al chequeo de `mlflow` antes de arrancar a quienes dependen de él. No
      es un cuelgue y no hay que cortarlo: cortarlo deja contenedores a medio
      crear.
- [ ] Abrir una a una las siete direcciones en el navegador. Si alguna no
      responde con el servicio `healthy`, prueba `127.0.0.1` en lugar de
      `localhost` (ver la nota al final de la Parte 6) y, si funciona, usa
      esa forma durante toda la demo. Averiguarlo en directo cuesta un
      minuto que no tienes.
- [ ] Despausar los dos DAGs en Airflow (<http://localhost:8080>,
      admin / admin) y lanzar cada uno una vez a mano, para que la vista de
      cuadrícula tenga historial que enseñar.
- [ ] Verificar que la API sirve desde el registry, no desde disco:
      `curl.exe http://localhost:8000/health` debe decir
      `"model_source":"mlflow-registry"`.
- [ ] Mandar algo de tráfico normal, para que los dashboards no estén vacíos.
- [ ] Tener abiertas las pestañas: Streamlit, MLflow, Airflow, los dos
      dashboards de Grafana.
- [ ] Confirmar que los datos están subidos al remoto: `dvc status -c` desde
      el entorno de DVC debe decir que no falta nada. Si quedan ficheros por
      subir, `dvc push` continúa donde iba.
- [ ] Comprobar que la última ejecución de MLflow lleva los tags `data.*`. Si
      no los lleva, es que se entrenó antes de configurar DVC: lanza un
      entrenamiento pequeño por la API y ya los tendrá.
- [ ] Ensayarlo entero dos veces, con cronómetro.

## 5.2 El guion, con comandos

Todos los comandos desde la raíz del repositorio, en `cmd`.

### Paso 1 — Una predicción en vivo (Streamlit)

<http://localhost:8501> → **Live prediction** → clasificar 10 filas.

Señala el `model_version` de la respuesta: *la API sirve lo que el registry
bendijo, no un fichero que alguien dejó en disco*.

Tarda unos 20 segundos en CPU. Dilo mientras corre, no después.

### Paso 2 — El registro de modelos (MLflow)

<http://localhost:5000> → experimento `rakuten-classification`.

Enseña varias ejecuciones comparadas, los *tags* con los hashes de DVC, y el
modelo `rakuten-fusion` con sus versiones y la etiqueta `champion`.

El punto a transmitir: una versión se promueve **solo si gana**; si pierde se
queda como `challenger` y producción no se toca.

### Paso 3 — Tráfico normal

```cmd
python scripts\simulate_traffic.py --mode normal --batches 6 --batch-size 20
```

120 predicciones repartidas en seis tandas separadas 30 segundos, para que los
paneles de Grafana dibujen una **línea temporal** en vez de un pico
instantáneo. Unos 5-8 minutos en una máquina de 2 núcleos.

Mientras corre, enseña el dashboard de **API health** llenándose.

### Paso 4 — Tráfico desplazado

```cmd
python scripts\simulate_traffic.py --mode shifted --batches 6 --batch-size 20
```

El modo `shifted` restringe el tráfico a unas pocas categorías y recorta el
texto a tres palabras, vaciando la descripción. Mueve a la vez las dos
familias de columnas monitorizadas: `prdtypecode` (la mezcla de salidas se
estrecha) y `text_length` / `word_count` (el texto es deliberadamente corto).

De dónde salen esos datos, por si lo preguntan: `training.py` solo muestrea
unos cientos de filas por clase de las ~84.000 de `products`. Todo lo demás
nunca ha sido tocado por ningún entrenamiento — es un conjunto reservado de
facto, sin preparación adicional.

### Paso 5 — El chequeo de drift, ya

```cmd
curl.exe -X POST http://localhost:8100/run -H "Content-Type: application/json" -d "{\"current_limit\":120,\"reference_limit\":300,\"log_to_mlflow\":false}"
```

**Por qué con esos parámetros y no a secas.** El POST hace seis cosas
síncronas: muestrea la referencia con `ORDER BY RANDOM()` (que obliga a
recorrer y ordenar las ~84.000 filas), calcula los rasgos de texto fila a
fila, ejecuta los tests de Evidently, escribe un HTML de varios MB, **lo sube
a MLflow** y guarda el resumen. Con `log_to_mlflow: false` te saltas la
subida, que es el paso más lento; con `reference_limit: 300` recortas el
muestreo. La diferencia en directo puede ser entre diez segundos y dos minutos
de silencio incómodo.

`current_limit: 120` acota la ventana a las 120 predicciones que acabas de
mandar. Es tráfico desplazado puro, y el drift sale limpio.

**El truco que demuestra criterio.** Lánzalo también con la ventana por
defecto:

```cmd
curl.exe -X POST http://localhost:8100/run -H "Content-Type: application/json" -d "{\"current_limit\":500,\"log_to_mlflow\":false}"
```

Con 500 filas la ventana incluye el tráfico normal anterior, la señal se
diluye y puede no cruzar el umbral. Eso **no es un fallo**: es el tamaño de
ventana importando. En producción, con miles de predicciones por hora, 500
filas son unos minutos; aquí son varias sesiones — o sea, "actual" dejaría de
significar actual.

### Paso 6 — Airflow cerrando el círculo

<http://localhost:8080> → vista de cuadrícula.

Recorre las tres tareas de `rakuten_drift_check` y enseña que cada una es una
llamada HTTP. Explica por qué: si Airflow importara el código de
entrenamiento, habría que meter TensorFlow en su imagen y volvería el
conflicto de dependencias que obligó a partir el stack.

Si quieres enseñar un entrenamiento de verdad, lánzalo con configuración
pequeña desde "Trigger DAG w/ config":

```json
{"samples_per_class": 50, "epochs_lstm": 1, "epochs_vgg": 1}
```

### Paso 7 — Los dashboards

Los dos, con el recorrido de la Parte 6.

## 5.3 Qué decir cuando algo tarda

- **`/predict/` tarda** → "cada fila ejecuta una pasada de VGG16 en CPU; es
  lento por naturaleza, y por eso el endpoint de entrenamiento devuelve 202 en
  vez de mantener la conexión abierta".
- **Streamlit va lento durante la simulación** → "el contenedor de la API está
  consumiendo los dos núcleos con las inferencias; es contención de CPU, no un
  fallo".
- **Un panel dice "No data"** → Parte 6.4.

---

# Parte 6 — Los dashboards de Grafana, panel por panel

<http://localhost:3000> — admin / admin. Ambos se refrescan cada 30 segundos
y muestran la última hora por defecto.

## 6.1 Dashboard «API health»

Tráfico, latencia, errores y disponibilidad del servicio.

| Panel | Qué muestra | Cómo leerlo |
|---|---|---|
| **Requests / s by route** | Peticiones por segundo, separadas por ruta | Permite distinguir el tráfico de predicción del de consulta de estado |
| **p95 latency by route** | El percentil 95 de latencia, por ruta | El 95 % de las peticiones tardan menos que esto. `/predict/` es lento por naturaleza: vigílalo contra sí mismo, no contra una referencia de aplicación web |
| **Error rate** | Proporción de respuestas 4xx y 5xx | Verde por debajo del 5 %, naranja hasta el 20 %, rojo por encima |
| **Model loaded** | READY o NOT LOADED | La API arranca aunque no haya modelo; esta es la señal real de disponibilidad |
| **Training in progress** | TRAINING o idle | Activo mientras corre un entrenamiento en segundo plano |
| **Responses by status** | Peticiones por segundo, por código de estado | Dónde aparecen los errores cuando aparecen |
| **Completed training runs** | Ejecuciones terminadas por hora, por resultado | Solo cuenta las lanzadas **a través de la API**; ver 6.4 |
| **Total requests served** | Acumulado desde el último arranque del contenedor | Se pone a cero al reiniciar: es un contador en memoria |

## 6.2 Dashboard «Model & data drift»

El estado del modelo y la relación entre lo que ve ahora y aquello con lo que
se entrenó.

| Panel | Qué muestra | Cómo leerlo |
|---|---|---|
| **Dataset drift** | DRIFT / NO DRIFT | El último veredicto de Evidently |
| **Drifted columns** | Proporción de columnas derivadas | El umbral de reentrenamiento es 0,5. Con tres columnas monitorizadas, los valores posibles son 0 %, 33,3 %, 66,7 % y 100 % — por eso "33,3 %" significa exactamente "una de tres" |
| **Drift report age** | Antigüedad del último informe | Si sube sin parar, el monitor ha dejado de correr |
| **Serving model version** | Número de versión del registry | **0 significa que el registry no respondió y se está sirviendo desde el directorio local.** No es un error ruidoso, es una degradación etiquetada |
| **Predicted class distribution** | Predicciones acumuladas por clase | Un colapso sobre muy pocas clases es la primera señal visible de que algo va mal. Las etiquetas son los códigos numéricos de Rakuten; el dataset no trae nombres. Hay 27 posibles, así que el panel está acotado aunque la consulta no lo limite |
| **Mean prediction confidence** | Confianza media | Cerca de 1/27 ≈ 0,037 significaría que el modelo apenas supera al azar entre 27 clases |
| **Predictions / s** | Ritmo de predicciones | Sube durante la simulación de tráfico |
| **Predictions stored for monitoring** | Filas en la tabla `predictions` | Es el conjunto "actual" que usa el chequeo de drift. **Sobrevive a los reinicios**, porque se lee de SQLite |

## 6.3 Las dos alertas

Ambas provisionadas desde fichero, visibles en Alerting → Alert rules.

- **«Data drift above retraining threshold»** — se dispara cuando
  `rakuten_drift_share_of_drifted_columns > 0.5` de forma sostenida durante
  5 minutos. Mismo umbral que usa el DAG, leído de la misma métrica: lo que
  una persona ve dispararse aquí es exactamente la condición sobre la que
  actúa el orquestador.
- **«Drift monitor has stopped reporting»** — se dispara si no hay informe en
  24 horas. Esta importa tanto como la primera: **un monitor de drift que ha
  dejado de correr tiene exactamente el mismo aspecto que un sistema sano**.
  Su silencio es, en sí mismo, la alerta.

**Las dos informan; ninguna actúa.** Es una decisión, no un olvido (§2.6).

## 6.4 Cuando un panel dice «No data»

Dos casos distintos, y conviene no confundirlos:

- **Serie que nunca ha existido.** Si no se ha producido jamás un error HTTP,
  la serie de errores está vacía — y dividir por un vector vacío en PromQL da
  "No data", no cero. Se resuelve envolviendo el numerador con `or vector(0)`,
  que es lo que hace ahora el panel de *Error rate*.
- **Métrica real que todavía vale cero.** *Completed training runs* solo se
  incrementa cuando un entrenamiento se lanza **a través de `POST /training/`**.
  Si todos los entrenamientos se han hecho con `python src/training.py` desde
  la línea de comandos, el contador está legítimamente vacío. No es un fallo:
  es una métrica que mide exactamente lo que dice medir.

---

# Parte 7 — Preguntas probables

**«¿Por qué no usáis PostgreSQL?»**
Para los datos, porque el enunciado pide "SQL o NoSQL en local" y SQLite es
una base SQL completa que encaja con un despliegue de un nodo. Para la
metadata de Airflow, porque esa base es su contabilidad interna y no tiene
relación con los datos del proyecto. En ambos casos, migrar es configuración,
no rediseño: la capa de acceso está aislada en `db.py`.

**«Un compañero tenía un modelo con 0,9151 de F1. ¿Por qué no está en
producción?»**
Porque se juzgan en ejes distintos: 4,5 GB de pesos y varios segundos por
predicción frente a 150 MB y menos de un segundo. Para un proyecto evaluado
por criterios de MLOps, el ligero es la opción defendible. El sitio correcto
para el fuerte es el Model Registry, como versión competidora — que es
exactamente la comparación para la que existe un registry.

**«¿Por qué Airflow hace tan poco?»**
Porque hacer más implicaría instalar TensorFlow y Evidently en su imagen, y
eso recrearía el conflicto de dependencias que obligó a separar los servicios.
Airflow decide **cuándo** y en **qué orden**; los servicios deciden **cómo**.
Hay tests que fallan si un DAG importa algo pesado.

**«Tenéis alertas de drift. ¿Por qué no disparan el reentrenamiento?»**
Lo hacían, hasta que existió Airflow. Un webhook de Grafana llamaba a
`/training/`. Al llegar el orquestador, eso pasó a ser un segundo disparador
automático para la misma acción, con otro horario, activo aunque los DAGs
estuvieran pausados. El lock de `/training/` impedía que chocaran, lo cual
escondía la ambigüedad en vez de resolverla: cuando arrancaba una ejecución,
nada registraba cuál la había lanzado. Las alertas informan, Airflow actúa.

**«¿Qué pasa si MLflow se cae?»**
La API arranca igualmente, sirve desde el directorio local y lo declara en
`/health` como `model_source: local-directory`. El entrenamiento continúa sin
registrar. Ambas cosas son deliberadas: la observabilidad debe degradarse, no
llevarse la producción por delante. Y el camino degradado está **etiquetado**,
que es lo que permitió detectar un fallo real durante el desarrollo.

**«¿Cómo sabéis que el registry se está usando de verdad?»**
`/health` y `/model-info` reportan la versión y su origen, y el log de arranque
muestra la descarga de artefactos. El camino de respaldo es visible
precisamente porque está etiquetado de otra manera.

**«¿Qué significa exactamente "drift" aquí?»**
Tres columnas. `text_length` y `word_count` son la misma magnitud a ambos
lados: si se mueven, los productos que llegan ya no se parecen a los del
entrenamiento. `prdtypecode` compara etiquetas **reales** en la referencia con
**predichas** en la ventana actual: si se mueve, la mezcla de salidas ha
cambiado, lo que puede significar que cambió el tráfico o que el modelo se
degradó. Es un síntoma, no un diagnóstico. Y el drift de verdad fundamental
no es medible: las predicciones en producción no tienen etiqueta.

**«¿Cómo comparte el equipo la versión exacta de los datos?»**
Los ficheros `.dvc` están versionados en el repositorio y el remoto está en
DagsHub, así que quien clone hace `dvc pull` y recupera exactamente los
mismos datos: los tres CSV y las imágenes. Lo que viaja en el repositorio es
la URL del remoto (`.dvc/config`), no las credenciales — esas van en
`.dvc/config.local`, que está ignorado. Cada persona usa su propio token.

**«¿Por qué DagsHub y no Google Drive?»**
Porque Google bloquea la aplicación OAuth por defecto de DVC. No es un aviso
que se pueda saltar: la app pide acceso completo al Drive del usuario y DVC
no ha logrado pasar la verificación de Google; su mantenedor lo describe como
"atascado en el limbo". La alternativa era montar un proyecto propio en
Google Cloud, con tokens que caducan a los 7 días en modo de pruebas. DagsHub
usa autenticación por token, sin pantalla de consentimiento ni aplicación que
un tercero pueda bloquear, y da 100 GB gratis.

**«¿Cómo escalaríais esto?»**
La API no guarda estado salvo el modelo que carga, así que escala
horizontalmente tras un balanceador. El cuello de botella real es la pasada de
VGG16 en CPU: antes que replicar, un nodo con GPU o una capa de *batching*.

**«¿Por qué `SequentialExecutor` si Airflow avisa de que no es para
producción?»**
Porque es el tamaño correcto para dos DAGs con ejecuciones esporádicas en un
nodo de 2 núcleos que ya corre siete contenedores. Pasar a `LocalExecutor` y
Postgres es configuración, no rediseño.

---

# Parte 8 — Limitaciones conocidas

Enunciadas sin adornos, porque una limitación reconocida vale más que una
escondida.

1. **El drift de verdad fundamental no se puede medir.** Las predicciones en
   producción no llevan etiqueta, así que la exactitud real es inobservable.
   Cerrar ese hueco exigiría un circuito de retroalimentación —que alguien
   corrija o confirme las clasificaciones— y queda fuera del alcance.

2. **Airflow corre con `SequentialExecutor` y SQLite.** Una tarea a la vez,
   sin flota de workers. Correcto para una demo de un nodo, incorrecto para
   cualquier cosa paralela.

3. **Los DAGs nacen pausados.** Es deliberado —arrancar el stack no debe
   lanzar un entrenamiento de horas— pero significa que hay que acordarse de
   despausarlos.

4. **Los artefactos del modelo están versionados en dos sitios.** El Model
   Registry de MLflow es la fuente de verdad, pero `models/` también está
   rastreado en git (la línea `/models/` del `.gitignore` está comentada), de
   modo que cada entrenamiento ensucia el repositorio. Es una tensión real
   pendiente de resolver — y la respuesta natural es DVC, que ya está
   montado: los artefactos del modelo son exactamente la clase de fichero
   grande y binario para la que existe.

5. **DVC versiona los datos, no el pipeline.** Hay punteros y hashes, pero no
   hay `dvc.yaml` ni `dvc repro`. Es deliberado: un pipeline de DVC sería un
   **segundo orquestador** conviviendo con Airflow, decidiendo cuándo se
   reentrena con otro criterio. Es la misma trampa de los dos disparadores
   automáticos (§2.6), y Airflow ya tiene ese trabajo.

6. **La máquina de desarrollo es el cuello de botella.** Dos núcleos, siete
   contenedores. Los dos fallos de timeout encontrados y corregidos durante el
   desarrollo son consecuencia directa de eso, y podría aparecer un tercero.

7. **Tres tests se saltan en el entorno de desarrollo local.** Uno necesita
   Airflow y dos necesitan Evidently, que viven a propósito en sus propias
   imágenes. No es un hueco de cobertura: es el aislamiento de dependencias
   asomando en la batería de tests. El CI sí los ejecuta, en los contenedores
   donde esas dependencias existen.

---

## Referencia rápida de URLs

| Qué | Dónde | Credenciales |
|---|---|---|
| API (documentación interactiva) | <http://localhost:8000/docs> | — |
| Streamlit | <http://localhost:8501> | — |
| MLflow | <http://localhost:5000> | — |
| Servicio de drift | <http://localhost:8100/docs> | — |
| Airflow | <http://localhost:8080> | admin / admin |
| Prometheus | <http://localhost:9090> | — |
| Grafana | <http://localhost:3000> | admin / admin |

**Si una dirección no responde.** Comprueba primero que el servicio está
`healthy` (`docker compose ps`); si lo está y el navegador no carga, prueba
`127.0.0.1` en lugar de `localhost`.

El motivo: los servidores dentro de los contenedores escuchan en IPv4
(`--host 0.0.0.0`). En un equipo donde `localhost` resuelve primero a la
dirección IPv6 `::1` — lo normal en Windows —, la petición puede no llegar
aunque el servicio esté perfectamente vivo. Cambiar la dirección en la URL
es todo el arreglo; no hay que tocar nada del stack.

No afecta a los *healthchecks* de los contenedores, que usan `127.0.0.1`
desde dentro precisamente por esto.
