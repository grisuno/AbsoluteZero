# AbsoluteZero

![image](https://github.com/user-attachments/assets/e28927b0-203b-447c-99db-a7c60fa547f5)

## Autor

[Gris Iscomeback](https://github.com/grisuno)

Correo electrónico: grisun0[at]proton[dot]me

Fecha de creación: 19/05/2025

Licencia: [GPL v3](https://www.gnu.org/licenses/gpl-3.0.en.html)

## Descripción

AbsoluteZero es un sistema avanzado de razonamiento basado en inteligencia artificial diseñado para analizar código fuente. Utiliza un enfoque de aprendizaje continuo y curricular para mejorar su capacidad de deducción, abducción e inducción a partir de diferentes lenguajes de programación.

## Características Principales

* **Análisis de Código Inteligente:** Capacidad para procesar y comprender código en múltiples lenguajes (Python, JavaScript, C++, Java).
* **Razonamiento Adaptativo:** Genera tareas de análisis de código con dificultad adaptable ("basic", "intermediate", "advanced", "expert").
* **Banco de Memoria Persistente:** Almacena tareas y soluciones aprendidas para su reutilización y mejora continua.
* **Aprendizaje Curricular:** Ajusta automáticamente la dificultad de las tareas en función del rendimiento.
* **Interfaz de Línea de Comandos (CLI):** Interfaz interactiva para analizar directorios de código y visualizar el estado del sistema.
* **Análisis Paralelo (Opcional):** Permite acelerar el análisis de grandes proyectos de código utilizando threading.
* **Métricas Detalladas:** Proporciona estadísticas sobre las tareas procesadas, el rendimiento y el tiempo de ejecución.
* **Auto-Juego (Experimental):** Incluye un sistema para entrenamiento a través de torneos internos (ELO ratings para tareas).

## Requisitos

* Python 3.x
* Librerías Python:
    * `requests`
    * `collections`
    * `cmd2`
    * `argparse`
    * `rich`
    * `numpy`
    * `datetime`
    * `hashlib`
    * `typing`
    * `threading`
    * `time`
    * `dataclasses`
    * `pickle`
* Servidor local de DeepSeek ejecutándose en `http://localhost:11434`. El modelo predeterminado es `deepseek-r1:1.5b`.

## Instalación

1.  Clona el repositorio:
    ```bash
    git clone [https://github.com/grisuno/AbsoluteZero](https://github.com/grisuno/AbsoluteZero)
    cd AbsoluteZero
    ```
2.  Instala las dependencias Python:
    ```bash
    pip install -r requirements.txt  # (Crea un archivo requirements.txt con las dependencias)
    ```

## Uso

1.  Ejecuta la aplicación:
    ```bash
    python app.py
    ```
2.  Utiliza los comandos disponibles en la interfaz CLI:
    * `analyze --code-dir <directorio>`: Analiza el código fuente en el directorio especificado. Puedes usar la opción `--iterations` para controlar el número de iteraciones y `--parallel` para habilitar el procesamiento paralelo.
    * `stats`: Muestra las estadísticas actuales del sistema.
    * `curriculum`: Muestra el estado del aprendizaje curricular.
    * `save_memory`: Guarda manualmente el estado actual del banco de memoria.
    * `help`: Muestra la lista de comandos disponibles.
    * `exit`: Cierra la aplicación.

### Ejemplo de Análisis

Para analizar un directorio llamado `mi_proyecto`:

```bash
(AbsoluteZero) analyze --code-dir mi_proyecto --iterations 50

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Y8Y2Z73AV)
```

El modelo experimental utiliza tu historial de la Búsqueda Algunas funciones no están disponibles.
Markdown

# AbsoluteZero

## Autor

[Gris Iscomeback](https://github.com/grisuno)

Correo electrónico: grisiscomeback[at]gmail[dot]com

Fecha de creación: xx/xx/xxxx

Licencia: [GPL v3](https://www.gnu.org/licenses/gpl-3.0.en.html)

## Descripción

AbsoluteZero es un sistema avanzado de razonamiento basado en inteligencia artificial diseñado para analizar código fuente. Utiliza un enfoque de aprendizaje continuo y curricular para mejorar su capacidad de deducción, abducción e inducción a partir de diferentes lenguajes de programación.

## Características Principales

* **Análisis de Código Inteligente:** Capacidad para procesar y comprender código en múltiples lenguajes (Python, JavaScript, C++, Java).
* **Razonamiento Adaptativo:** Genera tareas de análisis de código con dificultad adaptable ("basic", "intermediate", "advanced", "expert").
* **Banco de Memoria Persistente:** Almacena tareas y soluciones aprendidas para su reutilización y mejora continua.
* **Aprendizaje Curricular:** Ajusta automáticamente la dificultad de las tareas en función del rendimiento.
* **Interfaz de Línea de Comandos (CLI):** Interfaz interactiva para analizar directorios de código y visualizar el estado del sistema.
* **Análisis Paralelo (Opcional):** Permite acelerar el análisis de grandes proyectos de código utilizando threading.
* **Métricas Detalladas:** Proporciona estadísticas sobre las tareas procesadas, el rendimiento y el tiempo de ejecución.
* **Auto-Juego (Experimental):** Incluye un sistema para entrenamiento a través de torneos internos (ELO ratings para tareas).

## Requisitos

* Python 3.x
* Librerías Python:
    * `requests`
    * `collections`
    * `cmd2`
    * `argparse`
    * `rich`
    * `numpy`
    * `datetime`
    * `hashlib`
    * `typing`
    * `threading`
    * `time`
    * `dataclasses`
    * `pickle`
* Servidor local de DeepSeek ejecutándose en `http://localhost:11434`. El modelo predeterminado es `deepseek-r1:1.5b`.

## Instalación

1.  Clona el repositorio:
    ```bash
    git clone [https://github.com/grisuno/AbsoluteZero](https://github.com/grisuno/AbsoluteZero)
    cd AbsoluteZero
    ```
2.  Instala las dependencias Python:
    ```bash
    pip install -r requirements.txt  # (Crea un archivo requirements.txt con las dependencias)
    ```

## Uso

1.  Ejecuta la aplicación:
    ```bash
    python app.py
    ```
2.  Utiliza los comandos disponibles en la interfaz CLI:
    * `analyze --code-dir <directorio>`: Analiza el código fuente en el directorio especificado. Puedes usar la opción `--iterations` para controlar el número de iteraciones y `--parallel` para habilitar el procesamiento paralelo.
    * `stats`: Muestra las estadísticas actuales del sistema.
    * `curriculum`: Muestra el estado del aprendizaje curricular.
    * `save_memory`: Guarda manualmente el estado actual del banco de memoria.
    * `help`: Muestra la lista de comandos disponibles.
    * `exit`: Cierra la aplicación.

### Ejemplo de Análisis

Para analizar un directorio llamado `mi_proyecto`:

```bash
(AbsoluteZero) analyze --code-dir mi_proyecto --iterations 50
```

Para analizar el mismo directorio en paralelo con 100 iteraciones:

```Bash

(AbsoluteZero) analyze --code-dir mi_proyecto --iterations 100 --parallel
```
## Configuración
Las siguientes variables se pueden configurar directamente en el script app.py:

- DEEPSEEK_API_URL: URL de la API de DeepSeek.
- DEEPSEEK_MODEL: Modelo de DeepSeek a utilizar.
- SOURCE_FILE_EXTENSIONS: Lista de extensiones de archivo consideradas como código fuente.
- BATCH_SIZE, MAX_BUFFER_SIZE, NUM_REFERENCES, NUM_ROLLOUTS, ITERATIONS, MIN_REWARD_THRESHOLD, DIVERSITY_THRESHOLD: Parámetros para controlar el comportamiento del sistema.
- DIFFICULTY_LEVELS, COMPLEXITY_METRICS, difficulty_thresholds: Configuración para el aprendizaje curricular y la dificultad de las tareas.

## Contribuciones
Las contribuciones son bienvenidas. Si encuentras algún problema o tienes sugerencias para mejorar AbsoluteZero, por favor, crea un "issue" o envía un "pull request" en el repositorio de GitHub.

## Licencia
Este proyecto está bajo la licencia GPL v3. Consulta el archivo LICENSE para obtener más detalles.

## Agradecimientos
Gracias a los creadores del paper Absolute Zero: Reinforced Self-play Reasoning with Zero Data por su gran investigación
Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu,  Yang Yue,  Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, and Gao Huang,
