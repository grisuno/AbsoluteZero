#!/usr/bin/env python3
# _*_ coding: utf8 _*_
"""
app.py

Autor: Gris Iscomeback
Correo electrónico: grisiscomeback[at]gmail[dot]com
Fecha de creación: xx/xx/xxxx
Licencia: GPL v3

Descripción:  
"""
import os
import json
import random
import re
import logging
import requests
from collections import deque, defaultdict
from cmd2 import Cmd, with_argparser
import argparse
from rich.console import Console
from rich.progress import Progress, TaskID
import numpy as np
from datetime import datetime
import hashlib
from typing import Dict, List, Any, Optional, Tuple
import threading
import time
from dataclasses import dataclass, asdict
import pickle

# Configuración de logging mejorada
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('absolute_zero.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuración global mejorada
DEEPSEEK_API_URL = "http://localhost:11434/api/generate"
DEEPSEEK_MODEL = "deepseek-r1:1.5b"
CONSOLE = Console()
SOURCE_FILE_EXTENSIONS = [".py", ".js", ".cpp", ".java"]  # Expandido para más lenguajes
BATCH_SIZE = 16
MAX_BUFFER_SIZE = 1000
NUM_REFERENCES = 5  # Incrementado para mejor contexto
NUM_ROLLOUTS = 3  # Reducido para eficiencia
ITERATIONS = 10
MIN_REWARD_THRESHOLD = 0.3  # Solo mantener tareas con buen rendimiento
DIVERSITY_THRESHOLD = 0.8  # Para evitar duplicados similares

# Configuración de dificultad adaptativa
DIFFICULTY_LEVELS = ["basic", "intermediate", "advanced", "expert"]
COMPLEXITY_METRICS = ["cyclomatic", "cognitive", "lines_of_code"]

@dataclass
class Task:
    """Estructura mejorada para tareas con metadatos."""
    program: str
    input: Any
    output: Any
    task_type: str
    difficulty: str = "basic"
    complexity_score: float = 0.0
    creation_time: datetime = None
    success_count: int = 0
    attempt_count: int = 0
    hash: str = None
    
    def __post_init__(self):
        if self.creation_time is None:
            self.creation_time = datetime.now()
        if self.hash is None:
            self.hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Computa hash único para detectar duplicados."""
        content = f"{self.program}{self.input}{self.output}{self.task_type}"
        return hashlib.md5(content.encode()).hexdigest()
    
    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.attempt_count)
    
    def to_dict(self) -> Dict:
        return asdict(self)

class MemoryBank:
    """Sistema de memoria mejorado con persistencia y clustering."""
    
    def __init__(self, persist_path: str = "memory_bank.pkl"):
        self.persist_path = persist_path
        self.tasks_by_type: Dict[str, List[Task]] = defaultdict(list)
        self.task_embeddings: Dict[str, np.ndarray] = {}
        self.performance_history: List[Dict] = []
        self.load_from_disk()
    
    def add_task(self, task: Task) -> bool:
        """Añade tarea si no es duplicada y cumple criterios de calidad."""
        # Verificar duplicados
        if any(existing.hash == task.hash for existing in self.tasks_by_type[task.task_type]):
            logger.info(f"Tarea duplicada rechazada: {task.hash[:8]}")
            return False
        
        # Verificar diversidad usando embeddings simples
        if self._is_too_similar(task):
            logger.info(f"Tarea muy similar rechazada: {task.hash[:8]}")
            return False
        
        self.tasks_by_type[task.task_type].append(task)
        self._maintain_buffer_size(task.task_type)
        self.save_to_disk()
        return True
    
    def _is_too_similar(self, new_task: Task) -> bool:
        """Verifica si la tarea es muy similar a las existentes."""
        if len(self.tasks_by_type[new_task.task_type]) < 5:
            return False
        
        # Embedding simple basado en longitud y estructura del programa
        new_embedding = self._simple_embedding(new_task.program)
        
        for existing_task in self.tasks_by_type[new_task.task_type][-10:]:  # Solo últimas 10
            existing_embedding = self._simple_embedding(existing_task.program)
            similarity = np.dot(new_embedding, existing_embedding) / (
                np.linalg.norm(new_embedding) * np.linalg.norm(existing_embedding) + 1e-8
            )
            if similarity > DIVERSITY_THRESHOLD:
                return True
        return False
    
    def _simple_embedding(self, program: str) -> np.ndarray:
        """Crea embedding simple del programa."""
        features = [
            len(program),
            program.count('def'),
            program.count('if'),
            program.count('for'),
            program.count('while'),
            program.count('return'),
        ]
        return np.array(features, dtype=float)
    
    def _maintain_buffer_size(self, task_type: str):
        """Mantiene el tamaño del buffer removiendo tareas con peor rendimiento."""
        if len(self.tasks_by_type[task_type]) > MAX_BUFFER_SIZE:
            # Ordenar por success_rate y mantener las mejores
            self.tasks_by_type[task_type].sort(key=lambda t: t.success_rate, reverse=True)
            self.tasks_by_type[task_type] = self.tasks_by_type[task_type][:MAX_BUFFER_SIZE]
    
    def get_reference_tasks(self, task_type: str, n: int) -> List[Task]:
        """Obtiene tareas de referencia con muestreo inteligente."""
        available = self.tasks_by_type[task_type]
        if len(available) <= n:
            return available
        
        # Muestreo estratificado: mejores tareas + diversidad
        best_tasks = sorted(available, key=lambda t: t.success_rate, reverse=True)[:n//2]
        diverse_tasks = random.sample(available, min(n//2, len(available) - len(best_tasks)))
        return best_tasks + diverse_tasks
    
    def save_to_disk(self):
        """Guarda el banco de memoria en disco."""
        try:
            with open(self.persist_path, 'wb') as f:
                pickle.dump(self.tasks_by_type, f)
        except Exception as e:
            logger.error(f"Error guardando memoria: {e}")
    
    def load_from_disk(self):
        """Carga el banco de memoria desde disco."""
        try:
            if os.path.exists(self.persist_path):
                with open(self.persist_path, 'rb') as f:
                    self.tasks_by_type = pickle.load(f)
                logger.info(f"Memoria cargada: {sum(len(tasks) for tasks in self.tasks_by_type.values())} tareas")
        except Exception as e:
            logger.error(f"Error cargando memoria: {e}")

class CurriculumLearning:
    """Sistema de aprendizaje curricular que ajusta dificultad automáticamente."""
    
    def __init__(self):
        self.current_difficulty = "basic"
        self.performance_window = deque(maxlen=20)
        self.difficulty_thresholds = {
            "basic": 0.7,
            "intermediate": 0.6,
            "advanced": 0.5,
            "expert": 0.4
        }
    
    def update_performance(self, reward: float):
        """Actualiza rendimiento y ajusta dificultad."""
        self.performance_window.append(reward)
        if len(self.performance_window) >= 10:
            avg_performance = np.mean(self.performance_window)
            self._adjust_difficulty(avg_performance)
    
    def _adjust_difficulty(self, avg_performance: float):
        """Ajusta dificultad basada en rendimiento."""
        current_idx = DIFFICULTY_LEVELS.index(self.current_difficulty)
        threshold = self.difficulty_thresholds[self.current_difficulty]
        
        if avg_performance > threshold and current_idx < len(DIFFICULTY_LEVELS) - 1:
            self.current_difficulty = DIFFICULTY_LEVELS[current_idx + 1]
            logger.info(f"Dificultad aumentada a: {self.current_difficulty}")
        elif avg_performance < threshold * 0.7 and current_idx > 0:
            self.current_difficulty = DIFFICULTY_LEVELS[current_idx - 1]
            logger.info(f"Dificultad reducida a: {self.current_difficulty}")

class AbsoluteZeroCmd(Cmd):
    """Sistema Absolute Zero mejorado con características avanzadas."""
    
    def __init__(self):
        super().__init__()
        self.prompt = "(AbsoluteZero) "
        self.memory_bank = MemoryBank()
        self.curriculum = CurriculumLearning()
        self.processed_files = set()
        self.learning_rate = 1e-4  # Ajustado
        self.model_weights = defaultdict(float)
        self.session_stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'avg_reward': 0.0,
            'start_time': datetime.now()
        }
        self._initialize_with_seed()
        CONSOLE.print("[INIT] Absolute Zero Reasoner v2.0 inicializado", style="bold green")

    def _initialize_with_seed(self):
        """Inicializa con triplete semilla mejorado."""
        seed_task = Task(
            program='def f(x):\n    return x * 2',
            input='5',
            output='10',
            task_type='deduction',
            difficulty='basic'
        )
        for task_type in ['deduction', 'abduction', 'induction']:
            task_copy = Task(**{**seed_task.to_dict(), 'task_type': task_type})
            self.memory_bank.add_task(task_copy)

    # --- Comandos CLI Mejorados ---

    analyze_parser = argparse.ArgumentParser()
    analyze_parser.add_argument('--code-dir', type=str, required=True, help='Directorio con código')
    analyze_parser.add_argument('--iterations', type=int, default=ITERATIONS, help='Número de iteraciones')
    analyze_parser.add_argument('--parallel', action='store_true', help='Procesamiento paralelo')

    @with_argparser(analyze_parser)
    def do_analyze(self, args):
        """Analiza código con opciones avanzadas."""
        CONSOLE.print(f"[ANALYZE] Analizando {args.code_dir} - {args.iterations} iteraciones", style="bold yellow")
        
        if args.parallel:
            self._analyze_parallel(args.code_dir, args.iterations)
        else:
            self._analyze_sequential(args.code_dir, args.iterations)

    def do_stats(self, arg):
        """Muestra estadísticas del sistema."""
        total_tasks = sum(len(tasks) for tasks in self.memory_bank.tasks_by_type.values())
        runtime = datetime.now() - self.session_stats['start_time']
        
        CONSOLE.print("\n[STATS] Estadísticas del Sistema", style="bold blue")
        CONSOLE.print(f"Tareas totales en memoria: {total_tasks}")
        CONSOLE.print(f"Dificultad actual: {self.curriculum.current_difficulty}")
        CONSOLE.print(f"Tiempo de ejecución: {runtime}")
        CONSOLE.print(f"Tareas exitosas: {self.session_stats['successful_tasks']}/{self.session_stats['total_tasks']}")
        
        for task_type, tasks in self.memory_bank.tasks_by_type.items():
            avg_success = np.mean([t.success_rate for t in tasks]) if tasks else 0
            CONSOLE.print(f"  {task_type}: {len(tasks)} tareas, {avg_success:.2%} éxito promedio")

    def do_curriculum(self, arg):
        """Muestra estado del curriculum learning."""
        CONSOLE.print(f"\n[CURRICULUM] Dificultad actual: {self.curriculum.current_difficulty}", style="bold magenta")
        if self.curriculum.performance_window:
            recent_perf = np.mean(list(self.curriculum.performance_window)[-5:])
            CONSOLE.print(f"Rendimiento reciente: {recent_perf:.3f}")

    def do_save_memory(self, arg):
        """Guarda manualmente el banco de memoria."""
        self.memory_bank.save_to_disk()
        CONSOLE.print("[SAVE] Memoria guardada exitosamente", style="green")

    # --- Funciones de Análisis Mejoradas ---

    def _analyze_sequential(self, directory: str, iterations: int):
        """Análisis secuencial mejorado."""
        files = self._get_code_files(directory)
        
        with Progress() as progress:
            task_id = progress.add_task("[cyan]Analizando archivos...", total=len(files))
            
            for file_path in files:
                if file_path not in self.processed_files:
                    self.processed_files.add(file_path)
                    self._analyze_code_file(file_path, iterations)
                progress.update(task_id, advance=1)

    def _analyze_parallel(self, directory: str, iterations: int):
        """Análisis paralelo con threading."""
        files = self._get_code_files(directory)
        threads = []
        
        for file_path in files:
            if file_path not in self.processed_files:
                thread = threading.Thread(
                    target=self._analyze_code_file,
                    args=(file_path, iterations)
                )
                threads.append(thread)
                thread.start()
        
        for thread in threads:
            thread.join()

    def _get_code_files(self, directory: str) -> List[str]:
        """Obtiene lista de archivos de código."""
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in SOURCE_FILE_EXTENSIONS):
                    files.append(os.path.join(root, filename))
        return files

    def _analyze_code_file(self, file_path: str, iterations: int):
        """Análisis mejorado de archivo de código."""
      
        with open(file_path, 'r', encoding='utf-8') as file:
            code_content = file.read()
            logger.info(f"Analizando {file_path}")
            self._absolute_zero_loop(code_content, file_path, iterations)


    # --- Ciclo Absolute Zero Mejorado ---

    def _absolute_zero_loop(self, code_content: str, file_path: str, iterations: int):
        """Ciclo principal mejorado con métricas."""
        for iteration in range(iterations):
            logger.info(f"Iteración {iteration + 1}/{iterations} para {file_path}")
            
            # Proponer tareas con dificultad adaptativa
            tasks = self._propose_tasks_adaptive(code_content, file_path)
            
            # Resolver tareas con validación mejorada
            solutions = self._solve_tasks_improved(tasks, file_path)
            
            # Actualizar modelo y curriculum
            self._update_model_improved(tasks, solutions)
            
            # Actualizar curriculum con rendimiento promedio
            avg_reward = self._calculate_average_reward(solutions)
            self.curriculum.update_performance(avg_reward)

    def _propose_tasks_adaptive(self, code_content: str, file_path: str) -> Dict[str, List[Task]]:
        """Propone tareas con dificultad adaptativa."""
        tasks = {'deduction': [], 'abduction': [], 'induction': []}
        
        for task_type in ['deduction', 'abduction', 'induction']:
            # Obtener referencias con muestreo inteligente
            ref_tasks = self.memory_bank.get_reference_tasks(task_type, NUM_REFERENCES)
            
            # Construir prompt con dificultad adaptativa
            prompt = self._build_adaptive_propose_prompt(
                task_type, code_content, ref_tasks, self.curriculum.current_difficulty
            )
            
            response = self._query_deepseek_improved(prompt)
            if response:
                try:
                    task_data = json.loads(response)
                    task = Task(**task_data, task_type=task_type, difficulty=self.curriculum.current_difficulty)
                    
                    if self._validate_task_improved(task):
                        tasks[task_type].append(task)
                        self.memory_bank.add_task(task)
                        self.session_stats['total_tasks'] += 1
                        logger.info(f"Tarea válida {task_type} añadida")
                except json.JSONDecodeError as e:
                    logger.error(f"Error parseando tarea {task_type}: {e}")
        
        return tasks

    def _build_adaptive_propose_prompt(self, task_type: str, code_content: str, 
                                     ref_tasks: List[Task], difficulty: str) -> str:
        """Construye prompt adaptativo basado en dificultad."""
        ref_examples = [task.to_dict() for task in ref_tasks[-3:]]  # Últimas 3 mejores
        ref_str = json.dumps(ref_examples, indent=2)
        
        difficulty_instructions = {
            "basic": "Create simple, straightforward tasks with clear logic.",
            "intermediate": "Create moderately complex tasks with some edge cases.",
            "advanced": "Create complex tasks with multiple conditions and optimizations.",
            "expert": "Create highly sophisticated tasks with algorithmic challenges."
        }
        
        code_snippet = code_content[:500] if code_content else "# No code provided"
        
        return f"""
System: Generate a {difficulty} difficulty {task_type} task. {difficulty_instructions[difficulty]}

Reference examples:
{ref_str}

Code context:
```python
{code_snippet}
```

Return only valid JSON with keys: program, input, output.
Ensure the program is deterministic and safe.
"""

    def _query_deepseek_improved(self, prompt: str) -> str:
        """Consulta API con manejo de errores mejorado y reintentos."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                headers = {'Content-Type': 'application/json'}
                data = {
                    "model": DEEPSEEK_MODEL,
                    "prompt": prompt,
                    "stream": False,  # Simplificado para mejor reliability
                    "temperature": 0.7,
                    "max_tokens": 512
                }
                
                response = requests.post(
                    DEEPSEEK_API_URL,
                    json=data,
                    headers=headers,
                    timeout=3000
                )
                response.raise_for_status()
                
                result = response.json()
                if 'response' in result:
                    return result['response'].strip()
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Intento {attempt + 1} falló: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Backoff exponencial
                
        logger.error("Todos los intentos fallaron")
        return ""

    def _validate_task_improved(self, task: Task) -> bool:
        """Validación mejorada con más verificaciones."""
        try:
            # Verificaciones de seguridad
            unsafe_patterns = ['import os', 'import sys', 'exec(', 'eval(', '__import__']
            if any(pattern in task.program for pattern in unsafe_patterns):
                return False
            
            # Verificación de ejecutabilidad
            if not self._is_executable(task.program):
                return False
            
            # Verificación específica por tipo de tarea
            if task.task_type == 'deduction':
                return self._validate_deduction_task(task)
            elif task.task_type == 'abduction':
                return self._validate_abduction_task(task)
            elif task.task_type == 'induction':
                return self._validate_induction_task(task)
                
            return True
        except Exception as e:
            logger.error(f"Error en validación: {e}")
            return False

    def _is_executable(self, program: str) -> bool:
        """Verifica si el programa es ejecutable."""
        try:
            compile(program, '<string>', 'exec')
            return True
        except SyntaxError:
            return False

    def _validate_deduction_task(self, task: Task) -> bool:
        """Valida tarea de deducción."""
        try:
            exec_globals = {}
            exec(task.program, exec_globals)
            func_name = self._extract_function_name(task.program)
            if func_name not in exec_globals:
                return False
            func = exec_globals[func_name]
            result = func(eval(str(task.input)))
            return str(result) == str(task.output)
        except:
            return False

    def _validate_abduction_task(self, task: Task) -> bool:
        """Valida tarea de abducción."""
        try:
            exec_globals = {}
            exec(task.program, exec_globals)
            func_name = self._extract_function_name(task.program)
            func = exec_globals[func_name]
            result = func(eval(str(task.input)))
            return str(result) == str(task.output)
        except:
            return False

    def _validate_induction_task(self, task: Task) -> bool:
        """Valida tarea de inducción."""
        try:
            if not isinstance(task.input, list):
                return False
            exec_globals = {}
            exec(task.program, exec_globals)
            func_name = self._extract_function_name(task.program)
            func = exec_globals[func_name]
            for pair in task.input:
                result = func(eval(str(pair['input'])))
                if str(result) != str(pair['output']):
                    return False
            return True
        except:
            return False

    def _extract_function_name(self, program: str) -> str:
        """Extrae nombre de función del programa."""
        match = re.search(r'def\s+(\w+)\s*\(', program)
        return match.group(1) if match else 'f'

    def _solve_tasks_improved(self, tasks: Dict[str, List[Task]], file_path: str) -> Dict[str, List[Dict]]:
        """Resolución mejorada de tareas con métricas detalladas."""
        solutions = {'deduction': [], 'abduction': [], 'induction': []}
        
        for task_type, task_list in tasks.items():
            for task in task_list:
                start_time = time.time()
                
                # Construir prompt de resolución
                solve_prompt = self._build_solve_prompt_improved(task)
                response = self._query_deepseek_improved(solve_prompt)
                
                if response:
                    try:
                        solution = self._extract_solution_improved(response, task.task_type)
                        reward = self._calculate_reward_improved(task, solution)
                        
                        # Actualizar estadísticas de la tarea
                        task.attempt_count += 1
                        if reward > MIN_REWARD_THRESHOLD:
                            task.success_count += 1
                            self.session_stats['successful_tasks'] += 1
                        
                        solve_time = time.time() - start_time
                        
                        solutions[task_type].append({
                            'task': task,
                            'solution': solution,
                            'reward': reward,
                            'solve_time': solve_time
                        })
                        
                        logger.info(f"Tarea {task_type} resuelta: recompensa={reward:.3f}, tiempo={solve_time:.2f}s")
                        
                    except Exception as e:
                        logger.error(f"Error resolviendo tarea {task_type}: {e}")
        
        return solutions

    def _build_solve_prompt_improved(self, task: Task) -> str:
        """Construye prompt mejorado de resolución."""
        base_prompts = {
            'deduction': f"""
Task: Execute the following Python code with the given input and return the output.

Code:
```python
{task.program}
```

Input: {task.input}

Return only the output value, no explanations.
""",
            'abduction': f"""
Task: Find an input that produces the given output when run through this code.

Code:
```python
{task.program}
```

Output: {task.output}

Return only a valid input value that would produce this output.
""",
            'induction': f"""
Task: Write a Python function that satisfies all the given input-output pairs.

Input-Output Pairs:
{json.dumps(task.input, indent=2)}

Additional context: {task.output}

Return only the Python function definition.
"""
        }
        return base_prompts.get(task.task_type, "")

    def _extract_solution_improved(self, response: str, task_type: str) -> str:
        """Extrae solución con múltiples estrategias."""
        # Intentar extraer de bloques de código primero
        code_match = re.search(r'```(?:python)?\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Intentar extraer líneas que parecen respuestas
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                return line
        
        # Fallback: retornar respuesta completa limpia
        return response.strip()

    def _calculate_reward_improved(self, task: Task, solution: str) -> float:
        """Cálculo mejorado de recompensa con múltiples factores."""
        try:
            base_reward = 0.0
            
            if task.task_type == 'deduction':
                base_reward = self._verify_deduction_solution(task, solution)
            elif task.task_type == 'abduction':
                base_reward = self._verify_abduction_solution(task, solution)
            elif task.task_type == 'induction':
                base_reward = self._verify_induction_solution(task, solution)
            
            # Factores adicionales de recompensa
            complexity_bonus = min(0.2, task.complexity_score / 100)  # Bonus por complejidad
            efficiency_bonus = 0.1 if len(solution) < 100 else 0.0    # Bonus por concisión
            
            final_reward = base_reward + complexity_bonus + efficiency_bonus
            return min(1.0, max(0.0, final_reward))  # Clamp entre 0 y 1
            
        except Exception as e:
            logger.error(f"Error calculando recompensa: {e}")
            return 0.0

    def _verify_deduction_solution(self, task: Task, solution: str) -> float:
        """Verifica solución de deducción."""
        try:
            exec_globals = {}
            exec(task.program, exec_globals)
            func_name = self._extract_function_name(task.program)
            func = exec_globals[func_name]
            expected = func(eval(str(task.input)))
            return 1.0 if str(expected) == solution.strip() else 0.0
        except:
            return 0.0

    def _verify_abduction_solution(self, task: Task, solution: str) -> float:
        """Verifica solución de abducción."""
        try:
            exec_globals = {}
            exec(task.program, exec_globals)
            func_name = self._extract_function_name(task.program)
            func = exec_globals[func_name]
            result = func(eval(solution))
            return 1.0 if str(result) == str(task.output) else 0.0
        except:
            return 0.0

    def _verify_induction_solution(self, task: Task, solution: str) -> float:
        """Verifica solución de inducción."""
        try:
            exec_globals = {}
            exec(solution, exec_globals)
            func_name = self._extract_function_name(solution)
            func = exec_globals[func_name]
            
            correct = 0
            total = len(task.input)
            
            for pair in task.input:
                try:
                    result = func(eval(str(pair['input'])))
                    if str(result) == str(pair['output']):
                        correct += 1
                except:
                    continue
            
            return correct / total if total > 0 else 0.0
        except:
            return 0.0

    def _calculate_average_reward(self, solutions: Dict[str, List[Dict]]) -> float:
        """Calcula recompensa promedio de todas las soluciones."""
        all_rewards = []
        for task_type_solutions in solutions.values():
            for solution_data in task_type_solutions:
                all_rewards.append(solution_data['reward'])
        return np.mean(all_rewards) if all_rewards else 0.0

    def _update_model_improved(self, tasks: Dict[str, List[Task]], solutions: Dict[str, List[Dict]]):
        """Actualización mejorada del modelo con normalización adaptativa."""
        logger.info("Actualizando modelo...")
        
        for task_type in ['deduction', 'abduction', 'induction']:
            if not solutions[task_type]:
                continue
            
            rewards = [s['reward'] for s in solutions[task_type]]
            
            # Normalización robusta con clipping
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards) + 1e-8
            advantages = np.clip(
                [(r - mean_reward) / std_reward for r in rewards],
                -3.0, 3.0  # Clipping para estabilidad
            )
            
            # Actualización de pesos con momentum
            momentum = 0.9
            for idx, (task, advantage) in enumerate(zip(tasks[task_type], advantages)):
                task_key = f"{task_type}_{task.hash[:8]}"
                old_weight = self.model_weights[task_key]
                gradient = self.learning_rate * advantage
                
                # Aplicar momentum
                self.model_weights[task_key] = momentum * old_weight + gradient
                
            # Actualizar estadísticas de sesión
            self.session_stats['avg_reward'] = mean_reward
            logger.info(f"Modelo actualizado para {task_type}: μ={mean_reward:.3f}, σ={std_reward:.3f}")

class SelfPlayTraining:
    """Sistema de entrenamiento por auto-juego avanzado."""
    
    def __init__(self, memory_bank: MemoryBank, cmd_instance: AbsoluteZeroCmd):
        self.memory_bank = memory_bank
        self.cmd = cmd_instance
        self.tournament_history = []
        self.elo_ratings = defaultdict(lambda: 1000)  # Sistema ELO para tareas
        
    def run_tournament(self, rounds: int = 10):
        """Ejecuta torneo entre diferentes versiones del modelo."""
        CONSOLE.print(f"[TOURNAMENT] Iniciando torneo con {rounds} rondas", style="bold magenta")
        
        for round_num in range(rounds):
            logger.info(f"Ronda de torneo {round_num + 1}/{rounds}")
            
            # Seleccionar tareas para el torneo
            tournament_tasks = self._select_tournament_tasks()
            
            # Evaluar modelo actual vs versiones anteriores
            results = self._evaluate_tournament_round(tournament_tasks)
            
            # Actualizar ratings ELO
            self._update_elo_ratings(results)
            
            # Guardar resultados
            self.tournament_history.append({
                'round': round_num + 1,
                'results': results,
                'timestamp': datetime.now()
            })
            
        self._display_tournament_summary()
    
    def _select_tournament_tasks(self, n_tasks: int = 20) -> List[Task]:
        """Selecciona tareas diversas para el torneo."""
        all_tasks = []
        for task_type, tasks in self.memory_bank.tasks_by_type.items():
            # Seleccionar tareas con diferentes niveles de éxito
            high_success = [t for t in tasks if t.success_rate > 0.7]
            medium_success = [t for t in tasks if 0.3 <= t.success_rate <= 0.7]
            low_success = [t for t in tasks if t.success_rate < 0.3]
            
            # Balancear selección
            selection = (
                random.sample(high_success, min(3, len(high_success))) +
                random.sample(medium_success, min(3, len(medium_success))) +
                random.sample(low_success, min(2, len(low_success)))
            )
            all_tasks.extend(selection)
        
        return random.sample(all_tasks, min(n_tasks, len(all_tasks)))
    
    def _evaluate_tournament_round(self, tasks: List[Task]) -> Dict:
        """Evalúa una ronda del torneo."""
        results = {
            'current_model': {'wins': 0, 'total': 0},
            'baseline': {'wins': 0, 'total': 0}
        }
        
        for task in tasks:
            # Evaluar modelo actual
            current_score = self._evaluate_single_task(task)
            
            # Evaluar baseline (versión simplificada)
            baseline_score = self._evaluate_baseline_task(task)
            
            results['current_model']['total'] += 1
            results['baseline']['total'] += 1
            
            if current_score > baseline_score:
                results['current_model']['wins'] += 1
            elif baseline_score > current_score:
                results['baseline']['wins'] += 1
        
        return results
    
    def _evaluate_single_task(self, task: Task) -> float:
        """Evalúa una tarea individual con el modelo actual."""
        solve_prompt = self.cmd._build_solve_prompt_improved(task)
        response = self.cmd._query_deepseek_improved(solve_prompt)
        
        if response:
            solution = self.cmd._extract_solution_improved(response, task.task_type)
            return self.cmd._calculate_reward_improved(task, solution)
        return 0.0
    
    def _evaluate_baseline_task(self, task: Task) -> float:
        """Evalúa con un modelo baseline simple."""
        # Implementación simplificada como baseline
        if task.task_type == 'deduction':
            return random.uniform(0.3, 0.7)  # Performance baseline simulada
        elif task.task_type == 'abduction':
            return random.uniform(0.2, 0.6)
        else:  # induction
            return random.uniform(0.1, 0.5)
    
    def _update_elo_ratings(self, results: Dict):
        """Actualiza ratings ELO basado en resultados."""
        current_rating = self.elo_ratings['current_model']
        baseline_rating = self.elo_ratings['baseline']
        
        # Calcular resultado esperado
        expected_current = 1 / (1 + 10**((baseline_rating - current_rating) / 400))
        
        # Calcular resultado real
        actual_current = results['current_model']['wins'] / max(1, results['current_model']['total'])
        
        # Actualizar ratings
        k_factor = 32  # Factor de sensibilidad
        self.elo_ratings['current_model'] += k_factor * (actual_current - expected_current)
        self.elo_ratings['baseline'] -= k_factor * (actual_current - expected_current)
    
    def _display_tournament_summary(self):
        """Muestra resumen del torneo."""
        if not self.tournament_history:
            return
        
        total_wins = sum(r['results']['current_model']['wins'] for r in self.tournament_history)
        total_games = sum(r['results']['current_model']['total'] for r in self.tournament_history)
        win_rate = total_wins / max(1, total_games)
        
        CONSOLE.print("\n[TOURNAMENT] Resumen del Torneo", style="bold blue")
        CONSOLE.print(f"Partidas ganadas: {total_wins}/{total_games} ({win_rate:.1%})")
        CONSOLE.print(f"Rating ELO actual: {self.elo_ratings['current_model']:.0f}")
        CONSOLE.print(f"Mejora de rating: {self.elo_ratings['current_model'] - 1000:.0f}")

class MetaLearning:
    """Sistema de meta-aprendizaje para optimización de hiperparámetros."""
    
    def __init__(self):
        self.hyperparameter_history = []
        self.performance_history = []
        self.current_hyperparams = {
            'learning_rate': 1e-4,
            'num_references': 5,
            'num_rollouts': 3,
            'diversity_threshold': 0.8,
            'min_reward_threshold': 0.3
        }
        self.best_hyperparams = self.current_hyperparams.copy()
        self.best_performance = 0.0
    
    def optimize_hyperparameters(self, cmd_instance: AbsoluteZeroCmd, iterations: int = 10):
        """Optimiza hiperparámetros usando búsqueda aleatoria."""
        CONSOLE.print(f"[META] Optimizando hiperparámetros - {iterations} iteraciones", style="bold cyan")
        
        for iteration in range(iterations):
            # Generar nuevos hiperparámetros
            new_hyperparams = self._generate_hyperparameters()
            
            # Aplicar hiperparámetros
            self._apply_hyperparameters(cmd_instance, new_hyperparams)
            
            # Evaluar performance
            performance = self._evaluate_performance(cmd_instance)
            
            # Guardar resultados
            self.hyperparameter_history.append(new_hyperparams.copy())
            self.performance_history.append(performance)
            
            # Actualizar mejores hiperparámetros
            if performance > self.best_performance:
                self.best_performance = performance
                self.best_hyperparams = new_hyperparams.copy()
                logger.info(f"Nuevos mejores hiperparámetros encontrados: {performance:.3f}")
            
            logger.info(f"Meta-iteración {iteration + 1}: performance={performance:.3f}")
        
        # Aplicar mejores hiperparámetros finales
        self._apply_hyperparameters(cmd_instance, self.best_hyperparams)
        self._display_optimization_summary()
    
    def _generate_hyperparameters(self) -> Dict:
        """Genera nuevos hiperparámetros usando búsqueda aleatoria."""
        return {
            'learning_rate': random.uniform(1e-5, 1e-3),
            'num_references': random.randint(3, 10),
            'num_rollouts': random.randint(2, 8),
            'diversity_threshold': random.uniform(0.5, 0.95),
            'min_reward_threshold': random.uniform(0.1, 0.5)
        }
    
    def _apply_hyperparameters(self, cmd_instance: AbsoluteZeroCmd, hyperparams: Dict):
        """Aplica hiperparámetros al sistema."""
        cmd_instance.learning_rate = hyperparams['learning_rate']
        # Actualizar otras configuraciones globales si es necesario
        globals()['NUM_REFERENCES'] = hyperparams['num_references']
        globals()['NUM_ROLLOUTS'] = hyperparams['num_rollouts']
        globals()['DIVERSITY_THRESHOLD'] = hyperparams['diversity_threshold']
        globals()['MIN_REWARD_THRESHOLD'] = hyperparams['min_reward_threshold']
    
    def _evaluate_performance(self, cmd_instance: AbsoluteZeroCmd) -> float:
        """Evalúa performance del sistema con hiperparámetros actuales."""
        # Ejecutar mini-ciclo de evaluación
        test_tasks = self._generate_test_tasks()
        total_reward = 0.0
        
        for task in test_tasks:
            solve_prompt = cmd_instance._build_solve_prompt_improved(task)
            response = cmd_instance._query_deepseek_improved(solve_prompt)
            if response:
                solution = cmd_instance._extract_solution_improved(response, task.task_type)
                reward = cmd_instance._calculate_reward_improved(task, solution)
                total_reward += reward
        
        return total_reward / max(1, len(test_tasks))
    
    def _generate_test_tasks(self) -> List[Task]:
        """Genera tareas de prueba para evaluación."""
        test_tasks = [
            Task(
                program='def f(x):\n    return x + 10',
                input='5',
                output='15',
                task_type='deduction'
            ),
            Task(
                program='def f(x):\n    return x * x',
                input='3',
                output='9',
                task_type='abduction'
            ),
            Task(
                program='def f(x):\n    return x % 2 == 0',
                input=[{'input': '2', 'output': 'True'}, {'input': '3', 'output': 'False'}],
                output='Check if number is even',
                task_type='induction'
            )
        ]
        return test_tasks
    
    def _display_optimization_summary(self):
        """Muestra resumen de optimización."""
        CONSOLE.print("\n[META] Resumen de Optimización", style="bold green")
        CONSOLE.print(f"Mejor performance: {self.best_performance:.3f}")
        CONSOLE.print("Mejores hiperparámetros:")
        for key, value in self.best_hyperparams.items():
            CONSOLE.print(f"  {key}: {value}")

class AdvancedAnalytics:
    """Sistema de análisis avanzado y visualización."""
    
    def __init__(self):
        self.analytics_data = {
            'task_distribution': defaultdict(int),
            'difficulty_progression': [],
            'reward_trends': [],
            'learning_velocity': [],
            'error_patterns': defaultdict(list)
        }
    
    def track_task_creation(self, task: Task):
        """Rastrea creación de tareas."""
        self.analytics_data['task_distribution'][task.task_type] += 1
        self.analytics_data['difficulty_progression'].append({
            'timestamp': datetime.now(),
            'difficulty': task.difficulty,
            'task_type': task.task_type
        })
    
    def track_performance(self, reward: float, task_type: str):
        """Rastrea performance del sistema."""
        self.analytics_data['reward_trends'].append({
            'timestamp': datetime.now(),
            'reward': reward,
            'task_type': task_type
        })
    
    def track_learning_velocity(self, success_rate: float):
        """Rastrea velocidad de aprendizaje."""
        self.analytics_data['learning_velocity'].append({
            'timestamp': datetime.now(),
            'success_rate': success_rate
        })
    
    def track_error(self, error_type: str, context: str):
        """Rastrea errores del sistema."""
        self.analytics_data['error_patterns'][error_type].append({
            'timestamp': datetime.now(),
            'context': context
        })
    
    def generate_report(self) -> str:
        """Genera reporte de analytics detallado."""
        report = ["# Reporte de Analytics Absolute Zero\n"]
        
        # Distribución de tareas
        report.append("## Distribución de Tareas")
        for task_type, count in self.analytics_data['task_distribution'].items():
            report.append(f"- {task_type}: {count} tareas")
        
        # Tendencias de recompensa
        if self.analytics_data['reward_trends']:
            recent_rewards = self.analytics_data['reward_trends'][-20:]
            avg_reward = np.mean([r['reward'] for r in recent_rewards])
            report.append(f"\n## Performance Reciente")
            report.append(f"- Recompensa promedio (últimas 20): {avg_reward:.3f}")
        
        # Velocidad de aprendizaje
        if self.analytics_data['learning_velocity']:
            recent_velocity = self.analytics_data['learning_velocity'][-10:]
            avg_velocity = np.mean([v['success_rate'] for v in recent_velocity])
            report.append(f"- Tasa de éxito promedio: {avg_velocity:.1%}")
        
        # Patrones de error
        report.append(f"\n## Patrones de Error")
        for error_type, occurrences in self.analytics_data['error_patterns'].items():
            report.append(f"- {error_type}: {len(occurrences)} ocurrencias")
        
        return "\n".join(report)
    
    def save_analytics(self, filepath: str = "analytics_report.md"):
        """Guarda reporte de analytics."""
        report = self.generate_report()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        CONSOLE.print(f"[ANALYTICS] Reporte guardado en {filepath}", style="green")

# Extensión de la clase principal con nuevas funcionalidades
class AbsoluteZeroCmd(AbsoluteZeroCmd):
    """Extensión de la clase principal con funcionalidades avanzadas."""
    
    def __init__(self):
        super().__init__()
        self.selfplay_trainer = SelfPlayTraining(self.memory_bank, self)
        self.meta_learner = MetaLearning()
        self.analytics = AdvancedAnalytics()
        self.auto_save_interval = 100  # Cada 100 tareas
        self.tasks_processed = 0
    
    # Nuevos comandos CLI
    def do_tournament(self, arg):
        """Ejecuta torneo de auto-juego."""
        try:
            rounds = int(arg) if arg else 10
            self.selfplay_trainer.run_tournament(rounds)
        except ValueError:
            CONSOLE.print("[ERROR] Número de rondas debe ser entero", style="red")
    
    def do_optimize(self, arg):
        """Optimiza hiperparámetros del sistema."""
        try:
            iterations = int(arg) if arg else 10
            self.meta_learner.optimize_hyperparameters(self, iterations)
        except ValueError:
            CONSOLE.print("[ERROR] Número de iteraciones debe ser entero", style="red")
    
    def do_analytics(self, arg):
        """Genera y muestra reporte de analytics."""
        if arg == "save":
            self.analytics.save_analytics()
        else:
            report = self.analytics.generate_report()
            CONSOLE.print(report)
    
    def do_export_tasks(self, arg):
        """Exporta tareas a archivo JSON."""
        export_data = {}
        for task_type, tasks in self.memory_bank.tasks_by_type.items():
            export_data[task_type] = [task.to_dict() for task in tasks]
        
        filename = arg if arg else f"tasks_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        CONSOLE.print(f"[EXPORT] Tareas exportadas a {filename}", style="green")
    
    def do_import_tasks(self, arg):
        """Importa tareas desde archivo JSON."""
        if not arg:
            CONSOLE.print("[ERROR] Especificar archivo a importar", style="red")
            return
        
        try:
            with open(arg, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            imported_count = 0
            for task_type, tasks_data in import_data.items():
                for task_data in tasks_data:
                    task = Task(**task_data)
                    if self.memory_bank.add_task(task):
                        imported_count += 1
            
            CONSOLE.print(f"[IMPORT] {imported_count} tareas importadas exitosamente", style="green")
        except Exception as e:
            CONSOLE.print(f"[ERROR] Error importando tareas: {e}", style="red")
    
    def do_benchmark(self, arg):
        """Ejecuta benchmark de performance del sistema."""
        CONSOLE.print("[BENCHMARK] Iniciando benchmark...", style="bold yellow")
        
        # Crear conjunto de tareas de benchmark
        benchmark_tasks = self._create_benchmark_tasks()
        
        start_time = time.time()
        total_reward = 0.0
        successful_tasks = 0
        
        with Progress() as progress:
            task_id = progress.add_task("[cyan]Ejecutando benchmark...", total=len(benchmark_tasks))
            
            for task in benchmark_tasks:
                solve_prompt = self._build_solve_prompt_improved(task)
                response = self._query_deepseek_improved(solve_prompt)
                
                if response:
                    solution = self._extract_solution_improved(response, task.task_type)
                    reward = self._calculate_reward_improved(task, solution)
                    total_reward += reward
                    if reward > MIN_REWARD_THRESHOLD:
                        successful_tasks += 1
                
                progress.update(task_id, advance=1)
        
        end_time = time.time()
        avg_reward = total_reward / len(benchmark_tasks)
        success_rate = successful_tasks / len(benchmark_tasks)
        total_time = end_time - start_time
        
        CONSOLE.print("\n[BENCHMARK] Resultados", style="bold green")
        CONSOLE.print(f"Tareas evaluadas: {len(benchmark_tasks)}")
        CONSOLE.print(f"Recompensa promedio: {avg_reward:.3f}")
        CONSOLE.print(f"Tasa de éxito: {success_rate:.1%}")
        CONSOLE.print(f"Tiempo total: {total_time:.2f}s")
        CONSOLE.print(f"Tiempo por tarea: {total_time/len(benchmark_tasks):.2f}s")
    
    def _create_benchmark_tasks(self) -> List[Task]:
        """Crea conjunto estándar de tareas para benchmark."""
        benchmark_tasks = [
            # Deducción básica
            Task(program='def f(x):\n    return x * 2', input='5', output='10', task_type='deduction'),
            Task(program='def f(x):\n    return x + x', input='7', output='14', task_type='deduction'),
            
            # Abducción básica
            Task(program='def f(x):\n    return x * 3', input='4', output='12', task_type='abduction'),
            Task(program='def f(x):\n    return x - 5', input='8', output='3', task_type='abduction'),
            
            # Inducción básica
            Task(
                program='def f(x):\n    return x % 2 == 0',
                input=[{'input': '4', 'output': 'True'}, {'input': '5', 'output': 'False'}],
                output='Check if even',
                task_type='induction'
            ),
            Task(
                program='def f(x):\n    return x ** 2',
                input=[{'input': '2', 'output': '4'}, {'input': '3', 'output': '9'}],
                output='Square function',
                task_type='induction'
            ),
        ]
        return benchmark_tasks
    
    # Override del método principal para incluir analytics
    def _absolute_zero_loop(self, code_content: str, file_path: str, iterations: int):
        """Ciclo principal con analytics integrados."""
        for iteration in range(iterations):
            logger.info(f"Iteración {iteration + 1}/{iterations} para {file_path}")
            
            # Ejecutar ciclo base
            tasks = self._propose_tasks_adaptive(code_content, file_path)
            solutions = self._solve_tasks_improved(tasks, file_path)
            self._update_model_improved(tasks, solutions)
            
            # Analytics y tracking
            for task_type, task_list in tasks.items():
                for task in task_list:
                    self.analytics.track_task_creation(task)
                    self.tasks_processed += 1
            
            for task_type, solution_list in solutions.items():
                for solution_data in solution_list:
                    self.analytics.track_performance(solution_data['reward'], task_type)
            
            # Auto-save periódico
            if self.tasks_processed % self.auto_save_interval == 0:
                self.memory_bank.save_to_disk()
                CONSOLE.print(f"[AUTO-SAVE] Progreso guardado ({self.tasks_processed} tareas)", style="blue")
            
            # Actualizar curriculum
            avg_reward = self._calculate_average_reward(solutions)
            self.curriculum.update_performance(avg_reward)
            self.analytics.track_learning_velocity(avg_reward)

if __name__ == "__main__":
    app = AbsoluteZeroCmd()
    try:
        app.cmdloop()
    except KeyboardInterrupt:
        CONSOLE.print("\n[EXIT] Guardando estado final...", style="yellow")
        app.memory_bank.save_to_disk()
        app.analytics.save_analytics()
        CONSOLE.print("[EXIT] Estado guardado. ¡Hasta luego!", style="green")
