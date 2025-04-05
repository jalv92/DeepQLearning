import os
import asyncio
import time
import socket
import sqlite3
import pandas as pd
import numpy as np
import math
import uuid
import torch
import random
from datetime import datetime
from collections import deque
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from sklearn.preprocessing import StandardScaler

# Configuración para Windows
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os.system('color')

# ANSI Color Codes para mejor legibilidad en la consola
class Colors:
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    END = "\033[0m"

class TradingEnv(gym.Env):
    """
    Entorno de trading compatible con Gymnasium para aprendizaje por refuerzo.
    Versión simplificada que utiliza vectores de características sin LSTM.
    """
    def __init__(self, feature_dimension):
        super(TradingEnv, self).__init__()
        # Definir el espacio de acciones y observaciones
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.feature_dimension = feature_dimension
        
        # El espacio de observación es un tensor 1D [feature_dimension]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.feature_dimension,), 
            dtype=np.float32
        )
        
        # Estado de trading
        self.current_price = 0
        self.previous_price = 0
        self.position = 0  # 0: sin posición, 1: long, -1: short
        
        # Estado actual
        self.current_state = None
        
        # Seguimiento de rendimiento
        self.cumulative_reward = 0
        self.trade_count = 0
        
    def reset(self, *, seed=None, options=None, initial_state=None):
        """Reinicia el entorno y devuelve el estado inicial"""
        # Crear estado inicial
        if initial_state is not None:
            self.current_state = self._ensure_correct_state_format(initial_state)
        else:
            self.current_state = np.zeros(self.feature_dimension, dtype=np.float32)
        
        # Reiniciar métricas
        self.position = 0
        self.cumulative_reward = 0
        self.trade_count = 0
        
        return self.current_state, {}  # Devuelve el estado y diccionario de info vacío
        
    def step(self, action):
        """Ejecuta una acción en el entorno y devuelve el resultado"""
        # Calcular recompensa basada en la acción y movimiento del precio
        reward = 0
        done = False
        truncated = False  # Parámetro requerido por Gymnasium
        
        # Precio aumentó
        if self.current_price > self.previous_price:
            if action == 1:  # Comprar/Long
                reward = 1.5  # Recompensa mejorada
            elif action == 2:  # Vender/Short
                reward = -1.5
        # Precio disminuyó
        elif self.current_price < self.previous_price:
            if action == 1:  # Comprar/Long
                reward = -1.5
            elif action == 2:  # Vender/Short
                reward = 1.5  # Recompensa mejorada
        # Precio sin cambios
        else:
            if action == 0:  # Mantener
                reward = 0.01  # Pequeña recompensa por no operar sin tendencia
            else:
                reward = -0.1  # Pequeña penalización por operar sin tendencia
            
        # Actualizar métricas acumulativas
        self.cumulative_reward += reward
        if action > 0:
            self.trade_count += 1
            
        # Actualizar posición según la acción
        if action == 1:
            self.position = 1
        elif action == 2:
            self.position = -1
        else:
            self.position = 0
            
        info = {
            'cumulative_reward': self.cumulative_reward,
            'trade_count': self.trade_count,
            'position': self.position
        }
        
        return self.current_state, reward, done, truncated, info
    
    def update_state(self, new_state, current_price, previous_price):
        """Actualiza el estado actual con un nuevo estado y precios"""
        # Actualizar precios
        self.current_price = current_price
        self.previous_price = previous_price
        
        # Procesar y actualizar el estado
        self.current_state = self._ensure_correct_state_format(new_state)
    
    def _ensure_correct_state_format(self, state):
        """Asegura que el estado tenga el formato correcto para el entorno"""
        # Convertir a array numpy si no lo es ya
        if not isinstance(state, np.ndarray):
            state = np.array([state], dtype=np.float32)
            
        # Asegurar que el estado tenga la dimensión correcta
        if len(state.shape) == 0:  # Es un escalar
            state = np.array([state], dtype=np.float32)
            
        # Rellenar o recortar a la dimensión correcta
        if state.shape[0] < self.feature_dimension:
            padded_state = np.zeros(self.feature_dimension, dtype=np.float32)
            padded_state[:state.shape[0]] = state
            return padded_state
        elif state.shape[0] > self.feature_dimension:
            return state[:self.feature_dimension].astype(np.float32)
        else:
            return state.astype(np.float32)

# Configuración global
class Config:
    # Ventanas de tiempo
    DEFAULT_LAG_WINDOW_SIZE = 3000
    LAG_WINDOW_SIZE = DEFAULT_LAG_WINDOW_SIZE
    ROLLING_WINDOW_SIZE = LAG_WINDOW_SIZE + 1000
    MIN_DATA = LAG_WINDOW_SIZE - 1
    
    # Parámetros de entrenamiento
    RETRAIN_INTERVAL = 10  # Entrenar cada 10 iteraciones
    BATCH_SIZE = 64
    
    # Puertos TCP por defecto
    DATA_PORT = 5555
    METRICS_PORT = 5554
    SIGNALS_PORT = 5590
    RESULTS_PORT = 5591
    
    # Versión del software
    VERSION = "1.1.36"  # Versión actualizada con limpieza de proyecto
    
    # Modos de operación
    MODE_NORMAL = 0      # Modo normal, con conexión a todos los servicios
    MODE_SIMULATION = 1  # Modo simulación, cuando no hay conexión con el servidor de resultados
    CURRENT_MODE = MODE_NORMAL  # Modo por defecto, se actualizará al iniciar

# Clase base para comunicación TCP
class TCPBase:
    def __init__(self, host='localhost', port=5555):
        self.host = host
        self.port = port
        self.sock = None
        self.buffer = b''
        self.connected = False
    
    def connect(self):
        """Intenta conectar con el servidor"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.sock.setblocking(False)
            self.connected = True
            print(f"{Colors.BRIGHT_GREEN}Conectado al servidor {self.host}:{self.port}{Colors.END}")
            return True
        except Exception as e:
            print(f"{Colors.RED}Error al conectar con {self.host}:{self.port}: {e}{Colors.END}")
            self.connected = False
            return False
    
    def close(self):
        """Cierra la conexión"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        self.connected = False

class TCPClient(TCPBase):
    """Cliente TCP para recibir datos"""
    def recv_message(self):
        """Recibe mensajes del servidor"""
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            data = self.sock.recv(4096)
            if not data:
                self.connected = False
                print(f"{Colors.YELLOW}Conexión cerrada por el servidor{Colors.END}")
                return None
                
            self.buffer += data
            
            # Procesamos mensajes completos terminados en newline o ;
            messages = []
            while b'\n' in self.buffer or b';' in self.buffer:
                newline_pos = self.buffer.find(b'\n')
                semicolon_pos = self.buffer.find(b';')
                
                if newline_pos == -1:
                    delimiter_pos = semicolon_pos
                elif semicolon_pos == -1:
                    delimiter_pos = newline_pos
                else:
                    delimiter_pos = min(newline_pos, semicolon_pos)
                
                message = self.buffer[:delimiter_pos].decode('utf-8')
                self.buffer = self.buffer[delimiter_pos+1:]
                messages.append(message)
                
            return messages
                
        except BlockingIOError:
            # No hay datos disponibles
            return []
        except Exception as e:
            print(f"{Colors.RED}Error al recibir datos: {e}{Colors.END}")
            self.connected = False
            return None
            
    def send_message(self, message):
        """Envía un mensaje al servidor"""
        if not self.connected:
            if not self.connect():
                return False
                
        try:
            self.sock.sendall(message.encode('utf-8') + b'\n')
            return True
        except Exception as e:
            print(f"{Colors.RED}Error al enviar mensaje: {e}{Colors.END}")
            self.connected = False
            return False

class TCPServer(TCPBase):
    """Servidor TCP para enviar señales a clientes"""
    def __init__(self, host='localhost', port=5590):
        super().__init__(host, port)
        self.clients = []
        self.running = False
    
    def start(self):
        """Inicia el servidor TCP"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.host, self.port))
            self.sock.listen(5)
            self.sock.setblocking(False)
            self.running = True
            print(f"{Colors.BRIGHT_GREEN}Servidor TCP iniciado en {self.host}:{self.port}{Colors.END}")
            return True
        except Exception as e:
            print(f"{Colors.RED}Error al iniciar servidor TCP en {self.host}:{self.port}: {e}{Colors.END}")
            self.running = False
            return False
            
    def check_new_clients(self):
        """Verifica y acepta nuevos clientes"""
        if not self.running:
            return
            
        try:
            client, addr = self.sock.accept()
            client.setblocking(False)
            self.clients.append(client)
            print(f"{Colors.BRIGHT_GREEN}Nuevo cliente conectado desde {addr}{Colors.END}")
        except BlockingIOError:
            # No hay nuevas conexiones
            pass
        except Exception as e:
            print(f"{Colors.RED}Error al aceptar cliente: {e}{Colors.END}")
            
    def broadcast(self, message):
        """Envía un mensaje a todos los clientes conectados"""
        if not self.running:
            return
            
        disconnected = []
        for client in self.clients:
            try:
                client.sendall(message.encode('utf-8') + b'\n')
            except:
                disconnected.append(client)
                
        # Eliminar clientes desconectados
        for client in disconnected:
            try:
                client.close()
            except:
                pass
            self.clients.remove(client)
            
    def close(self):
        """Cierra el servidor y todas las conexiones de clientes"""
        self.running = False
        for client in self.clients:
            try:
                client.close()
            except:
                pass
        self.clients = []
        
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None

class TCPResultsClient(TCPClient):
    """Cliente TCP específico para recibir resultados de operaciones"""
    def __init__(self, host='localhost', port=5555):
        super().__init__(host, port)
        self.message_parts = {}  # Para almacenar partes de mensajes
        self.complete_message_buffer = []  # Buffer para mensajes completos
        self.last_heartbeat = time.time()
        
    def recv_message(self):
        """Recibe mensajes del servidor con soporte para mensajes fragmentados"""
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            data = self.sock.recv(4096)
            if not data:
                self.connected = False
                print(f"{Colors.YELLOW}Conexión cerrada por el servidor{Colors.END}")
                return None
                
            self.buffer += data
            self.last_heartbeat = time.time()  # Actualizar timestamp de heartbeat
            
            # Procesar mensajes completos
            messages = []
            
            # Primero intentamos extraer mensajes completos con delimitadores
            while b'\n' in self.buffer or b';' in self.buffer:
                newline_pos = self.buffer.find(b'\n')
                semicolon_pos = self.buffer.find(b';')
                
                if newline_pos == -1:
                    delimiter_pos = semicolon_pos
                elif semicolon_pos == -1:
                    delimiter_pos = newline_pos
                else:
                    delimiter_pos = min(newline_pos, semicolon_pos)
                
                if delimiter_pos >= 0:
                    message = self.buffer[:delimiter_pos].decode('utf-8')
                    self.buffer = self.buffer[delimiter_pos+1:]
                    
                    # Intentar identificar si es un mensaje de resultado completo
                    if self._is_complete_result_message(message):
                        messages.append(message)
                    else:
                        # Si no es completo, podría ser parte de un mensaje más grande
                        self._process_message_part(message)
            
            # Verificar si podemos reconstruir mensajes completos desde partes
            reconstructed = self._check_for_complete_messages()
            if reconstructed:
                messages.extend(reconstructed)
                
            # Si hay mensajes completos en el buffer, agregarlos a la lista
            if self.complete_message_buffer:
                messages.extend(self.complete_message_buffer)
                self.complete_message_buffer = []
                
            return messages
                
        except BlockingIOError:
            # No hay datos disponibles, verificar heartbeat
            self._check_heartbeat()
            return []
        except Exception as e:
            print(f"{Colors.RED}Error al recibir datos en ResultsClient: {e}{Colors.END}")
            self.connected = False
            return None
            
    def _is_complete_result_message(self, message):
        """Verifica si un mensaje es un resultado de operación completo"""
        # Un mensaje completo debe tener al menos 9 partes separadas por ;
        parts = message.split(';')
        return len(parts) >= 9
        
    def _process_message_part(self, part):
        """Procesa una parte de mensaje e intenta identificar a qué mensaje pertenece"""
        # Intentar identificar si es un UUID (posible ID de operación)
        if len(part) == 36 and part.count('-') == 4:
            # Parece ser un UUID, asumimos que es el inicio de un mensaje
            self.message_parts[part] = [part]
            print(f"{Colors.BRIGHT_CYAN}Inicio de mensaje detectado: {part}{Colors.END}")
        else:
            # Si no es un UUID, buscar en qué mensaje encaja
            # Por ahora simplemente lo añadimos como un mensaje independiente
            self.complete_message_buffer.append(part)
            
    def _check_for_complete_messages(self):
        """Verifica si hay mensajes completos en las partes almacenadas"""
        complete_messages = []
        
        # Recorrer todas las partes de mensajes almacenadas
        for msg_id, parts in list(self.message_parts.items()):
            if len(parts) >= 9:  # Si tenemos al menos 9 partes (formato completo)
                # Reconstruir el mensaje completo
                complete_message = ';'.join(parts)
                complete_messages.append(complete_message)
                # Eliminar del diccionario
                del self.message_parts[msg_id]
                
        return complete_messages
    
    def _check_heartbeat(self):
        """Verifica si ha pasado demasiado tiempo sin comunicación"""
        current_time = time.time()
        # Si han pasado más de 30 segundos sin comunicación, intentar reconectar
        if current_time - self.last_heartbeat > 30:
            print(f"{Colors.YELLOW}No se ha recibido comunicación en 30s, reconectando...{Colors.END}")
            self.connected = False
            self.connect()
            self.last_heartbeat = current_time

# Inicializar variables globales
rolling_data = pd.DataFrame()
replay_buffer = deque(maxlen=2000)
train_counter = 0  # Contador para el entrenamiento
last_retrain_time = time.time()
# Usamos entropía fija de 0.25 en lugar del scheduler

# Crear conexiones TCP
data_client = TCPClient(host='localhost', port=Config.DATA_PORT)
metrics_client = TCPClient(host='localhost', port=Config.METRICS_PORT)
results_server = TCPServer(host='localhost', port=Config.SIGNALS_PORT)
results_client = TCPResultsClient(host='localhost', port=Config.RESULTS_PORT)

# Iniciar servidor y conexiones
results_server.start()
results_client.connect()

# Banner y créditos
banner = f"""{Colors.BRIGHT_MAGENTA}
██████╗ ███████╗███████╗██████╗      ██████╗ 
██╔══██╗██╔════╝██╔════╝██╔══██╗    ██╔═══██╗
██║  ██║█████╗  █████╗  ██████╔╝    ██║   ██║
██║  ██║██╔══╝  ██╔══╝  ██╔═══╝     ██║▄▄ ██║
██████╔╝███████╗███████╗██║         ╚██████╔╝
╚═════╝ ╚══════╝╚══════╝╚═╝          ╚══▀▀═╝ {Colors.END}"""

credit = f"""{Colors.BRIGHT_RED}By HFT ALGO  - Deep Q-Network v{Config.VERSION}{Colors.END} {Colors.BRIGHT_MAGENTA}  Data: IN/OUT Port: {Config.METRICS_PORT}-{Config.DATA_PORT} / {Config.SIGNALS_PORT} {Colors.END}"""

# Configuración de la base de datos
db_path = 'dqn_learning.db'
update_target_interval = 1

# Create extended database schema with tables for signals, operation results, and experiences
def setup_database():
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        
        # Original table for model performance (kept for compatibility)
        c.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            run_id INTEGER,
            action BLOB,
            reward REAL,
            state TEXT,
            next_state TEXT,
            done INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Table for emitted signals
        c.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            signal_id TEXT PRIMARY KEY,
            timestamp DATETIME,
            action INTEGER,
            confidence REAL,
            features TEXT
        )
        ''')
        
        # Table for operation results
        c.execute('''
        CREATE TABLE IF NOT EXISTS operation_results (
            operation_id TEXT PRIMARY KEY,
            signal_id TEXT,
            entry_time DATETIME,
            entry_price REAL,
            direction INTEGER,
            exit_time DATETIME,
            exit_price REAL,
            pnl REAL,
            close_reason TEXT,
            FOREIGN KEY (signal_id) REFERENCES signals(signal_id)
        )
        ''')
        
        # Table for modified experiences with real rewards
        c.execute('''
        CREATE TABLE IF NOT EXISTS real_experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id TEXT,
            action INTEGER,
            simulated_reward REAL,
            real_reward REAL,
            combined_reward REAL,
            state TEXT,
            next_state TEXT,
            done INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (signal_id) REFERENCES signals(signal_id)
        )
        ''')
        
        conn.commit()

setup_database()

# Sistema de recompensas híbrido
class RewardSystem:
    def __init__(self, initial_alpha=0.7, decay_factor=0.995, min_alpha=0.3):
        self.alpha = initial_alpha  # Factor de ponderación para recompensas simuladas
        self.decay_factor = decay_factor  # Factor de decaimiento de alpha
        self.min_alpha = min_alpha  # Valor mínimo de alpha
        
    def decay_alpha(self):
        # Disminuir alpha gradualmente para dar más peso a las recompensas reales con el tiempo
        self.alpha = max(self.min_alpha, self.alpha * self.decay_factor)
        
    def calculate_real_reward(self, pnl, risk_amount=100):
        # Normaliza el P&L basado en la cantidad de riesgo
        normalized_pnl = pnl / risk_amount
        # Función sigmoide para mantener valores en rango razonable [-1, 1]
        real_reward = 2 / (1 + math.exp(-normalized_pnl)) - 1
        return real_reward
        
    def calculate_combined_reward(self, simulated_reward, real_reward):
        # Combinar recompensas simuladas y reales según el factor alpha
        return (self.alpha * simulated_reward) + ((1 - self.alpha) * real_reward)
        
    def store_experience(self, signal_id, action, simulated_reward, real_reward, state, next_state, done=0):
        combined_reward = self.calculate_combined_reward(simulated_reward, real_reward)
        
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute('''
            INSERT INTO real_experiences 
            (signal_id, action, simulated_reward, real_reward, combined_reward, state, next_state, done)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (signal_id, action, simulated_reward, real_reward, combined_reward, 
                 str(state.tolist()), str(next_state.tolist()), done))
            conn.commit()
        
        return combined_reward

# Diccionario para almacenar las experiencias pendientes
pending_experiences = {}

# Instanciar el sistema de recompensas
reward_system = RewardSystem()

def log_performance(run_id, action, reward, state, next_state, done):
    # Extraer el valor escalar del array numpy antes de convertirlo
    action_int = int(action.item() if hasattr(action, 'item') else action)
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        try:
            c.execute("INSERT INTO model_performance (run_id, action, reward, state, next_state, done) VALUES (?, ?, ?, ?, ?, ?)",
                      (run_id, action_int.to_bytes(1, 'little'), reward, str(state.tolist()), str(next_state.tolist()), int(done)))
            conn.commit()
        except Exception as e:
            print(f"Failed to log performance: {e}")

def store_experience(state, action, reward, next_state, done):
    # Extraer el valor escalar del array numpy antes de convertirlo
    action_int = int(action.item() if hasattr(action, 'item') else action)
    replay_buffer.append((state, action_int, reward, next_state, done))

def load_replay_buffer():
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute("SELECT state, action, reward, next_state, done FROM model_performance")
        rows = c.fetchall()
        for row in rows:
            state = np.array(eval(row[0]))
            action = int.from_bytes(row[1], 'little') if row[1] is not None else 0
            reward = float(row[2])
            next_state = np.array(eval(row[3]))
            done = bool(row[4])
            replay_buffer.append((state, action, reward, next_state, done))

def print_welcome():
    """Muestra el banner de bienvenida y créditos"""
    print(banner)
    print(credit)
    print(f"{Colors.BRIGHT_GREEN}Iniciando DeepQ con SB3 (PPO) para trading de futuros{Colors.END}")
    print(f"{Colors.BRIGHT_YELLOW}Conectando a NinjaTrader en puertos TCP estándar...{Colors.END}")

def print_lag_window_in_minutes(lag_window_size):
    """Calcula y muestra la ventana de tiempo en minutos"""
    seconds_per_data_point = 0.1  # Each data point is 100ms, which is 0.1 seconds
    total_seconds = lag_window_size * seconds_per_data_point
    minutes = total_seconds / 60
    rounded_minutes = round(minutes, 2)
    print(f"{Colors.BRIGHT_YELLOW}Settings: Window size of {lag_window_size} data points is equivalent to ~{rounded_minutes} minutes.{Colors.END}")

def configure_lag_window():
    """Permite al usuario configurar el tamaño de la ventana de retraso"""
    print_lag_window_in_minutes(Config.LAG_WINDOW_SIZE)
    
    user_decision = input(f"{Colors.BRIGHT_WHITE}Would you like to change the default lag window size? (Y/N): {Colors.END}").strip().lower()
    if user_decision == 'y':
        try:
            user_input = input(f"{Colors.BRIGHT_WHITE}Enter a new lag window size (default is {Config.DEFAULT_LAG_WINDOW_SIZE}): {Colors.END}")
            Config.LAG_WINDOW_SIZE = int(user_input)
            if Config.LAG_WINDOW_SIZE < 1:
                print(f"{Colors.BRIGHT_YELLOW}Invalid input. Using default value.{Colors.END}")
                Config.LAG_WINDOW_SIZE = Config.DEFAULT_LAG_WINDOW_SIZE
        except ValueError:
            print(f"{Colors.BRIGHT_YELLOW}Invalid input. Using default value.{Colors.END}")
            Config.LAG_WINDOW_SIZE = Config.DEFAULT_LAG_WINDOW_SIZE
    else:
        print(f"{Colors.BRIGHT_WHITE}No changes made. Using default lag window size of {Config.DEFAULT_LAG_WINDOW_SIZE}.{Colors.END}")

    Config.ROLLING_WINDOW_SIZE = Config.LAG_WINDOW_SIZE + 1000
    Config.MIN_DATA = Config.LAG_WINDOW_SIZE + 10

def make_env(feature_dim):
    """Función auxiliar para crear el entorno de trading vectorizado"""
    def _init():
        env = TradingEnv(feature_dim)
        return env
    return _init

# Función para generar un ID único para señales
def generate_signal_id():
    return str(uuid.uuid4())

# Función para guardar una señal emitida en la base de datos
# Adaptador personalizado para datetime en sqlite3
def adapt_datetime(dt):
    """Convierte un objeto datetime a texto ISO8601 para almacenamiento en SQLite"""
    return dt.isoformat()

def convert_datetime(text):
    """Convierte texto ISO8601 de SQLite a objeto datetime"""
    return datetime.fromisoformat(text)

# Registrar adaptadores de datetime para SQLite
sqlite3.register_adapter(datetime, adapt_datetime)
sqlite3.register_converter("datetime", convert_datetime)

def save_signal(signal_id, action, confidence, features):
    with sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        c = conn.cursor()
        # Extraer el valor escalar del array numpy antes de convertirlo
        action_value = action.item() if hasattr(action, 'item') else action
        confidence_value = float(confidence.item() if hasattr(confidence, 'item') else confidence)
        c.execute('''
        INSERT INTO signals (signal_id, timestamp, action, confidence, features)
        VALUES (?, ?, ?, ?, ?)
        ''', (signal_id, datetime.now(), int(action_value), confidence_value, str(features.tolist())))
        conn.commit()

# Función para procesar los resultados de operaciones recibidos
def process_operation_result(result_message):
    """
    Procesa un mensaje de resultado de operación.
    Formato esperado: operationId;signalId;entryTime;entryPrice;direction;exitTime;exitPrice;pnl;closeReason
    """
    print(f"{Colors.BRIGHT_CYAN}Procesando mensaje de resultado: {result_message}{Colors.END}")
    
    # Primero verificamos si el mensaje tiene contenido válido
    if not result_message or len(result_message.strip()) == 0:
        print(f"{Colors.RED}Mensaje de resultado vacío{Colors.END}")
        return None
    
    # Eliminamos espacios y caracteres de control
    result_message = result_message.strip()
    
    # Dividimos el mensaje en sus componentes
    parts = result_message.split(';')
    
    # Verificamos que tengamos todas las partes necesarias
    if len(parts) < 9:
        print(f"{Colors.RED}Mensaje de resultado incompleto ({len(parts)} partes): {result_message}{Colors.END}")
        # Intentamos mostrar las partes que sí tenemos para diagnóstico
        for i, part in enumerate(parts):
            print(f"{Colors.RED}  Parte {i}: {part}{Colors.END}")
        return None
    
    try:
        # Extraemos cada componente con manejo de errores detallado
        operation_id = parts[0].strip()
        signal_id = parts[1].strip()
        
        # Validamos que los IDs sean UUIDs
        try:
            if not all(x.isalnum() or x == '-' for x in operation_id) or not all(x.isalnum() or x == '-' for x in signal_id):
                print(f"{Colors.YELLOW}Advertencia: IDs posiblemente no válidos: op={operation_id}, signal={signal_id}{Colors.END}")
        except Exception as id_ex:
            print(f"{Colors.YELLOW}Error al validar IDs: {id_ex}{Colors.END}")
        
        # Procesamos las fechas
        try:
            entry_time = datetime.strptime(parts[2].strip(), '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            # Intentamos formato alternativo
            try:
                entry_time = datetime.strptime(parts[2].strip(), '%Y-%m-%d %H:%M:%S')
            except ValueError as e:
                print(f"{Colors.RED}Error al analizar entry_time: {parts[2]} - {e}{Colors.END}")
                entry_time = datetime.now()  # Valor predeterminado
        
        # Procesamos valores numéricos con manejo de errores extenso
        try:
            entry_price = float(parts[3].strip())
        except ValueError as e:
            print(f"{Colors.RED}Error al convertir entry_price a float: {parts[3]} - {e}{Colors.END}")
            entry_price = 0.0
        
        try:
            direction = int(parts[4].strip())
        except ValueError as e:
            print(f"{Colors.RED}Error al convertir direction a int: {parts[4]} - {e}{Colors.END}")
            direction = 0
        
        try:
            exit_time = datetime.strptime(parts[5].strip(), '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            try:
                exit_time = datetime.strptime(parts[5].strip(), '%Y-%m-%d %H:%M:%S')
            except ValueError as e:
                print(f"{Colors.RED}Error al analizar exit_time: {parts[5]} - {e}{Colors.END}")
                exit_time = datetime.now()
        
        try:
            exit_price = float(parts[6].strip())
        except ValueError as e:
            print(f"{Colors.RED}Error al convertir exit_price a float: {parts[6]} - {e}{Colors.END}")
            exit_price = 0.0
        
        try:
            pnl = float(parts[7].strip())
        except ValueError as e:
            print(f"{Colors.RED}Error al convertir pnl a float: {parts[7]} - {e}{Colors.END}")
            pnl = 0.0
        
        close_reason = parts[8].strip()
        
        # Mostramos información detallada sobre el resultado procesado
        print(f"{Colors.BRIGHT_GREEN}Resultado de operación procesado correctamente:{Colors.END}")
        print(f"{Colors.BRIGHT_GREEN}  OperationID: {operation_id}{Colors.END}")
        print(f"{Colors.BRIGHT_GREEN}  SignalID: {signal_id}{Colors.END}")
        print(f"{Colors.BRIGHT_GREEN}  PnL: {pnl}{Colors.END}")
        print(f"{Colors.BRIGHT_GREEN}  Razón de cierre: {close_reason}{Colors.END}")
        
        # Guardar el resultado en la base de datos
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute('''
            INSERT INTO operation_results 
            (operation_id, signal_id, entry_time, entry_price, direction, exit_time, exit_price, pnl, close_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (operation_id, signal_id, entry_time, entry_price, direction, exit_time, exit_price, pnl, close_reason))
            conn.commit()
        
        return {
            'operation_id': operation_id,
            'signal_id': signal_id,
            'pnl': pnl,
            'close_reason': close_reason
        }
    except Exception as e:
        import traceback
        print(f"{Colors.RED}Error al procesar resultado de operación: {e}{Colors.END}")
        traceback.print_exc()
        return None

# Función para actualizar experiencias pendientes con resultados reales
def update_pending_experience(signal_id, pnl):
    global pending_experiences, reward_system
    
    if signal_id in pending_experiences:
        experience = pending_experiences[signal_id]
        
        # Calcular recompensa real basada en el P&L
        real_reward = reward_system.calculate_real_reward(pnl)
        
        # Almacenar la experiencia con la recompensa real
        combined_reward = reward_system.store_experience(
            signal_id, 
            experience['action'], 
            experience['simulated_reward'], 
            real_reward,
            experience['state'],
            experience['next_state'],
            experience['done']
        )
        
        print(f"{Colors.BRIGHT_CYAN}Experiencia actualizada - SignalId: {signal_id}, Simulada: {experience['simulated_reward']:.2f}, Real: {real_reward:.2f}, Combinada: {combined_reward:.2f}, PnL: {pnl}{Colors.END}")
        
        # Reducir el factor alpha para dar más peso a las recompensas reales
        reward_system.decay_alpha()
        
        # Eliminar de pendientes
        del pending_experiences[signal_id]
        return True
    
    return False

async def receive_and_process_data():
    global last_retrain_time, rolling_data, train_counter, pending_experiences
    loop_counter = 0
    printOnce = False
    scaler = StandardScaler()
    previous_price = None
    env = None
    model = None
    training_started = False
    feature_dimension = 0
    
    # Intentar conectar con los servidores
    data_client.connect()
    
    # Manejo robusto para métricas (no crítico)
    try:
        metrics_connected = metrics_client.connect()
        if not metrics_connected:
            print(f"{Colors.YELLOW}Servidor de métricas no disponible (puerto {Config.METRICS_PORT}). Continuando sin métricas.{Colors.END}")
    except Exception as e:
        print(f"{Colors.YELLOW}Error al conectar con servidor de métricas: {e} - Continuando sin métricas.{Colors.END}")
    
    results_client.connect()

    while True:
        # Verificar nuevas conexiones entrantes en el servidor de resultados
        results_server.check_new_clients()
        
        # Intentar recibir resultados de operaciones
        result_messages = results_client.recv_message()
        if result_messages and len(result_messages) > 0:
            for result_message in result_messages:
                operation_result = process_operation_result(result_message)
                if operation_result:
                    signal_id = operation_result['signal_id']
                    pnl = operation_result['pnl']
                    
                    # Actualizar experiencias pendientes con resultados reales
                    if update_pending_experience(signal_id, pnl):
                        print(f"{Colors.BRIGHT_GREEN}Retroalimentación aplicada para SignalId: {signal_id}, PnL: {pnl}{Colors.END}")
                    else:
                        print(f"{Colors.YELLOW}No se encontró experiencia pendiente para SignalId: {signal_id}{Colors.END}")
        
        # Intentar recibir datos
        messages = data_client.recv_message()
        
        if not messages or len(messages) == 0:
            # Si no hay mensajes, esperar un poco antes de reintentar
            await asyncio.sleep(0.1)
            continue
            
        # Procesar todos los mensajes recibidos
        data = messages[-1]  # Usar solo el mensaje más reciente
        parsed_data = data.split(';')

        # Verificar que los datos sean válidos
        if not parsed_data or len(parsed_data) == 0:
            print(f"{Colors.YELLOW}Datos recibidos inválidos: {data}{Colors.END}")
            await asyncio.sleep(0.1)
            continue

        # Inicializar el DataFrame si es necesario
        if len(rolling_data.columns) == 0 or len(rolling_data.columns) < len(parsed_data):
            # Crear columnas con nombres significativos
            columns = []
            for i in range(len(parsed_data)):
                if i == 0:
                    columns.append('price')  # La primera columna es el precio
                else:
                    columns.append(f'feature_{i}')
            rolling_data = pd.DataFrame(columns=columns)
            print(f"{Colors.BRIGHT_BLUE}DataFrame inicializado con {len(columns)} columnas{Colors.END}")

        try:
            # Convertir los datos a valores numéricos
            features = []
            for x in parsed_data:
                x = x.strip()
                if x:  # Verificar que no esté vacío
                    try:
                        # Limpiar el valor para asegurar que sea un número válido
                        clean_x = ''.join(c for c in x if c.isdigit() or c == '.' or c == '-')
                        # Si hay múltiples puntos, quedarse solo con el primero
                        if clean_x.count('.') > 1:
                            first_dot = clean_x.find('.')
                            clean_x = clean_x[:first_dot+1] + clean_x[first_dot+1:].replace('.', '')
                        features.append(float(clean_x))
                    except ValueError:
                        print(f"{Colors.YELLOW}Valor no convertible a float, usando 0.0: '{x}'{Colors.END}")
                        features.append(0.0)
                else:
                    features.append(0.0)  # Valor por defecto para campos vacíos
                    
            # Crear una nueva fila y agregarla al DataFrame
            new_row = pd.DataFrame([features], columns=rolling_data.columns)
            rolling_data = pd.concat([rolling_data, new_row], ignore_index=True)
            
            # Mantener el tamaño del DataFrame dentro de los límites
            if len(rolling_data) > Config.ROLLING_WINDOW_SIZE:
                rolling_data = rolling_data.iloc[-Config.ROLLING_WINDOW_SIZE:]
            
            # Imprimir información sobre los datos recibidos (solo ocasionalmente)
            if loop_counter % 100 == 0:
                print(f"{Colors.BRIGHT_CYAN}Datos recibidos: {features[:3]}...{Colors.END}")
                
        except Exception as e:
            print(f"{Colors.RED}Error al procesar datos: {e}{Colors.END}")
            continue
            
        # Verificar si tenemos suficientes datos para entrenar
        if len(rolling_data) < Config.MIN_DATA:
            loop_counter += 1
            if loop_counter % 1 == 0:
                print(f"{Colors.BRIGHT_GREEN}Waiting For Data Frame Min {len(rolling_data)} of {Config.MIN_DATA}{Colors.END}", end='\r')
            continue
            
        # Crear datos desplazados para predecir el precio futuro
        lagged_data = rolling_data.shift(Config.LAG_WINDOW_SIZE).dropna()
        
        if lagged_data.empty:
            print(f"{Colors.CYAN}Esperando más datos para acumular.{Colors.END}")
            continue
            
        # Preparar datos para el entrenamiento
        if len(lagged_data.columns) > 1 and 'price' in lagged_data.columns:
            X = lagged_data.drop('price', axis=1).values
            y = rolling_data['price'].iloc[Config.LAG_WINDOW_SIZE:].values
        else:
            X = lagged_data.values
            y = rolling_data.iloc[Config.LAG_WINDOW_SIZE:].values
            
        # Verificar si es momento de reentrenar el modelo
        if len(X) >= 10 and X.shape[1] > 0 and (time.time() - last_retrain_time >= update_target_interval):
            X_scaled = scaler.fit_transform(X)
            latest_features = X_scaled[-1, :].reshape(-1)  # Solo para SB3
            
            # Inicializar el modelo si no existe
            if not training_started:
                feature_dimension = latest_features.shape[0]
                print(f"{Colors.BRIGHT_BLUE}Inicializando entorno simple - dimensión: {feature_dimension}{Colors.END}")
                
                # Configuración para entorno básico (sin LSTM)
                def make_simple_env(feature_dim):
                    def _init():
                        # Crear entorno simple
                        env = TradingEnv(feature_dim)
                        # Envolver con Monitor para seguimiento
                        return Monitor(env, os.path.join("./logs", f"trading_env_{time.time()}"))
                    return _init
                
                # Crear entorno vectorizado
                vec_env = DummyVecEnv([make_simple_env(feature_dimension)])
                
                # Crear modelo PPO con configuración mejorada
                device = "cpu"  # Forzar uso de CPU para mayor estabilidad
                print(f"{Colors.BRIGHT_GREEN}Usando dispositivo: {device} para entrenamiento con MlpPolicy{Colors.END}")
                
                # Crear el modelo con MlpPolicy mejorada - Usando valor fijo de entropía 0.25
                model = PPO(
                    "MlpPolicy",
                    vec_env, 
                    verbose=1,
                    learning_rate=0.0003,
                    n_steps=128,
                    batch_size=64,
                    gamma=0.97,  # Reducido de 0.99 a 0.97
                    ent_coef=0.35,  # Valor fijo de entropía: 0.35
                    clip_range=0.2,
                    n_epochs=5,  # Aumentado de 4 a 5
                    device=device,
                    # Parámetros de red mejorados
                    policy_kwargs=dict(
                        net_arch=[128, 128],  # Aumentado de [64, 64] a [128, 128]
                        activation_fn=torch.nn.ReLU  # Cambiado de Tanh a ReLU
                    )
                )
                
                # Configurar callbacks para checkpoint y monitoreo
                checkpoint_dir = "./model_checkpoints/"
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_callback = CheckpointCallback(
                    save_freq=1000,
                    save_path=checkpoint_dir,
                    name_prefix="ppo_trading_model",
                    save_replay_buffer=True,
                    save_vecnormalize=True,
                )
                
                # Inicializar entorno para inferencia
                env = TradingEnv(feature_dimension)
                env.reset(initial_state=latest_features)
                
                training_started = True
                print(f"{Colors.BRIGHT_GREEN}Modelo PPO inicializado correctamente con CPU{Colors.END}")
            
            # Actualizar el estado con los nuevos datos
            current_price = rolling_data['price'].iloc[-2]
            env.update_state(latest_features, current_price, previous_price if previous_price is not None else current_price)
            
            # Generar ID único para la señal
            signal_id = generate_signal_id()
            
            # Obtener el estado actual para la predicción
            current_state = env.current_state.reshape(1, -1)  # Asegurar formato (1, feature_dim) para predicción
            
            # Implementar mecanismo de forzado de exploración periódica
            # Ocasionalmente forzar una acción diferente a la que el modelo sugeriría
            if random.random() < 0.25:  # 25% de las veces
                # Seleccionar una acción aleatoria (todas tienen igual probabilidad)
                forced_action = random.randint(0, 2)
                action = forced_action
                print(f"{Colors.BRIGHT_MAGENTA}Forzando exploración - acción aleatoria: {action}{Colors.END}")
            else:
                # Predecir acción usando el modelo PPO con el estado simple
                action, _ = model.predict(current_state, deterministic=False)
            
            # Mostrar la acción predicha
            print('')
            if action == 0:
                print(f"{Colors.BRIGHT_YELLOW}Recommended trading action: Stay out (Hold){Colors.END}")
            elif action == 1:
                print(f"{Colors.BRIGHT_GREEN}Recommended trading action: Go long (Buy){Colors.END}")
            else:  # action == 2
                print(f"{Colors.BRIGHT_RED}Recommended trading action: Go short (Sell){Colors.END}")
            
            # Ejecutar paso en el entorno para calcular recompensa simulada
            if previous_price is not None:
                _, simulated_reward, done, truncated, info = env.step(action)
                
                # Calcular confianza basada en la recompensa acumulativa
                confidence = min(0.99, max(0.5, (info['cumulative_reward'] + 10) / 20))
                
                # Guardar la señal en la base de datos
                save_signal(signal_id, action, confidence, latest_features)
                
                # Almacenar como experiencia pendiente para actualizar con recompensa real
                pending_experiences[signal_id] = {
                    'action': action,
                    'simulated_reward': simulated_reward,
                    'state': env.current_state,
                    'next_state': env.current_state, # Para simplicidad usamos el mismo estado como estado siguiente
                    'done': int(done)
                }
                
                # Registrar la experiencia para compatibilidad
                log_performance(loop_counter, action, simulated_reward, np.array([latest_features]), np.array([latest_features]), int(done))
                
                # Mostrar información sobre la operación
                print(f"{Colors.BRIGHT_CYAN}Signal ID: {signal_id} | Simulated Reward: {simulated_reward} | Cumulative: {info['cumulative_reward']} | Confidence: {confidence:.4f}{Colors.END}")
                
                # Incrementar contador de entrenamiento
                train_counter += 1
                
                # Entrenar el modelo periódicamente
                if train_counter >= Config.RETRAIN_INTERVAL:
                    print(f"{Colors.BRIGHT_BLUE}Entrenando modelo PPO (iteración {train_counter}){Colors.END}")
                    
                    # No hay decaimiento de entropía, se mantiene en 0.25
                    print(f"{Colors.BRIGHT_CYAN}Usando valor fijo de entropía: 0.25{Colors.END}")
                    
                    # Entrenar por 50 timesteps
                    model.learn(total_timesteps=50, callback=checkpoint_callback)
                    train_counter = 0
                
                # Enviar acción al socket con ID, confianza y timestamp
                # El formato debe ser: SignalId;Action;Confidence;Timestamp
                print(f"{Colors.BRIGHT_GREEN}Enviando señal: ID={signal_id}, Acción={action}, Confianza={confidence:.4f}{Colors.END}")
                # Extraer valores escalares para evitar advertencias
                action_value = action.item() if hasattr(action, 'item') else action
                results_server.broadcast(f"{signal_id};{float(action_value)};{confidence:.4f};{time.time()}")
            
            previous_price = current_price
            last_retrain_time = time.time()

async def check_connections():
    """Verifica y configura las conexiones, estableciendo el modo apropiado"""
    global Config
    
    # Verificar conexión con el servidor de datos (esencial)
    data_connected = data_client.connect()
    if not data_connected:
        print(f"{Colors.BRIGHT_RED}Error: No se pudo conectar al servidor de datos (puerto {Config.DATA_PORT}){Colors.END}")
        print(f"{Colors.BRIGHT_RED}Este servidor es esencial para la operación. Por favor verifique que NinjaTrader esté ejecutando con DataFeeder.{Colors.END}")
        return False
    
    # Verificar conexión con el servidor de resultados
    results_connected = results_client.connect()
    if not results_connected:
        print(f"{Colors.YELLOW}Advertencia: No se pudo conectar al servidor de resultados (puerto {Config.RESULTS_PORT}){Colors.END}")
        print(f"{Colors.YELLOW}Operando en MODO SIMULACIÓN. El aprendizaje será limitado a recompensas simuladas.{Colors.END}")
        Config.CURRENT_MODE = Config.MODE_SIMULATION
    else:
        Config.CURRENT_MODE = Config.MODE_NORMAL
        print(f"{Colors.BRIGHT_GREEN}Conectado al servidor de resultados. Operando en MODO NORMAL con recompensas reales.{Colors.END}")
    
    # Iniciar servidor de señales (siempre)
    results_server.start()
    
    # Intentar conexión con servidor de métricas (opcional)
    try:
        metrics_connected = metrics_client.connect()
        if not metrics_connected:
            print(f"{Colors.YELLOW}Servidor de métricas no disponible (puerto {Config.METRICS_PORT}). Continuando sin métricas.{Colors.END}")
    except Exception as e:
        print(f"{Colors.YELLOW}Error al conectar con servidor de métricas: {e} - Continuando sin métricas.{Colors.END}")
    
    return True

async def main():
    try:
        # Imprimir mensaje de bienvenida
        print_welcome()
        
        # Configurar ventana de retraso
        configure_lag_window()
        
        # Verificar conexiones y establecer modo
        connections_ok = await check_connections()
        if not connections_ok:
            print(f"{Colors.BRIGHT_RED}No se pueden establecer las conexiones principales. Cerrando aplicación.{Colors.END}")
            return
            
        # Mostrar modo actual
        if Config.CURRENT_MODE == Config.MODE_NORMAL:
            print(f"{Colors.BRIGHT_GREEN}Modo: NORMAL - Aprendizaje con recompensas reales{Colors.END}")
        else:
            print(f"{Colors.BRIGHT_YELLOW}Modo: SIMULACIÓN - Aprendizaje con recompensas simuladas{Colors.END}")
            
        # Cargar buffer de replay si existe la base de datos
        if os.path.exists(db_path):
            print(f"{Colors.BRIGHT_BLUE}Cargando experiencias previas desde la base de datos...{Colors.END}")
            load_replay_buffer()
            print(f"{Colors.BRIGHT_BLUE}Cargadas {len(replay_buffer)} experiencias.{Colors.END}")
        
        # Ejecutar bucle principal
        await receive_and_process_data()
    except Exception as e:
        import traceback
        print(f"{Colors.RED}Un error ha ocurrido: {e}{Colors.END}")
        traceback.print_exc()
    finally:
        # Cerrar conexiones
        data_client.close()
        metrics_client.close()
        results_client.close()
        results_server.close()
        print(f"{Colors.BRIGHT_RED}Conexiones cerradas{Colors.END}")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"{Colors.BRIGHT_YELLOW}Programa interrumpido por el usuario{Colors.END}")
    except Exception as e:
        import traceback
        print(f"{Colors.RED}Un error ha ocurrido: {e}{Colors.END}")
        traceback.print_exc()
    finally:
        input(f"{Colors.BRIGHT_WHITE}Presiona Enter para cerrar la consola...{Colors.END}")
