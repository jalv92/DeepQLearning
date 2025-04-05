import os
import asyncio
import sys
import time
import socket
import sqlite3
import pandas as pd
import numpy as np
import math
import uuid
from datetime import datetime
from collections import deque
import gymnasium as gym  # Cambiado de gym a gymnasium
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.preprocessing import StandardScaler
import random

if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os.system('color')

# ANSI Color Codes
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

# Define TradingEnv class compatible with Gymnasium
class TradingEnv(gym.Env):
    def __init__(self, feature_dimension):
        super(TradingEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(feature_dimension,), dtype=np.float32)
        
        # Trading state
        self.current_price = 0
        self.previous_price = 0
        self.current_state = None
        self.position = 0  # 0: no position, 1: long, -1: short
        
        # Performance tracking
        self.cumulative_reward = 0
        self.trade_count = 0
        
    def reset(self, *, seed=None, options=None, initial_state=None):
        # Ignoramos seed y options para compatibilidad con Gymnasium
        if initial_state is not None:
            self.current_state = initial_state
        else:
            self.current_state = np.zeros(self.observation_space.shape)
        self.position = 0
        self.cumulative_reward = 0
        self.trade_count = 0
        return self.current_state, {}  # Devuelve el estado y un diccionario de información vacío
        
    def step(self, action):
        # Calculate reward based on action and price movement
        reward = 0
        done = False
        truncated = False  # Nuevo parámetro requerido por Gymnasium
        info = {}
        
        # Price increased
        if self.current_price > self.previous_price:
            if action == 1:  # Buy/Long
                reward = 1.5  # Aumentado de 1 a 1.5
            elif action == 2:  # Sell/Short
                reward = -1.5
        # Price decreased
        elif self.current_price < self.previous_price:
            if action == 1:  # Buy/Long
                reward = -1.5
            elif action == 2:  # Sell/Short
                reward = 1.5  # Aumentado de 1 a 1.5
        # Price unchanged
        else:
            if action == 0:  # Hold
                reward = 0.01  # Reducido de 0.1 a 0.01
            else:
                reward = -0.1  # Pequeña penalización por operar cuando no hay cambio
            
        # Update cumulative metrics
        self.cumulative_reward += reward
        if action > 0:
            self.trade_count += 1
            
        # Update position state based on action
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
        self.current_state = new_state
        self.current_price = current_price
        self.previous_price = previous_price

# Initialize variables
default_lag_window_size = 3000
lag_window_size = default_lag_window_size
rolling_window_size = lag_window_size + 1000
min_data = lag_window_size - 1
retrain_interval = 10  # Entrenar cada 10 iteraciones
train_counter = 0  # Contador para el entrenamiento
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
batch_size = 64

rolling_data = pd.DataFrame()
replay_buffer = deque(maxlen=2000)
action_size = 3  # [Hold, Buy, Sell]


# Set up TCP client sockets
import socket

class TCPClient:
    def __init__(self, host='localhost', port=5555):
        self.host = host
        self.port = port
        self.sock = None
        self.buffer = b''
        self.connected = False
        
    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.sock.setblocking(False)
            self.connected = True
            print(f"{BRIGHT_GREEN}Conectado al servidor {self.host}:{self.port}{END}")
            return True
        except Exception as e:
            print(f"{RED}Error al conectar con {self.host}:{self.port}: {e}{END}")
            self.connected = False
            return False
            
    def recv_message(self):
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            data = self.sock.recv(4096)
            if not data:
                self.connected = False
                print(f"{YELLOW}Conexión cerrada por el servidor{END}")
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
            print(f"{RED}Error al recibir datos: {e}{END}")
            self.connected = False
            return None
            
    def send_message(self, message):
        if not self.connected:
            if not self.connect():
                return False
                
        try:
            self.sock.sendall(message.encode('utf-8') + b'\n')
            return True
        except Exception as e:
            print(f"{RED}Error al enviar mensaje: {e}{END}")
            self.connected = False
            return False
            
    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        self.connected = False

# Crear conexiones TCP
data_client = TCPClient(host='localhost', port=5555)
metrics_client = TCPClient(host='localhost', port=5554)

# Socket para enviar resultados (actuará como servidor)
class TCPServer:
    def __init__(self, host='localhost', port=5590):
        self.host = host
        self.port = port
        self.sock = None
        self.clients = []
        self.running = False
        
    def start(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.host, self.port))
            self.sock.listen(5)
            self.sock.setblocking(False)
            self.running = True
            print(f"{BRIGHT_GREEN}Servidor TCP iniciado en {self.host}:{self.port}{END}")
            return True
        except Exception as e:
            print(f"{RED}Error al iniciar servidor TCP en {self.host}:{self.port}: {e}{END}")
            self.running = False
            return False
            
    def check_new_clients(self):
        if not self.running:
            return
            
        try:
            client, addr = self.sock.accept()
            client.setblocking(False)
            self.clients.append(client)
            print(f"{BRIGHT_GREEN}Nuevo cliente conectado desde {addr}{END}")
        except BlockingIOError:
            # No hay nuevas conexiones
            pass
        except Exception as e:
            print(f"{RED}Error al aceptar cliente: {e}{END}")
            
    def broadcast(self, message):
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

# Cliente TCP para recibir resultados de operaciones
class TCPResultsClient:
    def __init__(self, host='localhost', port=5591):
        self.host = host
        self.port = port
        self.sock = None
        self.buffer = b''
        self.connected = False
        
    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.sock.setblocking(False)
            self.connected = True
            print(f"{BRIGHT_GREEN}Conectado al servidor de resultados {self.host}:{self.port}{END}")
            return True
        except Exception as e:
            print(f"{RED}Error al conectar con servidor de resultados {self.host}:{self.port}: {e}{END}")
            self.connected = False
            return False
            
    def recv_message(self):
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            data = self.sock.recv(4096)
            if not data:
                self.connected = False
                print(f"{YELLOW}Conexión cerrada por el servidor de resultados{END}")
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
            print(f"{RED}Error al recibir datos de resultados: {e}{END}")
            self.connected = False
            return None
            
    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        self.connected = False

# Iniciar servidor de señales
results_server = TCPServer(host='localhost', port=5590)
results_server.start()

# Iniciar cliente de resultados
results_client = TCPResultsClient(host='localhost', port=5591)
results_client.connect()

# Evitar iniciar el servidor dos veces
server_already_started = True

version = "1.1.18"  # Versión actualizada con mejoras en exploración y procesamiento de resultados
credit = f"""{BRIGHT_RED}By HFT ALGO  - Deep Q-Network v{version}{END} {BRIGHT_MAGENTA}  Data: IN/OUT Port: 5554-55 / 5580 {END}"""
banner = f"""{BRIGHT_MAGENTA}
██████╗ ███████╗███████╗██████╗      ██████╗ 
██╔══██╗██╔════╝██╔════╝██╔══██╗    ██╔═══██╗
██║  ██║█████╗  █████╗  ██████╔╝    ██║   ██║
██║  ██║██╔══╝  ██╔══╝  ██╔═══╝     ██║▄▄ ██║
██████╔╝███████╗███████╗██║         ╚██████╔╝
╚═════╝ ╚══════╝╚══════╝╚═╝          ╚══▀▀═╝ {END}"""
print(banner)
print(credit)

# Initialize variables for later use
input_shape = None
model = None
target_model = None
last_retrain_time = time.time()
update_target_interval = 1

# Set up database for storing performance data
db_path = 'dqn_learning.db'

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
    action_int = int(action)  # Convert numpy.int64 to standard int
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        try:
            c.execute("INSERT INTO model_performance (run_id, action, reward, state, next_state, done) VALUES (?, ?, ?, ?, ?, ?)",
                      (run_id, action_int.to_bytes(1, 'little'), reward, str(state.tolist()), str(next_state.tolist()), int(done)))
            conn.commit()
        except Exception as e:
            print(f"Failed to log performance: {e}")

def store_experience(state, action, reward, next_state, done):
    action_int = int(action)  # Convert numpy.int64 to standard int
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

# Esta función ya no es necesaria con SB3, se mantiene solo para compatibilidad
def train_dqn(batch_size=64, gamma=0.99):
    print(f"{BRIGHT_YELLOW}Esta función ya no se utiliza con SB3{END}")
    pass

def print_lag_window_in_minutes(lag_window_size):
    seconds_per_data_point = 0.1  # Each data point is 100ms, which is 0.1 seconds
    total_seconds = lag_window_size * seconds_per_data_point
    minutes = total_seconds / 60
    rounded_minutes = round(minutes, 2)
    print(f"{BRIGHT_YELLOW}Settings: Window size of {lag_window_size} data points is equivalent to ~{rounded_minutes} minutes.{END}")

print_lag_window_in_minutes(lag_window_size)

user_decision = input(f"{BRIGHT_WHITE}Would you like to change the default lag window size? (Y/N): {END}").strip().lower()
if user_decision == 'y':
    try:
        user_input = input(f"{BRIGHT_WHITE}Enter a new lag window size (default is {default_lag_window_size}): {END}")
        lag_window_size = int(user_input)
        if lag_window_size < 1:
            print(f"{BRIGHT_YELLOW}Invalid input. Using default value.{END}")
            lag_window_size = default_lag_window_size
    except ValueError:
        print(f"{BRIGHT_YELLOW}Invalid input. Using default value.{END}")
        lag_window_size = default_lag_window_size
else:
    print(f"{BRIGHT_WHITE}No changes made. Using default lag window size of {default_lag_window_size}.{END}")

rolling_window_size = lag_window_size + 1000
min_data = lag_window_size + 10

# Load replay buffer if database exists and contains data
if os.path.exists(db_path):
    load_replay_buffer()

# Función para simular el comportamiento de ZMQ (ya no se usa)
async def receive_most_recent_message(socket):
    print(f"{YELLOW}Esta función ya no se utiliza con TCP{END}")
    return None

# Función auxiliar para crear el entorno de trading vectorizado
def make_env(feature_dim):
    def _init():
        env = TradingEnv(feature_dim)
        return env
    return _init

# Función para generar un ID único para señales
def generate_signal_id():
    return str(uuid.uuid4())

# Función para guardar una señal emitida en la base de datos
def save_signal(signal_id, action, confidence, features):
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute('''
        INSERT INTO signals (signal_id, timestamp, action, confidence, features)
        VALUES (?, ?, ?, ?, ?)
        ''', (signal_id, datetime.now(), int(action), float(confidence), str(features.tolist())))
        conn.commit()

# Función para procesar los resultados de operaciones recibidos
def process_operation_result(result_message):
    """
    Procesa un mensaje de resultado de operación.
    Formato esperado: operationId;signalId;entryTime;entryPrice;direction;exitTime;exitPrice;pnl;closeReason
    """
    print(f"{BRIGHT_CYAN}Procesando mensaje de resultado: {result_message}{END}")
    
    # Primero verificamos si el mensaje tiene contenido válido
    if not result_message or len(result_message.strip()) == 0:
        print(f"{RED}Mensaje de resultado vacío{END}")
        return None
    
    # Eliminamos espacios y caracteres de control
    result_message = result_message.strip()
    
    # Dividimos el mensaje en sus componentes
    parts = result_message.split(';')
    
    # Verificamos que tengamos todas las partes necesarias
    if len(parts) < 9:
        print(f"{RED}Mensaje de resultado incompleto ({len(parts)} partes): {result_message}{END}")
        # Intentamos mostrar las partes que sí tenemos para diagnóstico
        for i, part in enumerate(parts):
            print(f"{RED}  Parte {i}: {part}{END}")
        return None
    
    try:
        # Extraemos cada componente con manejo de errores detallado
        operation_id = parts[0].strip()
        signal_id = parts[1].strip()
        
        # Validamos que los IDs sean UUIDs
        try:
            if not all(x.isalnum() or x == '-' for x in operation_id) or not all(x.isalnum() or x == '-' for x in signal_id):
                print(f"{YELLOW}Advertencia: IDs posiblemente no válidos: op={operation_id}, signal={signal_id}{END}")
        except Exception as id_ex:
            print(f"{YELLOW}Error al validar IDs: {id_ex}{END}")
        
        # Procesamos las fechas
        try:
            entry_time = datetime.strptime(parts[2].strip(), '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            # Intentamos formato alternativo
            try:
                entry_time = datetime.strptime(parts[2].strip(), '%Y-%m-%d %H:%M:%S')
            except ValueError as e:
                print(f"{RED}Error al analizar entry_time: {parts[2]} - {e}{END}")
                entry_time = datetime.now()  # Valor predeterminado
        
        # Procesamos valores numéricos con manejo de errores extenso
        try:
            entry_price = float(parts[3].strip())
        except ValueError as e:
            print(f"{RED}Error al convertir entry_price a float: {parts[3]} - {e}{END}")
            entry_price = 0.0
        
        try:
            direction = int(parts[4].strip())
        except ValueError as e:
            print(f"{RED}Error al convertir direction a int: {parts[4]} - {e}{END}")
            direction = 0
        
        try:
            exit_time = datetime.strptime(parts[5].strip(), '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            try:
                exit_time = datetime.strptime(parts[5].strip(), '%Y-%m-%d %H:%M:%S')
            except ValueError as e:
                print(f"{RED}Error al analizar exit_time: {parts[5]} - {e}{END}")
                exit_time = datetime.now()
        
        try:
            exit_price = float(parts[6].strip())
        except ValueError as e:
            print(f"{RED}Error al convertir exit_price a float: {parts[6]} - {e}{END}")
            exit_price = 0.0
        
        try:
            pnl = float(parts[7].strip())
        except ValueError as e:
            print(f"{RED}Error al convertir pnl a float: {parts[7]} - {e}{END}")
            pnl = 0.0
        
        close_reason = parts[8].strip()
        
        # Mostramos información detallada sobre el resultado procesado
        print(f"{BRIGHT_GREEN}Resultado de operación procesado correctamente:{END}")
        print(f"{BRIGHT_GREEN}  OperationID: {operation_id}{END}")
        print(f"{BRIGHT_GREEN}  SignalID: {signal_id}{END}")
        print(f"{BRIGHT_GREEN}  PnL: {pnl}{END}")
        print(f"{BRIGHT_GREEN}  Razón de cierre: {close_reason}{END}")
        
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
        print(f"{RED}Error al procesar resultado de operación: {e}{END}")
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
        
        print(f"{BRIGHT_CYAN}Experiencia actualizada - SignalId: {signal_id}, Simulada: {experience['simulated_reward']:.2f}, Real: {real_reward:.2f}, Combinada: {combined_reward:.2f}, PnL: {pnl}{END}")
        
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
    metrics_client.connect()
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
                        print(f"{BRIGHT_GREEN}Retroalimentación aplicada para SignalId: {signal_id}, PnL: {pnl}{END}")
                    else:
                        print(f"{YELLOW}No se encontró experiencia pendiente para SignalId: {signal_id}{END}")
        
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
            print(f"{YELLOW}Datos recibidos inválidos: {data}{END}")
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
            print(f"{BRIGHT_BLUE}DataFrame inicializado con {len(columns)} columnas{END}")

        try:
            # Convertir los datos a valores numéricos
            features = []
            for x in parsed_data:
                x = x.strip()
                if x:  # Verificar que no esté vacío
                    try:
                        # Limpiar el valor para asegurar que sea un número válido
                        # Eliminar cualquier carácter no numérico excepto punto y signo negativo
                        clean_x = ''.join(c for c in x if c.isdigit() or c == '.' or c == '-')
                        # Si hay múltiples puntos, quedarse solo con el primero
                        if clean_x.count('.') > 1:
                            first_dot = clean_x.find('.')
                            clean_x = clean_x[:first_dot+1] + clean_x[first_dot+1:].replace('.', '')
                        features.append(float(clean_x))
                    except ValueError:
                        print(f"{YELLOW}Valor no convertible a float, usando 0.0: '{x}'{END}")
                        features.append(0.0)
                else:
                    features.append(0.0)  # Valor por defecto para campos vacíos
                    
            # Crear una nueva fila y agregarla al DataFrame
            new_row = pd.DataFrame([features], columns=rolling_data.columns)
            rolling_data = pd.concat([rolling_data, new_row], ignore_index=True)
            
            # Mantener el tamaño del DataFrame dentro de los límites
            if len(rolling_data) > rolling_window_size:
                rolling_data = rolling_data.iloc[-rolling_window_size:]
            
            # Imprimir información sobre los datos recibidos (solo ocasionalmente)
            if loop_counter % 100 == 0:
                print(f"{BRIGHT_CYAN}Datos recibidos: {features[:3]}...{END}")
                
        except Exception as e:
            print(f"{RED}Error al procesar datos: {e}{END}")
            continue
            
        # Verificar si tenemos suficientes datos para entrenar
        if len(rolling_data) < min_data:
            loop_counter += 1
            if loop_counter % 1 == 0:
                print(f"{BRIGHT_GREEN}Waiting For Data Frame Min {len(rolling_data)} of {min_data}{END}", end='\r')
            continue
            
        # Crear datos desplazados para predecir el precio futuro
        lagged_data = rolling_data.shift(lag_window_size).dropna()
        
        if lagged_data.empty:
            print(f"{CYAN}Esperando más datos para acumular.{END}")
            continue
            
        # Preparar datos para el entrenamiento
        if len(lagged_data.columns) > 1 and 'price' in lagged_data.columns:
            X = lagged_data.drop('price', axis=1).values
            y = rolling_data['price'].iloc[lag_window_size:].values
        else:
            X = lagged_data.values
            y = rolling_data.iloc[lag_window_size:].values
            
        # Verificar si es momento de reentrenar el modelo
        if len(X) >= 10 and X.shape[1] > 0 and (time.time() - last_retrain_time >= update_target_interval):
            X_scaled = scaler.fit_transform(X)
            latest_features = X_scaled[-1, :].reshape(-1)  # Solo para SB3
            
            # Inicializar el modelo si no existe
            if not training_started:
                feature_dimension = latest_features.shape[0]
                print(f"{BRIGHT_BLUE}Inicializando entorno con dimensión de características: {feature_dimension}{END}")
                
                # Crear entorno vectorizado
                vec_env = DummyVecEnv([make_env(feature_dimension)])
                
                # Inicializar modelo PPO con mucha mayor exploración (ent_coef=0.25)
                model = PPO("MlpPolicy", vec_env, verbose=0, 
                           learning_rate=0.0003, 
                           n_steps=2048,
                           batch_size=64,
                           gamma=0.99,
                           ent_coef=0.25,  # Aumentado significativamente para mucha más exploración
                           clip_range=0.2,
                           n_epochs=10)
                
                # Inicializar entorno para inferencia
                env = TradingEnv(feature_dimension)
                env.reset(initial_state=latest_features)
                
                training_started = True
                print(f"{BRIGHT_GREEN}Modelo PPO inicializado correctamente{END}")
            
            # Actualizar el estado con los nuevos datos
            current_price = rolling_data['price'].iloc[-2]
            env.update_state(latest_features, current_price, previous_price if previous_price is not None else current_price)
            
            # Generar ID único para la señal
            signal_id = generate_signal_id()
            
            # Implementar mecanismo de forzado de exploración periódica
            # Ocasionalmente forzar una acción diferente a la que el modelo sugeriría
            if random.random() < 0.15:  # 15% de las veces
                # Seleccionar una acción aleatoria (todas tienen igual probabilidad)
                forced_action = random.randint(0, 2)
                action = forced_action
                print(f"{BRIGHT_MAGENTA}Forzando exploración - acción aleatoria: {action}{END}")
            else:
                # Predecir acción usando el modelo PPO
                action, _ = model.predict(latest_features, deterministic=False)
            
            # Mostrar la acción predicha
            print('')
            if action == 0:
                print(f"{BRIGHT_YELLOW}Recommended trading action: Stay out (Hold){END}")
            elif action == 1:
                print(f"{BRIGHT_GREEN}Recommended trading action: Go long (Buy){END}")
            else:  # action == 2
                print(f"{BRIGHT_RED}Recommended trading action: Go short (Sell){END}")
            
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
                    'state': np.array([latest_features]),
                    'next_state': np.array([latest_features]),
                    'done': int(done)
                }
                
                # Registrar la experiencia para compatibilidad
                log_performance(loop_counter, action, simulated_reward, np.array([latest_features]), np.array([latest_features]), int(done))
                
                # Mostrar información sobre la operación
                print(f"{BRIGHT_CYAN}Signal ID: {signal_id} | Simulated Reward: {simulated_reward} | Cumulative: {info['cumulative_reward']} | Confidence: {confidence:.4f}{END}")
                
                # Incrementar contador de entrenamiento
                train_counter += 1
                
                # Entrenar el modelo periódicamente
                if train_counter >= retrain_interval:
                    print(f"{BRIGHT_BLUE}Entrenando modelo PPO (iteración {train_counter}){END}")
                    # Entrenar por 50 timesteps (aumentado de 10 a 50 para un aprendizaje más profundo)
                    model.learn(total_timesteps=50)
                    train_counter = 0
                
                # Enviar acción al socket con ID, confianza y timestamp
                # El formato debe ser: SignalId;Action;Confidence;Timestamp
                print(f"{BRIGHT_GREEN}Enviando señal: ID={signal_id}, Acción={action}, Confianza={confidence:.4f}{END}")
                results_server.broadcast(f"{signal_id};{float(action)};{confidence:.4f};{time.time()}")
            
            previous_price = current_price
            last_retrain_time = time.time()

async def main():
    try:
        # Imprimir mensaje de inicio
        print(f"{BRIGHT_GREEN}Iniciando DeepQ con SB3 (PPO) para trading de futuros{END}")
        print(f"{BRIGHT_YELLOW}Conectando a NinjaTrader en puertos TCP estándar...{END}")
        
        # Ejecutar bucle principal
        await receive_and_process_data()
    except Exception as e:
        import traceback
        print(f"{RED}Un error ha ocurrido: {e}{END}")
        traceback.print_exc()
    finally:
        # Cerrar conexiones
        data_client.close()
        metrics_client.close()
        results_client.close()
        results_server.close()
        print(f"{BRIGHT_RED}Conexiones cerradas{END}")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"{BRIGHT_YELLOW}Programa interrumpido por el usuario{END}")
    except Exception as e:
        import traceback
        print(f"{RED}Un error ha ocurrido: {e}{END}")
        traceback.print_exc()
    finally:
        input(f"{BRIGHT_WHITE}Presiona Enter para cerrar la consola...{END}")
