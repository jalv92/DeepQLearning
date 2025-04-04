import os
import asyncio
import sys
import time
import socket
import sqlite3
import pandas as pd
import numpy as np
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

# Iniciar servidor para resultados
results_server = TCPServer(host='localhost', port=5590)
results_server.start()

# Evitar iniciar el servidor dos veces
server_already_started = True

version = "1.1.14"  # Versión actualizada según changelog
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

# Create a table to store model performance if it doesn't exist
def setup_database():
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
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
        conn.commit()

setup_database()

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

async def receive_and_process_data():
    global last_retrain_time, rolling_data, train_counter
    version = "1.1.14"  # Versión actualizada según changelog
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

    while True:
        # Verificar nuevas conexiones entrantes en el servidor de resultados
        results_server.check_new_clients()
        
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
            
            # Asegurarse de que la longitud coincida con las columnas
            while len(features) < len(rolling_data.columns):
                features.append(0.0)
            
            # Crear una nueva fila y agregarla al DataFrame
            new_row = pd.DataFrame([features], columns=rolling_data.columns)
            rolling_data = pd.concat([rolling_data, new_row], ignore_index=True)
            
            # Imprimir información sobre los datos recibidos (solo ocasionalmente)
            if loop_counter % 100 == 0:
                print(f"{BRIGHT_CYAN}Datos recibidos: {features[:3]}...{END}")
                
        except ValueError as e:
            print(f"{RED}Error al convertir datos: {e} - Datos: {parsed_data}{END}")
            continue
        except Exception as e:
            print(f"{RED}Error al procesar datos: {e}{END}")
            continue

        if len(rolling_data) < min_data:
            loop_counter += 1
            sys.stdout.write(f"\r{BRIGHT_GREEN}Waiting For Data Frame Min {loop_counter} of {min_data}{END}")
            sys.stdout.flush()
            continue

        if len(rolling_data) > rolling_window_size:
            rolling_data = rolling_data.iloc[-rolling_window_size:]

        lagged_data = rolling_data.shift(lag_window_size).dropna()

        if lagged_data.empty:
            print('')
            print(f"{CYAN}Waiting for more data to accumulate.{END}")
            continue

        # Verificar que haya columnas antes de intentar eliminar la columna de precio
        if len(lagged_data.columns) > 1 and 'price' in lagged_data.columns:
            X = lagged_data.drop('price', axis=1).values
            y = rolling_data['price'].iloc[lag_window_size:].values
        else:
            # Si solo hay una columna o ninguna, usar todos los datos disponibles
            X = lagged_data.values
            y = rolling_data.iloc[lag_window_size:].values

        if len(X) >= 10 and X.shape[1] > 0 and (time.time() - last_retrain_time >= update_target_interval):
            X_scaled = scaler.fit_transform(X)
            latest_features = X_scaled[-1, :].reshape(-1)  # Sin reshape adicional para SB3
            
            # Si es la primera vez, configuramos el entorno y el modelo
            if not training_started:
                feature_dimension = latest_features.shape[0]
                print(f"{BRIGHT_BLUE}Inicializando entorno con dimensión de características: {feature_dimension}{END}")
                
                # Crear entorno vectorizado
                vec_env = DummyVecEnv([make_env(feature_dimension)])
                
                # Inicializar modelo PPO para trading con mayor exploración
                model = PPO("MlpPolicy", vec_env, verbose=0, 
                           learning_rate=0.0003, 
                           n_steps=2048,
                           batch_size=64,
                           gamma=0.99,
                           ent_coef=0.05,  # Aumentado de 0.01 a 0.05 para mayor exploración
                           clip_range=0.2,
                           n_epochs=10)
                
                # Inicializar entorno para inferencia
                env = TradingEnv(feature_dimension)
                env.reset(initial_state=latest_features)
                
                training_started = True
                print(f"{BRIGHT_GREEN}Modelo PPO inicializado correctamente{END}")
            
            # Actualizar el estado del entorno con los nuevos datos
            current_price = rolling_data['price'].iloc[-2]
            env.update_state(latest_features, current_price, previous_price if previous_price is not None else current_price)
            
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
            
            # Ejecutar paso en el entorno para calcular recompensa
            if previous_price is not None:
                _, reward, done, truncated, info = env.step(action)
                
                # Registrar la experiencia y el rendimiento
                log_performance(loop_counter, action, reward, np.array([latest_features]), np.array([latest_features]), int(done))
                
                # Mostrar información sobre la operación
                print(f"{BRIGHT_CYAN}Reward: {reward} | Cumulative: {info['cumulative_reward']}{END}")
                
                # Incrementar contador de entrenamiento
                train_counter += 1
                
                # Entrenar el modelo periodicamente
                if train_counter >= retrain_interval:
                    print(f"{BRIGHT_BLUE}Entrenando modelo PPO (iteración {train_counter}){END}")
                    # Entrenar por 10 timesteps (un mini-batch)
                    model.learn(total_timesteps=10)
                    train_counter = 0
                
                # Calcular confianza basada en la recompensa acumulativa normalizada
                confidence = min(0.99, max(0.5, (info['cumulative_reward'] + 10) / 20))
                # Enviar acción al socket con nivel de confianza
                results_server.broadcast(f"{float(action)};{confidence:.4f};{time.time()}")
            
            previous_price = current_price
            last_retrain_time = time.time()

async def main():
    try:
        await receive_and_process_data()
    finally:
        # Cerrar conexiones
        data_client.close()
        metrics_client.close()
        results_server.close()
        print(f"{BRIGHT_RED}Conexiones cerradas{END}")

if __name__ == '__main__':
    try:
        # Imprimir mensaje de inicio
        print(f"{BRIGHT_GREEN}Iniciando DeepQ con SB3 (PPO) para trading de futuros{END}")
        print(f"{BRIGHT_YELLOW}Conectando a NinjaTrader en puertos TCP estándar...{END}")
        
        # Ejecutar bucle principal
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"{BRIGHT_YELLOW}Programa interrumpido por el usuario{END}")
    except Exception as e:
        import traceback
        print(f"{RED}Un error ha ocurrido: {e}{END}")
        traceback.print_exc()
    finally:
        input(f"{BRIGHT_WHITE}Presiona Enter para cerrar la consola...{END}")
