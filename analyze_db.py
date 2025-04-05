import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import seaborn as sns
from collections import Counter
import json

# Configuración para gráficos más bonitos
plt.style.use('ggplot')
sns.set(style="darkgrid")

class DBAnalyzer:
    def __init__(self, db_path='dqn_learning.db'):
        """Inicializa el analizador de base de datos."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.tables = []
        self.analysis_results = {}
        self.output_dir = "analysis_results"
        
        # Crear directorio para resultados si no existe
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def connect(self):
        """Establece conexión con la base de datos."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Para obtener resultados como diccionarios
            self.cursor = self.conn.cursor()
            print(f"✅ Conexión exitosa a {self.db_path}")
            return True
        except Exception as e:
            print(f"❌ Error al conectar a la base de datos: {e}")
            return False
            
    def get_table_names(self):
        """Obtiene nombres de tablas en la base de datos."""
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        self.tables = [row[0] for row in self.cursor.fetchall()]
        return self.tables
        
    def get_table_info(self, table_name):
        """Obtiene información sobre la estructura de una tabla."""
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        return self.cursor.fetchall()
        
    def count_records(self, table_name):
        """Cuenta el número de registros en una tabla."""
        self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        return self.cursor.fetchone()[0]
        
    def get_date_range(self, table_name):
        """Obtiene el rango de fechas para tablas con timestamps."""
        date_columns = ['timestamp', 'entry_time', 'exit_time']
        for col in date_columns:
            try:
                self.cursor.execute(f"SELECT MIN({col}), MAX({col}) FROM {table_name}")
                min_date, max_date = self.cursor.fetchone()
                if min_date and max_date:
                    return min_date, max_date
            except:
                continue
        return None, None
        
    def analyze_signals(self):
        """Analiza la distribución de señales y su evolución temporal."""
        if 'signals' not in self.tables:
            print("❌ La tabla 'signals' no existe en la base de datos.")
            return None
            
        # Obtener todas las señales
        self.cursor.execute("SELECT * FROM signals ORDER BY timestamp")
        rows = self.cursor.fetchall()
        
        if not rows:
            print("❌ No hay registros en la tabla 'signals'.")
            return None
            
        # Convertir a DataFrame
        signals_df = pd.DataFrame([dict(row) for row in rows])
        
        # Convertir timestamp a datetime si es string, usando formato ISO8601
        if signals_df['timestamp'].dtype == object:
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'], format='ISO8601')
            
        # Distribucion de acciones
        action_counts = signals_df['action'].value_counts().to_dict()
        
        # Calcular porcentajes
        total_signals = len(signals_df)
        action_percentages = {action: count/total_signals*100 for action, count in action_counts.items()}
        
        # Clasificar acciones
        action_map = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
        signals_df['action_name'] = signals_df['action'].map(action_map)
        
        # Evolución temporal - Agrupar por día
        signals_df['date'] = signals_df['timestamp'].dt.date
        daily_actions = signals_df.groupby(['date', 'action_name']).size().unstack(fill_value=0)
        
        # Evolución de confianza promedio por día y tipo de acción
        confidence_evolution = signals_df.groupby(['date', 'action_name'])['confidence'].mean().unstack(fill_value=0)
        
        # Calcular promedio de confianza por acción
        avg_confidence_by_action = signals_df.groupby('action_name')['confidence'].mean().to_dict()
        
        # Analizar tendencias recientes vs históricas
        if len(signals_df) > 10:
            # Dividir datos en recientes (último 25%) vs históricos (primeros 75%)
            split_idx = int(len(signals_df) * 0.75)
            historical_df = signals_df.iloc[:split_idx]
            recent_df = signals_df.iloc[split_idx:]
            
            # Comparar distribución de acciones
            historical_actions = historical_df['action'].value_counts(normalize=True).to_dict()
            recent_actions = recent_df['action'].value_counts(normalize=True).to_dict()
            
            # Asegurar que todas las acciones estén representadas
            for action in [0, 1, 2]:
                if action not in historical_actions:
                    historical_actions[action] = 0
                if action not in recent_actions:
                    recent_actions[action] = 0
            
            # Calcular cambio en porcentaje
            action_changes = {}
            for action in [0, 1, 2]:
                hist_pct = historical_actions.get(action, 0) * 100
                recent_pct = recent_actions.get(action, 0) * 100
                action_changes[action_map[action]] = {
                    'historical_pct': hist_pct,
                    'recent_pct': recent_pct,
                    'change': recent_pct - hist_pct
                }
                
            # Analizar evolución de la confianza
            hist_conf = historical_df.groupby('action')['confidence'].mean().to_dict()
            recent_conf = recent_df.groupby('action')['confidence'].mean().to_dict()
            
            confidence_changes = {}
            for action in [0, 1, 2]:
                hist_conf_val = hist_conf.get(action, 0)
                recent_conf_val = recent_conf.get(action, 0)
                confidence_changes[action_map[action]] = {
                    'historical_conf': hist_conf_val,
                    'recent_conf': recent_conf_val,
                    'change': recent_conf_val - hist_conf_val
                }
        else:
            action_changes = None
            confidence_changes = None
        
        # Guardar resultados
        signal_analysis = {
            'total_signals': total_signals,
            'action_counts': action_counts,
            'action_percentages': action_percentages,
            'avg_confidence_by_action': avg_confidence_by_action,
            'action_changes': action_changes,
            'confidence_changes': confidence_changes,
            'daily_actions': daily_actions.to_dict(),
            'confidence_evolution': confidence_evolution.to_dict()
        }
        
        self.analysis_results['signals'] = signal_analysis
        return signal_analysis
    
    def analyze_operations(self):
        """Analiza los resultados de operaciones."""
        if 'operation_results' not in self.tables:
            print("❌ La tabla 'operation_results' no existe en la base de datos.")
            return None
            
        # Obtener todas las operaciones
        self.cursor.execute("SELECT * FROM operation_results ORDER BY entry_time")
        rows = self.cursor.fetchall()
        
        if not rows:
            print("❌ No hay registros en la tabla 'operation_results'.")
            return None
            
        # Convertir a DataFrame
        ops_df = pd.DataFrame([dict(row) for row in rows])
        
        # Convertir timestamps a datetime con formato ISO8601
        for col in ['entry_time', 'exit_time']:
            if col in ops_df.columns and ops_df[col].dtype == object:
                ops_df[col] = pd.to_datetime(ops_df[col], format='ISO8601')
        
        # Estadísticas generales
        total_ops = len(ops_df)
        
        # Resultados por dirección (Long/Short)
        direction_map = {1: 'Long', 2: 'Short'}
        ops_df['direction_name'] = ops_df['direction'].map(direction_map)
        
        # Análisis de PnL
        pnl_stats = {
            'total_pnl': ops_df['pnl'].sum(),
            'avg_pnl': ops_df['pnl'].mean(),
            'median_pnl': ops_df['pnl'].median(),
            'max_pnl': ops_df['pnl'].max(),
            'min_pnl': ops_df['pnl'].min(),
            'profitable_ops': (ops_df['pnl'] > 0).sum(),
            'losing_ops': (ops_df['pnl'] <= 0).sum(),
            'win_rate': (ops_df['pnl'] > 0).sum() / total_ops if total_ops > 0 else 0
        }
        
        # PnL por dirección
        pnl_by_direction = ops_df.groupby('direction_name')['pnl'].agg([
            'sum', 'mean', 'median', 'count'
        ]).to_dict()
        
        # Win rate por dirección
        win_rate_by_direction = {}
        for direction in ops_df['direction_name'].unique():
            dir_df = ops_df[ops_df['direction_name'] == direction]
            win_rate_by_direction[direction] = (dir_df['pnl'] > 0).sum() / len(dir_df) if len(dir_df) > 0 else 0
            
        # Análisis temporal
        if 'entry_time' in ops_df.columns:
            ops_df['entry_date'] = ops_df['entry_time'].dt.date
            daily_pnl = ops_df.groupby('entry_date')['pnl'].sum()
            daily_count = ops_df.groupby('entry_date').size()
            
            # Tendencias recientes vs históricas
            if len(ops_df) > 10:
                # Dividir datos en recientes (último 25%) vs históricos (primeros 75%)
                split_idx = int(len(ops_df) * 0.75)
                historical_df = ops_df.iloc[:split_idx]
                recent_df = ops_df.iloc[split_idx:]
                
                # Comparar distribución de direcciones
                historical_direction = historical_df['direction_name'].value_counts(normalize=True).to_dict()
                recent_direction = recent_df['direction_name'].value_counts(normalize=True).to_dict()
                
                # Comparar PnL
                historical_pnl = {
                    'avg_pnl': historical_df['pnl'].mean(),
                    'win_rate': (historical_df['pnl'] > 0).sum() / len(historical_df) if len(historical_df) > 0 else 0
                }
                
                recent_pnl = {
                    'avg_pnl': recent_df['pnl'].mean(),
                    'win_rate': (recent_df['pnl'] > 0).sum() / len(recent_df) if len(recent_df) > 0 else 0
                }
                
                # Frecuencia de operaciones
                if 'entry_time' in historical_df.columns and 'entry_time' in recent_df.columns:
                    historical_period = (historical_df['entry_time'].max() - historical_df['entry_time'].min()).days
                    recent_period = (recent_df['entry_time'].max() - recent_df['entry_time'].min()).days
                    
                    historical_freq = len(historical_df) / max(1, historical_period)  # Evitar división por cero
                    recent_freq = len(recent_df) / max(1, recent_period)
                    
                    trading_frequency = {
                        'historical_freq_per_day': historical_freq,
                        'recent_freq_per_day': recent_freq,
                        'change_pct': (recent_freq - historical_freq) / max(0.01, historical_freq) * 100
                    }
                else:
                    trading_frequency = None
            else:
                historical_direction = None
                recent_direction = None
                historical_pnl = None
                recent_pnl = None
                trading_frequency = None
        else:
            daily_pnl = None
            daily_count = None
            historical_direction = None
            recent_direction = None
            historical_pnl = None
            recent_pnl = None
            trading_frequency = None
            
        # Guardar resultados
        operations_analysis = {
            'total_operations': total_ops,
            'pnl_stats': pnl_stats,
            'pnl_by_direction': pnl_by_direction,
            'win_rate_by_direction': win_rate_by_direction,
            'daily_pnl': daily_pnl.to_dict() if daily_pnl is not None else None,
            'daily_count': daily_count.to_dict() if daily_count is not None else None,
            'historical_direction': historical_direction,
            'recent_direction': recent_direction,
            'historical_pnl': historical_pnl,
            'recent_pnl': recent_pnl,
            'trading_frequency': trading_frequency
        }
        
        self.analysis_results['operations'] = operations_analysis
        return operations_analysis
    
    def analyze_experiences(self):
        """Analiza las experiencias de aprendizaje."""
        if 'real_experiences' not in self.tables:
            print("❌ La tabla 'real_experiences' no existe en la base de datos.")
            return None
            
        # Obtener experiencias
        self.cursor.execute("SELECT * FROM real_experiences ORDER BY timestamp")
        rows = self.cursor.fetchall()
        
        if not rows:
            print("❌ No hay registros en la tabla 'real_experiences'.")
            return None
            
        # Convertir a DataFrame
        exp_df = pd.DataFrame([dict(row) for row in rows])
        
        # Convertir timestamp a datetime con formato ISO8601
        if 'timestamp' in exp_df.columns and exp_df['timestamp'].dtype == object:
            exp_df['timestamp'] = pd.to_datetime(exp_df['timestamp'], format='ISO8601')
            
        # Estadísticas de recompensas
        reward_stats = {
            'avg_simulated_reward': exp_df['simulated_reward'].mean(),
            'avg_real_reward': exp_df['real_reward'].mean(),
            'avg_combined_reward': exp_df['combined_reward'].mean(),
            'median_simulated_reward': exp_df['simulated_reward'].median(),
            'median_real_reward': exp_df['real_reward'].median(),
            'median_combined_reward': exp_df['combined_reward'].median()
        }
        
        # Distribución de acciones
        action_counts = exp_df['action'].value_counts().to_dict()
        
        # Recompensas por acción
        rewards_by_action = exp_df.groupby('action').agg({
            'simulated_reward': 'mean',
            'real_reward': 'mean',
            'combined_reward': 'mean'
        }).to_dict()
        
        # Análisis temporal
        if 'timestamp' in exp_df.columns:
            exp_df['date'] = exp_df['timestamp'].dt.date
            daily_rewards = exp_df.groupby('date').agg({
                'simulated_reward': 'mean',
                'real_reward': 'mean',
                'combined_reward': 'mean'
            })
            
            # Evolución del alpha (relación entre recompensas simuladas y reales)
            # Alpha se puede estimar como la proporción de la recompensa combinada que viene de la simulada
            if all(col in exp_df.columns for col in ['simulated_reward', 'real_reward', 'combined_reward']):
                # Estimar alpha para cada registro
                exp_df['estimated_alpha'] = np.nan
                non_zero_mask = (exp_df['simulated_reward'] - exp_df['real_reward']) != 0
                exp_df.loc[non_zero_mask, 'estimated_alpha'] = (
                    (exp_df.loc[non_zero_mask, 'combined_reward'] - exp_df.loc[non_zero_mask, 'real_reward']) / 
                    (exp_df.loc[non_zero_mask, 'simulated_reward'] - exp_df.loc[non_zero_mask, 'real_reward'])
                )
                
                # Evolución del alpha a lo largo del tiempo
                alpha_evolution = exp_df.groupby('date')['estimated_alpha'].mean().to_dict()
            else:
                alpha_evolution = None
                
            # Tendencias recientes vs históricas
            if len(exp_df) > 10:
                # Dividir datos en recientes (último 25%) vs históricos (primeros 75%)
                split_idx = int(len(exp_df) * 0.75)
                historical_df = exp_df.iloc[:split_idx]
                recent_df = exp_df.iloc[split_idx:]
                
                # Comparar distribución de acciones
                historical_actions = historical_df['action'].value_counts(normalize=True).to_dict()
                recent_actions = recent_df['action'].value_counts(normalize=True).to_dict()
                
                # Comparar recompensas
                historical_rewards = {
                    'avg_simulated': historical_df['simulated_reward'].mean(),
                    'avg_real': historical_df['real_reward'].mean(),
                    'avg_combined': historical_df['combined_reward'].mean()
                }
                
                recent_rewards = {
                    'avg_simulated': recent_df['simulated_reward'].mean(),
                    'avg_real': recent_df['real_reward'].mean(),
                    'avg_combined': recent_df['combined_reward'].mean()
                }
                
                # Calcular cambios en recompensas
                reward_changes = {
                    'simulated_change': recent_rewards['avg_simulated'] - historical_rewards['avg_simulated'],
                    'real_change': recent_rewards['avg_real'] - historical_rewards['avg_real'],
                    'combined_change': recent_rewards['avg_combined'] - historical_rewards['avg_combined']
                }
            else:
                historical_actions = None
                recent_actions = None
                historical_rewards = None
                recent_rewards = None
                reward_changes = None
        else:
            daily_rewards = None
            alpha_evolution = None
            historical_actions = None
            recent_actions = None
            historical_rewards = None
            recent_rewards = None
            reward_changes = None
            
        # Guardar resultados
        experiences_analysis = {
            'total_experiences': len(exp_df),
            'reward_stats': reward_stats,
            'action_counts': action_counts,
            'rewards_by_action': rewards_by_action,
            'daily_rewards': daily_rewards.to_dict() if daily_rewards is not None else None,
            'alpha_evolution': alpha_evolution,
            'historical_actions': historical_actions,
            'recent_actions': recent_actions,
            'historical_rewards': historical_rewards,
            'recent_rewards': recent_rewards,
            'reward_changes': reward_changes
        }
        
        self.analysis_results['experiences'] = experiences_analysis
        return experiences_analysis
    
    def analyze_model_performance(self):
        """Analiza el rendimiento del modelo a lo largo del tiempo."""
        if 'model_performance' not in self.tables:
            print("❌ La tabla 'model_performance' no existe en la base de datos.")
            return None
            
        # Obtener datos de rendimiento
        self.cursor.execute("SELECT * FROM model_performance ORDER BY timestamp")
        rows = self.cursor.fetchall()
        
        if not rows:
            print("❌ No hay registros en la tabla 'model_performance'.")
            return None
            
        # Convertir a DataFrame
        perf_df = pd.DataFrame([dict(row) for row in rows])
        
        # Convertir timestamp a datetime con formato ISO8601
        if 'timestamp' in perf_df.columns and perf_df['timestamp'].dtype == object:
            perf_df['timestamp'] = pd.to_datetime(perf_df['timestamp'], format='ISO8601')
            
        # Extraer acción como entero
        if 'action' in perf_df.columns and perf_df['action'].dtype == object:
            perf_df['action_int'] = perf_df['action'].apply(lambda x: int.from_bytes(x, 'little') if isinstance(x, bytes) else x)
        else:
            perf_df['action_int'] = perf_df['action']
            
        # Estadísticas generales
        action_counts = perf_df['action_int'].value_counts().to_dict()
        reward_stats = {
            'avg_reward': perf_df['reward'].mean(),
            'median_reward': perf_df['reward'].median(),
            'max_reward': perf_df['reward'].max(),
            'min_reward': perf_df['reward'].min()
        }
        
        # Recompensas por acción
        rewards_by_action = perf_df.groupby('action_int')['reward'].mean().to_dict()
        
        # Análisis temporal
        if 'timestamp' in perf_df.columns:
            perf_df['date'] = perf_df['timestamp'].dt.date
            daily_actions = perf_df.groupby(['date', 'action_int']).size().unstack(fill_value=0)
            daily_rewards = perf_df.groupby('date')['reward'].mean()
            
            # Tendencias recientes vs históricas
            if len(perf_df) > 10:
                # Dividir datos en recientes (último 25%) vs históricos (primeros 75%)
                split_idx = int(len(perf_df) * 0.75)
                historical_df = perf_df.iloc[:split_idx]
                recent_df = perf_df.iloc[split_idx:]
                
                # Comparar distribución de acciones
                historical_actions = historical_df['action_int'].value_counts(normalize=True).to_dict()
                recent_actions = recent_df['action_int'].value_counts(normalize=True).to_dict()
                
                # Asegurar que todas las acciones estén representadas
                for action in [0, 1, 2]:
                    if action not in historical_actions:
                        historical_actions[action] = 0
                    if action not in recent_actions:
                        recent_actions[action] = 0
                
                # Comparar recompensas
                historical_rewards = historical_df.groupby('action_int')['reward'].mean().to_dict()
                recent_rewards = recent_df.groupby('action_int')['reward'].mean().to_dict()
                
                # Analizar evolución de la distribución de acciones
                action_map = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
                action_evolution = {}
                for action in [0, 1, 2]:
                    hist_pct = historical_actions.get(action, 0) * 100
                    recent_pct = recent_actions.get(action, 0) * 100
                    action_evolution[action_map[action]] = {
                        'historical_pct': hist_pct,
                        'recent_pct': recent_pct,
                        'change': recent_pct - hist_pct
                    }
            else:
                historical_actions = None
                recent_actions = None
                historical_rewards = None
                recent_rewards = None
                action_evolution = None
        else:
            daily_actions = None
            daily_rewards = None
            historical_actions = None
            recent_actions = None
            historical_rewards = None
            recent_rewards = None
            action_evolution = None
            
        # Guardar resultados
        model_analysis = {
            'total_records': len(perf_df),
            'action_counts': action_counts,
            'reward_stats': reward_stats,
            'rewards_by_action': rewards_by_action,
            'daily_actions': daily_actions.to_dict() if daily_actions is not None else None,
            'daily_rewards': daily_rewards.to_dict() if daily_rewards is not None else None,
            'historical_actions': historical_actions,
            'recent_actions': recent_actions,
            'historical_rewards': historical_rewards,
            'recent_rewards': recent_rewards,
            'action_evolution': action_evolution
        }
        
        self.analysis_results['model_performance'] = model_analysis
        return model_analysis
    
    def analyze_database(self):
        """Ejecuta análisis completo de la base de datos."""
        if not self.connect():
            return False
            
        print("\n📊 Analizando base de datos...\n")
        
        # Obtener información básica
        tables = self.get_table_names()
        if not tables:
            print("❌ No se encontraron tablas en la base de datos.")
            return False
            
        print(f"🔍 Tablas encontradas: {', '.join(tables)}")
        
        table_stats = {}
        for table in tables:
            record_count = self.count_records(table)
            min_date, max_date = self.get_date_range(table)
            table_stats[table] = {
                'record_count': record_count,
                'date_range': (min_date, max_date) if min_date and max_date else None
            }
            
        self.analysis_results['table_stats'] = table_stats
        
        # Ejecutar análisis específicos
        print("\n📈 Analizando señales de trading...")
        signal_analysis = self.analyze_signals()
        
        print("\n💰 Analizando resultados de operaciones...")
        operations_analysis = self.analyze_operations()
        
        print("\n🧠 Analizando experiencias de aprendizaje...")
        experiences_analysis = self.analyze_experiences()
        
        print("\n🤖 Analizando rendimiento del modelo...")
        model_analysis = self.analyze_model_performance()
        
        # Detectar anomalías y problemas
        print("\n🔍 Detectando posibles problemas...")
        self.detect_issues()
        
        # Guardar resultados
        with open(os.path.join(self.output_dir, 'analysis_results.json'), 'w') as f:
            # Usar encoder personalizado para manejar tipos no serializables
            json.dump(self.analysis_results, f, default=self._json_encoder, indent=2)
            
        return True
    
    def _json_encoder(self, obj):
        """Encoder personalizado para JSON para manejar tipos no serializables."""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        # Manejar objetos date
        if hasattr(obj, 'isoformat') and callable(getattr(obj, 'isoformat')):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def detect_issues(self):
        """Detecta problemas potenciales en los datos."""
        issues = []
        
        # Verificar conexión con servidor de resultados
        if ('operations' not in self.analysis_results or self.analysis_results['operations'] is None) and \
           ('experiences' in self.analysis_results and self.analysis_results['experiences'] is not None):
            issues.append({
                'severity': 'HIGH',
                'issue': 'Posible problema de conexión con servidor de resultados',
                'description': 'Hay experiencias almacenadas, pero no hay registros de operaciones.',
                'recommendation': 'Verificar la conexión TCP con el puerto 5591 (servidor de resultados) y asegurarse de que DataFeeder.cs esté configurado correctamente.'
            })
            
        # Verificar exceso de 'Hold' en señales recientes
        if 'signals' in self.analysis_results and self.analysis_results['signals'] is not None:
            signal_analysis = self.analysis_results['signals']
            if signal_analysis['action_changes'] is not None:
                hold_change = signal_analysis['action_changes'].get('Hold', {}).get('change', 0)
                if hold_change > 20:  # 20% de aumento en Hold
                    issues.append({
                        'severity': 'HIGH',
                        'issue': 'Aumento significativo en señales de Hold',
                        'description': f'Las señales de Hold han aumentado en {hold_change:.2f}%.',
                        'recommendation': 'Considerar reducir el umbral de confianza (MinConfidenceThreshold) para permitir más operaciones.'
                    })
                    
        # Verificar disminución en la confianza para Buy/Sell
        if 'signals' in self.analysis_results and self.analysis_results['signals'] is not None:
            signal_analysis = self.analysis_results['signals']
            if signal_analysis['confidence_changes'] is not None:
                buy_conf_change = signal_analysis['confidence_changes'].get('Buy', {}).get('change', 0)
                sell_conf_change = signal_analysis['confidence_changes'].get('Sell', {}).get('change', 0)
                
                if buy_conf_change < -0.1 or sell_conf_change < -0.1:
                    issues.append({
                        'severity': 'MEDIUM',
                        'issue': 'Disminución en la confianza para acciones de trading',
                        'description': f'La confianza promedio ha disminuido: Buy: {buy_conf_change:.4f}, Sell: {sell_conf_change:.4f}',
                        'recommendation': 'El modelo podría estar volviéndose más conservador. Considerar aumentar el nivel de entropía o agregar más exploración forzada.'
                    })
                    
        # Verificar distribución de recompensas
        if 'model_performance' in self.analysis_results and self.analysis_results['model_performance'] is not None:
            perf_analysis = self.analysis_results['model_performance']
            if perf_analysis['action_evolution'] is not None:
                hold_change = perf_analysis['action_evolution'].get('Hold', {}).get('change', 0)
                buy_change = perf_analysis['action_evolution'].get('Buy', {}).get('change', 0)
                sell_change = perf_analysis['action_evolution'].get('Sell', {}).get('change', 0)
                
                # Si Hold está aumentando y Buy/Sell disminuyendo, hay una tendencia conservadora
                if hold_change > 15 and (buy_change < -5 or sell_change < -5):
                    issues.append({
                        'severity': 'HIGH',
                        'issue': 'Tendencia hacia comportamiento conservador',
                        'description': f'Se observa un aumento de Hold ({hold_change:.2f}%) y disminución de Buy/Sell ({buy_change:.2f}%/{sell_change:.2f}%).',
                        'recommendation': 'El modelo está volviéndose más conservador. Recomendaciones: 1) Reducir umbral de confianza, 2) Aumentar la forzada exploración (actualmente 15%), 3) Usar valor de entropía más alto.'
                    })
                    
        # Verificar si hay problemas con conexión TCP afectando retroalimentación
        if 'operation_results' in self.tables and self.count_records('operation_results') == 0:
            issues.append({
                'severity': 'CRITICAL',
                'issue': 'No hay resultados de operaciones registrados',
                'description': 'La tabla operation_results existe pero está vacía, lo que indica problemas de retroalimentación.',
                'recommendation': 'Verificar la comunicación entre DeepQ.py y DataFeeder. El problema podría estar en la conexión TCP puerto 5591.'
            })
            
        # Verificar si hay recompensas negativas excesivas para Buy/Sell
        if 'model_performance' in self.analysis_results and self.analysis_results['model_performance'] is not None:
            perf_analysis = self.analysis_results['model_performance']
            rewards_by_action = perf_analysis.get('rewards_by_action', {})
            
            buy_reward = rewards_by_action.get(1, 0)
            sell_reward = rewards_by_action.get(2, 0)
            
            if buy_reward < -0.5 or sell_reward < -0.5:
                issues.append({
                    'severity': 'HIGH',
                    'issue': 'Recompensas negativas elevadas para acciones de trading',
                    'description': f'Las acciones de trading tienen recompensas muy negativas: Buy: {buy_reward:.4f}, Sell: {sell_reward:.4f}',
                    'recommendation': 'El modelo está siendo penalizado excesivamente por operar. Revisar la función de recompensa o incrementar el incentivo para operaciones.'
                })
                
        # Imprimir problemas detectados
        print("\n🚨 Problemas detectados:")
        if not issues:
            print("✅ No se detectaron problemas críticos.")
        else:
            for issue in issues:
                severity = issue['severity']
                icon = "🔴" if severity == "CRITICAL" or severity == "HIGH" else "🟠" if severity == "MEDIUM" else "🟡"
                print(f"{icon} [{severity}] {issue['issue']}")
                print(f"   Descripción: {issue['description']}")
                print(f"   Recomendación: {issue['recommendation']}")
                print("")
                
        self.analysis_results['issues'] = issues
        return issues
        
    def generate_report(self):
        """Genera un informe resumido con los hallazgos más importantes."""
        if not self.analysis_results:
            print("❌ No hay resultados de análisis disponibles.")
            return
            
        # Crear directorio para gráficos
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generar resumen de señales
        if 'signals' in self.analysis_results and self.analysis_results['signals'] is not None:
            signal_analysis = self.analysis_results['signals']
            
            # Gráfico de distribución de acciones
            if 'action_counts' in signal_analysis:
                action_map = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
                action_counts = signal_analysis['action_counts']
                actions = [action_map.get(action, str(action)) for action in action_counts.keys()]
                counts = list(action_counts.values())
                
                plt.figure(figsize=(10, 6))
                ax = plt.bar(actions, counts)
                plt.title('Distribución de Señales por Acción')
                plt.xlabel('Acción')
                plt.ylabel('Cantidad')
                
                # Añadir etiquetas con porcentajes
                total = sum(counts)
                for i, p in enumerate(ax):
                    height = p.get_height()
                    plt.text(p.get_x() + p.get_width()/2., height + 0.1,
                            f'{height} ({height/total*100:.1f}%)',
                            ha="center")
                
                plt.savefig(os.path.join(plots_dir, 'action_distribution.png'))
                plt.close()
                
            # Tendencias en señales
            if 'action_changes' in signal_analysis and signal_analysis['action_changes'] is not None:
                action_changes = signal_analysis['action_changes']
                
                actions = list(action_changes.keys())
                historical = [action_changes[a]['historical_pct'] for a in actions]
                recent = [action_changes[a]['recent_pct'] for a in actions]
                
                plt.figure(figsize=(12, 6))
                x = np.arange(len(actions))
                width = 0.35
                
                plt.bar(x - width/2, historical, width, label='Histórico')
                plt.bar(x + width/2, recent, width, label='Reciente')
                
                plt.xlabel('Acción')
                plt.ylabel('Porcentaje')
                plt.title('Comparación de Distribución de Acciones: Histórico vs Reciente')
                plt.xticks(x, actions)
                plt.legend()
                
                # Añadir etiquetas con los valores
                for i, v in enumerate(historical):
                    plt.text(i - width/2, v + 0.5, f'{v:.1f}%', ha='center')
                for i, v in enumerate(recent):
                    plt.text(i + width/2, v + 0.5, f'{v:.1f}%', ha='center')
                    
                plt.savefig(os.path.join(plots_dir, 'action_trends.png'))
                plt.close()
                
        # Generar reporte de texto
        with open(os.path.join(self.output_dir, 'analysis_report.txt'), 'w') as f:
            f.write("===============================================\n")
            f.write("  REPORTE DE ANÁLISIS DE DEEP Q-LEARNING\n")
            f.write("===============================================\n\n")
            
            # Información básica
            f.write("INFORMACIÓN GENERAL:\n")
            f.write("-----------------\n")
            if 'table_stats' in self.analysis_results:
                table_stats = self.analysis_results['table_stats']
                for table, stats in table_stats.items():
                    f.write(f"Tabla: {table}\n")
                    f.write(f"  Registros: {stats['record_count']}\n")
                    if stats['date_range']:
                        f.write(f"  Rango de fechas: {stats['date_range'][0]} a {stats['date_range'][1]}\n")
                    f.write("\n")
            
            # Análisis de señales
            if 'signals' in self.analysis_results and self.analysis_results['signals'] is not None:
                signal_analysis = self.analysis_results['signals']
                f.write("\nANÁLISIS DE SEÑALES:\n")
                f.write("-----------------\n")
                f.write(f"Total de señales: {signal_analysis.get('total_signals', 'N/A')}\n\n")
                
                f.write("Distribución de acciones:\n")
                action_map = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
                for action, count in signal_analysis.get('action_counts', {}).items():
                    action_name = action_map.get(action, str(action))
                    percentage = signal_analysis.get('action_percentages', {}).get(action, 0)
                    f.write(f"  {action_name}: {count} ({percentage:.2f}%)\n")
                
                f.write("\nConfianza promedio por acción:\n")
                for action_name, confidence in signal_analysis.get('avg_confidence_by_action', {}).items():
                    f.write(f"  {action_name}: {confidence:.4f}\n")
                
                if signal_analysis.get('action_changes') is not None:
                    f.write("\nCambios en distribución de acciones (Histórico vs Reciente):\n")
                    for action, changes in signal_analysis['action_changes'].items():
                        f.write(f"  {action}: {changes['historical_pct']:.2f}% → {changes['recent_pct']:.2f}% (Cambio: {changes['change']:.2f}%)\n")
                
                if signal_analysis.get('confidence_changes') is not None:
                    f.write("\nCambios en confianza (Histórico vs Reciente):\n")
                    for action, changes in signal_analysis['confidence_changes'].items():
                        f.write(f"  {action}: {changes['historical_conf']:.4f} → {changes['recent_conf']:.4f} (Cambio: {changes['change']:.4f})\n")
            
            # Análisis de operaciones
            if 'operations' in self.analysis_results and self.analysis_results['operations'] is not None:
                ops_analysis = self.analysis_results['operations']
                f.write("\nANÁLISIS DE OPERACIONES:\n")
                f.write("-----------------\n")
                f.write(f"Total de operaciones: {ops_analysis.get('total_operations', 'N/A')}\n\n")
                
                if 'pnl_stats' in ops_analysis:
                    pnl_stats = ops_analysis['pnl_stats']
                    f.write("Estadísticas de P&L:\n")
                    f.write(f"  P&L Total: {pnl_stats.get('total_pnl', 'N/A')}\n")
                    f.write(f"  P&L Promedio: {pnl_stats.get('avg_pnl', 'N/A')}\n")
                    f.write(f"  Win Rate: {pnl_stats.get('win_rate', 'N/A')*100:.2f}%\n")
                    f.write(f"  Operaciones Rentables: {pnl_stats.get('profitable_ops', 'N/A')}\n")
                    f.write(f"  Operaciones Perdedoras: {pnl_stats.get('losing_ops', 'N/A')}\n")
                
                if 'trading_frequency' in ops_analysis and ops_analysis['trading_frequency'] is not None:
                    freq = ops_analysis['trading_frequency']
                    f.write("\nFrecuencia de operaciones:\n")
                    f.write(f"  Histórica: {freq.get('historical_freq_per_day', 'N/A'):.2f} ops/día\n")
                    f.write(f"  Reciente: {freq.get('recent_freq_per_day', 'N/A'):.2f} ops/día\n")
                    f.write(f"  Cambio: {freq.get('change_pct', 'N/A'):.2f}%\n")
            
            # Análisis de modelo
            if 'model_performance' in self.analysis_results and self.analysis_results['model_performance'] is not None:
                model_analysis = self.analysis_results['model_performance']
                f.write("\nANÁLISIS DE RENDIMIENTO DEL MODELO:\n")
                f.write("-----------------\n")
                f.write(f"Total de registros: {model_analysis.get('total_records', 'N/A')}\n\n")
                
                if 'action_evolution' in model_analysis and model_analysis['action_evolution'] is not None:
                    f.write("Evolución de acciones (Histórico vs Reciente):\n")
                    for action, changes in model_analysis['action_evolution'].items():
                        f.write(f"  {action}: {changes['historical_pct']:.2f}% → {changes['recent_pct']:.2f}% (Cambio: {changes['change']:.2f}%)\n")
                
                if 'reward_stats' in model_analysis:
                    reward_stats = model_analysis['reward_stats']
                    f.write("\nEstadísticas de recompensas:\n")
                    f.write(f"  Recompensa promedio: {reward_stats.get('avg_reward', 'N/A'):.4f}\n")
                    f.write(f"  Recompensa mediana: {reward_stats.get('median_reward', 'N/A'):.4f}\n")
                    f.write(f"  Recompensa máxima: {reward_stats.get('max_reward', 'N/A'):.4f}\n")
                    f.write(f"  Recompensa mínima: {reward_stats.get('min_reward', 'N/A'):.4f}\n")
                
                if 'rewards_by_action' in model_analysis:
                    rewards = model_analysis['rewards_by_action']
                    action_map = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
                    f.write("\nRecompensas por acción:\n")
                    for action, reward in rewards.items():
                        action_name = action_map.get(int(action), action)
                        f.write(f"  {action_name}: {reward:.4f}\n")
            
            # Problemas detectados
            if 'issues' in self.analysis_results:
                issues = self.analysis_results['issues']
                f.write("\nPROBLEMAS DETECTADOS:\n")
                f.write("-----------------\n")
                if not issues:
                    f.write("No se detectaron problemas críticos.\n")
                else:
                    for idx, issue in enumerate(issues, 1):
                        f.write(f"{idx}. [{issue['severity']}] {issue['issue']}\n")
                        f.write(f"   Descripción: {issue['description']}\n")
                        f.write(f"   Recomendación: {issue['recommendation']}\n\n")
            
            # Conclusiones y recomendaciones
            f.write("\nCONCLUSIONES Y RECOMENDACIONES:\n")
            f.write("-----------------\n")
            
            # Analizar si el modelo se está volviendo conservador
            conservative_trend = False
            
            if 'model_performance' in self.analysis_results and self.analysis_results['model_performance'] is not None:
                model_analysis = self.analysis_results['model_performance']
                if model_analysis.get('action_evolution') and 'Hold' in model_analysis['action_evolution']:
                    hold_change = model_analysis['action_evolution']['Hold'].get('change', 0)
                    if hold_change > 10:
                        conservative_trend = True
                        f.write("🚨 TENDENCIA CONSERVADORA DETECTADA: El modelo está mostrando un comportamiento más conservador,\n")
                        f.write(f"con un aumento de {hold_change:.2f}% en la frecuencia de acciones 'Hold'.\n\n")
            
            if 'operations' in self.analysis_results and self.analysis_results['operations'] is not None:
                ops_analysis = self.analysis_results['operations']
                if ops_analysis.get('trading_frequency') and ops_analysis['trading_frequency'].get('change_pct', 0) < -20:
                    conservative_trend = True
                    change = ops_analysis['trading_frequency']['change_pct']
                    f.write(f"🚨 DISMINUCIÓN DE FRECUENCIA: La frecuencia de operaciones ha disminuido un {abs(change):.2f}%.\n\n")
            
            if conservative_trend:
                f.write("RECOMENDACIONES PARA CORREGIR COMPORTAMIENTO CONSERVADOR:\n\n")
                f.write("1. REDUCIR UMBRAL DE CONFIANZA: Bajar MinConfidenceThreshold en DataFeeder.cs de 0.6 a un valor entre 0.4-0.5.\n")
                f.write("2. AUMENTAR EXPLORACIÓN: Incrementar el factor de exploración forzada de 15% a 25-30% en DeepQ.py.\n")
                f.write("3. AUMENTAR ENTROPÍA: Incrementar el valor fijo de entropía de 0.25 a 0.35-0.4 para fomentar la exploración.\n")
                f.write("4. REVISAR FUNCIONAMIENTO TCP: Verificar que la retroalimentación real de operaciones funcione correctamente.\n")
                f.write("5. CONSIDERAR REINICIO PARCIAL: En situaciones extremas, guardar el modelo actual pero reiniciar con\n")
                f.write("   parámetros más exploratorios durante un período de reentrenamiento.\n\n")

            # Problemas de conexión
            connection_issues = False
            if 'operation_results' in self.tables and self.count_records('operation_results') == 0:
                connection_issues = True
                f.write("❗ PROBLEMAS DE CONEXIÓN DETECTADOS: No hay resultados de operaciones registrados, lo que indica\n")
                f.write("problemas en la comunicación bidireccional entre DeepQ.py y DataFeeder.cs.\n\n")
                f.write("RECOMENDACIONES PARA RESOLVER PROBLEMAS DE CONEXIÓN:\n\n")
                f.write("1. Verificar que la conexión TCP en el puerto 5591 esté funcionando correctamente\n")
                f.write("2. Revisar los logs de errores en ambos lados (Python y NinjaTrader)\n")
                f.write("3. Ejecutar el script de diagnóstico fix_deepq_connection.py\n")
                f.write("4. Asegurar que ambos componentes estén ejecutándose simultáneamente\n\n")
            
            f.write("\nInforme generado el " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
        print(f"\n✅ Informe generado en {self.output_dir}/analysis_report.txt")
        print(f"✅ Gráficos guardados en {plots_dir}")
        
        return os.path.join(self.output_dir, 'analysis_report.txt')


def main():
    """Función principal"""
    print("\n" + "="*50)
    print(" 🔍 ANALIZADOR DE BASE DE DATOS DEEP Q-LEARNING 🔍")
    print("="*50 + "\n")
    
    # Crear y ejecutar el analizador
    analyzer = DBAnalyzer(db_path='dqn_learning.db')
    analyzer.analyze_database()
    
    # Generar informe
    analyzer.generate_report()
    
    print("\n" + "="*50)
    print(" 📊 ANÁLISIS COMPLETO 📊")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
