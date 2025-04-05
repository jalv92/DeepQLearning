#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de diagnóstico y corrección para resolver el problema de recepción
de resultados de operaciones en DeepQ.py.

Este script puede:
1. Verificar la tabla operation_results en la base de datos
2. Probar la conexión con el puerto 5591 directamente usando la clase TCPResultsClient
3. Insertar manualmente operaciones de prueba en la base de datos
4. Corregir problemas comunes en la recepción de mensajes TCP

Uso: python fix_deepq_receiver.py [comando]
Comandos:
  check - Verifica el estado actual
  test - Prueba la conexión TCP directamente
  fix - Intenta corregir problemas de recepción
  insert - Inserta operaciones de prueba en la base de datos
"""

import os
import sys
import time
import socket
import sqlite3
import argparse
import uuid
from datetime import datetime

# Configuración de la base de datos
DB_PATH = 'dqn_learning.db'
RESULTS_PORT = 5591

class TCPResultsClient:
    """Versión simplificada de la clase TCPResultsClient de DeepQ.py"""
    def __init__(self, host='localhost', port=RESULTS_PORT):
        self.host = host
        self.port = port
        self.sock = None
        self.buffer = b''
        self.connected = False
        self.last_heartbeat = time.time()
    
    def connect(self):
        """Intenta conectar con el servidor"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.sock.setblocking(False)
            self.connected = True
            print(f"✅ Conectado al servidor {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ Error al conectar con {self.host}:{self.port}: {e}")
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

    def recv_message(self):
        """Recibe mensajes del servidor con soporte mejorado para mensajes fragmentados"""
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            data = self.sock.recv(4096)
            if not data:
                self.connected = False
                print(f"⚠️ Conexión cerrada por el servidor")
                return None
                
            self.buffer += data
            self.last_heartbeat = time.time()  # Actualizar timestamp de heartbeat
            
            # Procesar mensajes completos
            messages = []
            
            # Intentamos extraer mensajes completos con delimitadores
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
                    messages.append(message)
            
            return messages
                
        except BlockingIOError:
            # No hay datos disponibles, verificar heartbeat
            if time.time() - self.last_heartbeat > 30:
                print(f"⚠️ No se ha recibido comunicación en 30s, reconectando...")
                self.connected = False
                self.connect()
                self.last_heartbeat = time.time()
            return []
        except Exception as e:
            print(f"❌ Error al recibir datos: {e}")
            self.connected = False
            return None

def check_database():
    """Verifica el estado de la base de datos"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Verificar si las tablas existen
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"📊 Tablas encontradas: {', '.join(tables)}")
        
        # Verificar conteo de registros en cada tabla
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"📈 Registros en {table}: {count}")
        
        # Verificar estructura de operation_results
        if 'operation_results' in tables:
            cursor.execute("PRAGMA table_info(operation_results)")
            columns = cursor.fetchall()
            print(f"\n📋 Estructura de operation_results:")
            for col in columns:
                print(f"    - {col[1]} ({col[2]})")
        
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error al verificar la base de datos: {e}")
        return False

def test_tcp_connection():
    """Prueba la conexión TCP directamente usando la implementación de TCPResultsClient"""
    print(f"\n🔌 Probando conexión TCP en puerto {RESULTS_PORT} con implementación directa...")
    
    client = TCPResultsClient()
    connected = client.connect()
    
    if connected:
        print(f"✅ Conexión establecida correctamente")
        
        # Intenta recibir mensajes durante 10 segundos
        print(f"👂 Escuchando mensajes durante 10 segundos...")
        start_time = time.time()
        message_count = 0
        
        try:
            while time.time() - start_time < 10:
                messages = client.recv_message()
                if messages:
                    message_count += len(messages)
                    for msg in messages:
                        print(f"📨 Mensaje recibido: {msg}")
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("⚠️ Prueba interrumpida por el usuario")
        finally:
            print(f"📊 Total de mensajes recibidos: {message_count}")
            client.close()
    else:
        print(f"❌ No se pudo establecer conexión")
    
    return connected

def insert_test_operations():
    """Inserta operaciones de prueba en la base de datos"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Verificar si la tabla operation_results existe
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='operation_results'")
        if not cursor.fetchone():
            print(f"❌ La tabla operation_results no existe")
            return False
        
        # Generar 5 operaciones de prueba
        operations = []
        for i in range(5):
            operation_id = str(uuid.uuid4())
            signal_id = str(uuid.uuid4())
            entry_time = datetime.now()
            entry_price = 4500.0 + (i * 10)
            direction = 1 if i % 2 == 0 else 2  # Alternar entre Long y Short
            exit_time = datetime.now()
            exit_price = entry_price + (10 if direction == 1 else -10)
            pnl = 10 if direction == 1 else -10
            close_reason = "TEST_OPERATION"
            
            operations.append((
                operation_id, signal_id, entry_time, entry_price, direction,
                exit_time, exit_price, pnl, close_reason
            ))
        
        # Insertar operaciones en la base de datos
        cursor.executemany('''
        INSERT INTO operation_results 
        (operation_id, signal_id, entry_time, entry_price, direction, exit_time, exit_price, pnl, close_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', operations)
        
        conn.commit()
        conn.close()
        
        print(f"✅ {len(operations)} operaciones de prueba insertadas correctamente")
        return True
    except Exception as e:
        print(f"❌ Error al insertar operaciones de prueba: {e}")
        return False

def fix_connection_issues():
    """Intenta corregir problemas comunes de conexión TCP"""
    print(f"\n🔧 Intentando corregir problemas de conexión TCP...")
    
    # 1. Verificar si el puerto está en uso por otro proceso
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.bind(('localhost', RESULTS_PORT))
        print(f"✅ El puerto {RESULTS_PORT} está disponible")
        test_socket.close()
        print(f"⚠️ Ningún proceso está escuchando en el puerto {RESULTS_PORT}. Asegúrate de que DeepQ.py esté en ejecución.")
    except OSError:
        print(f"✅ El puerto {RESULTS_PORT} está en uso (probablemente por DeepQ.py)")
    
    # 2. Verificar conexión bidireccional
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect(('localhost', RESULTS_PORT))
        print(f"✅ Conexión establecida con el puerto {RESULTS_PORT}")
        
        # Enviar un mensaje básico para probar
        message = "TEST_CONNECTION"
        client.sendall(message.encode('utf-8'))
        print(f"✅ Mensaje de prueba enviado: {message}")
        
        # Intentar recibir respuesta (no bloqueante)
        client.setblocking(False)
        try:
            response = client.recv(1024)
            print(f"✅ Respuesta recibida: {response.decode('utf-8')}")
        except BlockingIOError:
            print(f"ℹ️ No se recibió respuesta (normal si DeepQ.py no envía confirmación)")
        
        client.close()
    except Exception as e:
        print(f"❌ Error al conectar con el puerto {RESULTS_PORT}: {e}")
    
    # 3. Sugerencias para corregir problemas
    print("\n📋 Recomendaciones para solucionar problemas de comunicación TCP:")
    print("  1. Asegúrate de que DeepQ.py esté en ejecución cuando envías resultados de operaciones.")
    print("  2. Verifica que NinjaTrader y DataFeeder.cs estén configurados correctamente.")
    print("  3. Comprueba que no haya firewalls bloqueando la comunicación.")
    print("  4. Reinicia DeepQ.py y NinjaTrader para asegurar que las conexiones se inicien correctamente.")
    print("  5. Si el problema persiste, considera ejecutar DeepQ.py con permisos de administrador.")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Fix DeepQ Receiver - Soluciona problemas de recepción de resultados TCP')
    parser.add_argument('command', choices=['check', 'test', 'fix', 'insert'], 
                        help='Comando a ejecutar')
    
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print(f" 🔧 FIX DEEPQ RECEIVER - DIAGNÓSTICO Y CORRECCIÓN 🔧")
    print(f"{'='*50}\n")
    
    if args.command == 'check':
        check_database()
    elif args.command == 'test':
        test_tcp_connection()
    elif args.command == 'fix':
        fix_connection_issues()
    elif args.command == 'insert':
        insert_test_operations()
    
    print(f"\n{'='*50}")
    print(f" ✅ OPERACIÓN COMPLETADA ✅")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
