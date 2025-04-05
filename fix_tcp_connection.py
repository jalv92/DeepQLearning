import socket
import time
import uuid
import datetime
import argparse
from threading import Thread

def run_server(port, verbose=False):
    """Ejecuta un servidor TCP simple en el puerto especificado."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_socket.bind(('localhost', port))
        server_socket.listen(5)
        print(f"✅ Servidor escuchando en puerto {port}")
        
        while True:
            client_socket, address = server_socket.accept()
            print(f"✅ Conexión aceptada desde {address}")
            client_thread = Thread(target=handle_client, args=(client_socket, verbose))
            client_thread.daemon = True
            client_thread.start()
    except Exception as e:
        print(f"❌ Error en el servidor: {e}")
    finally:
        server_socket.close()

def handle_client(client_socket, verbose):
    """Maneja un cliente conectado al servidor TCP."""
    try:
        while True:
            data = client_socket.recv(4096)
            if not data:
                break
                
            if verbose:
                print(f"📥 Mensaje recibido: {data.decode('utf-8')}")
    except Exception as e:
        print(f"❌ Error al manejar cliente: {e}")
    finally:
        client_socket.close()

def run_client(port):
    """Ejecuta un cliente TCP que se conecta al puerto especificado."""
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', port))
        print(f"✅ Conectado al servidor en puerto {port}")
        return client_socket
    except Exception as e:
        print(f"❌ Error al conectar con el servidor: {e}")
        return None

def simulate_operation_result(client_socket):
    """Simula un resultado de operación y lo envía al socket del cliente."""
    # Crear un mensaje en el formato esperado:
    # operationId;signalId;entryTime;entryPrice;direction;exitTime;exitPrice;pnl;closeReason
    operation_id = str(uuid.uuid4())
    signal_id = str(uuid.uuid4())
    entry_time = datetime.datetime.now() - datetime.timedelta(minutes=10)
    entry_price = 4500.25  # Precio ejemplo
    direction = 1  # 1=Long, 2=Short
    exit_time = datetime.datetime.now()
    exit_price = 4525.75  # Precio ejemplo
    pnl = 25.5  # Ganancia ejemplo
    close_reason = "ATM_Strategy_Closed"
    
    # Formatear el mensaje
    message = f"{operation_id};{signal_id};{entry_time.strftime('%Y-%m-%d %H:%M:%S.%f')};{entry_price:.4f};{direction};"
    message += f"{exit_time.strftime('%Y-%m-%d %H:%M:%S.%f')};{exit_price:.4f};{pnl:.4f};{close_reason}\n"
    
    try:
        client_socket.sendall(message.encode('utf-8'))
        print(f"✅ Mensaje enviado: Operación {operation_id} con P&L {pnl}")
        return True
    except Exception as e:
        print(f"❌ Error al enviar mensaje: {e}")
        return False

def test_connection(port, send=False):
    """Prueba la conexión a un puerto específico y opcionalmente envía datos."""
    print(f"\n=== Probando conexión TCP en puerto {port} ===\n")
    
    # Intento 1: Comprobar si el puerto está abierto
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex(('localhost', port))
    
    if result == 0:
        print(f"✅ Puerto {port} está abierto")
        sock.close()
        
        if send:
            # Intento 2: Enviar datos
            client = run_client(port)
            if client:
                print("\n🔄 Enviando mensaje de prueba...")
                if simulate_operation_result(client):
                    print("✅ Mensaje enviado correctamente")
                else:
                    print("❌ Error al enviar mensaje")
                client.close()
    else:
        print(f"❌ Puerto {port} está cerrado o no está escuchando")
        print("   Posibles causas:")
        print("   - DeepQ.py o NinjaTrader no están ejecutándose")
        print("   - La configuración del puerto TCP es incorrecta")
        print("   - Un firewall está bloqueando la conexión")
    
    print("\n=== Prueba de conexión completada ===\n")

def simulate_results_server(port=5591, count=5, interval=10):
    """Simula un servidor de resultados en el puerto especificado."""
    print(f"\n=== Iniciando servidor simulado en puerto {port} ===\n")
    
    # Iniciar servidor en un hilo separado
    server_thread = Thread(target=run_server, args=(port, True))
    server_thread.daemon = True
    server_thread.start()
    
    # Esperar a que el servidor esté listo
    time.sleep(1)
    
    print(f"⚠️ Este script simulará {count} operaciones, una cada {interval} segundos")
    print("⚠️ Asegúrate de que DeepQ.py esté ejecutándose para recibir estos mensajes\n")
    
    # Conectar como cliente y enviar mensajes periódicamente
    client = run_client(port)
    if client:
        for i in range(1, count + 1):
            print(f"\n🔄 Enviando operación simulada {i}/{count}...")
            if simulate_operation_result(client):
                print(f"✅ Operación {i} enviada correctamente")
            else:
                print(f"❌ Error al enviar operación {i}")
                
            if i < count:
                print(f"⏱️ Esperando {interval} segundos...")
                time.sleep(interval)
                
        client.close()
        print("\n✅ Simulación completada")
    else:
        print("\n❌ No se pudo conectar para enviar resultados simulados")
    
    print("\n=== Simulación finalizada ===\n")
    
    # Mantener el servidor ejecutándose
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Servidor detenido por el usuario")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Herramienta de diagnóstico y simulación de conexiones TCP para DeepQ')
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Comando test
    test_parser = subparsers.add_parser('test', help='Probar conexión TCP')
    test_parser.add_argument('--port', type=int, default=5591, help='Puerto a probar (default: 5591)')
    test_parser.add_argument('--send', action='store_true', help='Enviar un mensaje de prueba')
    
    # Comando simulate
    simulate_parser = subparsers.add_parser('simulate', help='Simular un servidor de resultados')
    simulate_parser.add_argument('--port', type=int, default=5591, help='Puerto para el servidor (default: 5591)')
    simulate_parser.add_argument('--count', type=int, default=5, help='Número de operaciones a simular (default: 5)')
    simulate_parser.add_argument('--interval', type=int, default=10, help='Intervalo entre operaciones en segundos (default: 10)')
    
    # Servidor simple
    server_parser = subparsers.add_parser('server', help='Ejecutar un servidor TCP simple')
    server_parser.add_argument('--port', type=int, default=5591, help='Puerto para el servidor (default: 5591)')
    server_parser.add_argument('--verbose', action='store_true', help='Mostrar mensajes detallados')
    
    args = parser.parse_args()
    
    if args.command == 'test':
        test_connection(args.port, args.send)
    elif args.command == 'simulate':
        simulate_results_server(args.port, args.count, args.interval)
    elif args.command == 'server':
        try:
            run_server(args.port, args.verbose)
        except KeyboardInterrupt:
            print("\n👋 Servidor detenido por el usuario")
    else:
        print("\n⚠️ Por favor, especifique un comando. Use --help para más información.")
        print("Ejemplos de uso:")
        print("  python fix_tcp_connection.py test --port 5591")
        print("  python fix_tcp_connection.py simulate --port 5591 --count 10 --interval 5")
        print("  python fix_tcp_connection.py server --port 5591 --verbose")
