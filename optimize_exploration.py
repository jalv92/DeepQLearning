#!/usr/bin/env python
# coding: utf-8

"""
Script para optimizar los parámetros de exploración en DeepQ.py y DataFeeder.cs
Este script ajusta los siguientes parámetros para hacer que el robot sea menos conservador:

1. En DeepQ.py:
   - Aumenta el valor de entropía de 0.25 a 0.35
   - Aumenta la exploración forzada de 15% a 25%

2. En DataFeeder.cs:
   - Reduce el umbral de confianza (MinConfidenceThreshold) de 0.6 a 0.45
"""

import os
import re
import sys
import shutil
import datetime
import sqlite3
from pathlib import Path

def backup_file(file_path):
    """Crea una copia de seguridad del archivo con fecha y hora."""
    backup_path = f"{file_path}.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    shutil.copy2(file_path, backup_path)
    print(f"✅ Copia de seguridad creada: {backup_path}")
    return backup_path

def modify_deepq_parameters():
    """Modifica los parámetros clave en DeepQ.py para incrementar la exploración."""
    file_path = 'DeepQ.py'
    
    if not os.path.exists(file_path):
        print(f"❌ Error: El archivo {file_path} no existe")
        return False
    
    backup_file(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 1. Aumentar el valor de entropía de 0.25 a 0.35
    # Buscar patrones como: ent_coef=0.25, 
    content = re.sub(r'ent_coef=0\.25', 'ent_coef=0.35', content)
    # También buscar comentarios que mencionan el valor de entropía
    content = re.sub(r'# Valor fijo de entropía: 0\.25', '# Valor fijo de entropía: 0.35', content)
    
    # 2. Aumentar la exploración forzada de 15% a 25%
    # Buscar patrones como: if random.random() < 0.15:
    content = re.sub(r'if random\.random\(\) < 0\.15', 'if random.random() < 0.25', content)
    # También buscar comentarios que mencionan el porcentaje
    content = re.sub(r'# 15% de las veces', '# 25% de las veces', content)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    
    print(f"✅ Parámetros de exploración en {file_path} modificados con éxito:")
    print("   - Valor de entropía aumentado de 0.25 a 0.35")
    print("   - Exploración forzada aumentada de 15% a 25%")
    
    return True

def modify_datafeeder_parameters():
    """Modifica el umbral de confianza en DataFeeder.cs para permitir más operaciones."""
    file_path = 'DataFeeder.cs'
    
    if not os.path.exists(file_path):
        print(f"❌ Error: El archivo {file_path} no existe")
        return False
    
    backup_file(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 1. Reducir MinConfidenceThreshold de 0.6 a 0.45
    
    # Cambiar el valor predeterminado en la definición de la propiedad
    content = re.sub(r'(MinConfidenceThreshold\s*=\s*)0\.6', r'\g<1>0.45', content)
    
    # Cambiar cualquier comentario que mencione el valor
    content = re.sub(r'Umbral de confianza \(MinConfidenceThreshold\) de 0\.6', 'Umbral de confianza (MinConfidenceThreshold) de 0.45', content)
    
    # Cambiar en la clase VotingSystem
    content = re.sub(r'(public double MinConfidenceThreshold\s*{\s*get;\s*set;\s*}\s*=\s*)0\.6', r'\g<1>0.45', content)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    
    print(f"✅ Umbral de confianza en {file_path} reducido con éxito:")
    print("   - MinConfidenceThreshold reducido de 0.6 a 0.45")
    
    return True

def analyze_database_connectivity():
    """Analiza la base de datos para verificar la conectividad y recomendar soluciones."""
    db_path = 'dqn_learning.db'
    
    if not os.path.exists(db_path):
        print(f"❌ Error: La base de datos {db_path} no existe")
        return False
    
    print("🔍 Analizando la base de datos para problemas de conectividad...")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Verificar si las tablas existen
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"📊 Tablas encontradas: {', '.join(tables)}")
        
        # Verificar registros en operation_results
        if 'operation_results' in tables:
            cursor.execute("SELECT COUNT(*) FROM operation_results")
            count = cursor.fetchone()[0]
            print(f"💰 Registros en operation_results: {count}")
            
            if count == 0:
                print("\n🚨 PROBLEMA DETECTADO: La tabla operation_results está vacía")
                print("   Esto indica un problema de comunicación TCP entre DeepQ.py y DataFeeder.cs")
                print("   Recomendación: Usar el script fix_tcp_connection.py para diagnosticar y reparar la conexión.")
        else:
            print("❌ La tabla operation_results no existe")
        
        # Verificar registros en signals
        if 'signals' in tables:
            cursor.execute("SELECT COUNT(*) FROM signals")
            count = cursor.fetchone()[0]
            print(f"🔔 Registros en signals: {count}")
            
            if count > 0:
                # Analizar distribución de acciones
                cursor.execute("""
                SELECT action, COUNT(*) as count, 
                       COUNT(*) * 100.0 / (SELECT COUNT(*) FROM signals) as percentage
                FROM signals
                GROUP BY action
                ORDER BY action
                """)
                rows = cursor.fetchall()
                
                print("\n📊 Distribución de acciones:")
                action_map = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
                for row in rows:
                    action = action_map.get(row[0], str(row[0]))
                    count = row[1]
                    percentage = row[2]
                    print(f"   {action}: {count} ({percentage:.2f}%)")
                
                # Detectar tendencia conservadora
                hold_percent = 0
                for row in rows:
                    if row[0] == 0:  # Hold
                        hold_percent = row[2]
                
                if hold_percent > 70:
                    print(f"\n🚨 TENDENCIA CONSERVADORA DETECTADA: {hold_percent:.2f}% de Hold")
                    print("   Esto confirma que el robot es demasiado conservador.")
        else:
            print("❌ La tabla signals no existe")
            
        conn.close()
    except Exception as e:
        print(f"❌ Error al analizar la base de datos: {e}")
        return False
    
    return True

def update_changelog():
    """Actualiza el archivo changelog.md con la información de los cambios realizados."""
    changelog_path = 'changelog.md'
    
    # Crear el archivo si no existe
    if not os.path.exists(changelog_path):
        with open(changelog_path, 'w', encoding='utf-8') as file:
            file.write("# Changelog\n\n")
    
    # Leer el contenido actual
    with open(changelog_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Preparar la entrada del changelog
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry = f"""
## [1.1.36] - {timestamp}

### Modificado
- Aumentado el valor de entropía en DeepQ.py de 0.25 a 0.35 para incrementar la exploración
- Aumentada la exploración forzada en DeepQ.py de 15% a 25%
- Reducido el umbral de confianza (MinConfidenceThreshold) en DataFeeder.cs de 0.6 a 0.45
- Estas modificaciones buscan hacer que el robot sea menos conservador y abra más operaciones

### Arreglado
- Identificado problema de conexión TCP que impide retroalimentación adecuada
- Añadido script fix_tcp_connection.py para diagnosticar y reparar problemas de conexión
"""

    # Añadir la entrada al inicio o al final
    # Según instrucciones, añadir al final para mantener orden cronológico ascendente
    content += entry
    
    # Guardar el changelog actualizado
    with open(changelog_path, 'w', encoding='utf-8') as file:
        file.write(content)
    
    print(f"✅ Changelog actualizado: {changelog_path}")
    
    return True

def update_version_number():
    """Actualiza el número de versión en DeepQ.py y otros archivos relevantes."""
    # Actualizar en DeepQ.py
    file_path = 'DeepQ.py'
    
    if not os.path.exists(file_path):
        print(f"❌ Error: El archivo {file_path} no existe")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Buscar patrones como: VERSION = "1.1.35"
    content = re.sub(r'VERSION\s*=\s*"1\.1\.35"', 'VERSION = "1.1.36"', content)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    
    print(f"✅ Versión actualizada en {file_path}: 1.1.36")
    
    return True

def main():
    """Función principal que ejecuta todas las modificaciones."""
    print("\n" + "="*50)
    print(" 🛠️  OPTIMIZADOR DE EXPLORACIÓN Y CONECTIVIDAD 🛠️")
    print("="*50 + "\n")
    
    print("Este script va a modificar parámetros clave para hacer que el robot sea menos conservador")
    print("y analizará problemas de conectividad que podrían estar afectando el aprendizaje.\n")
    
    confirm = input("¿Desea continuar? (S/N): ").strip().upper()
    if confirm != 'S':
        print("Operación cancelada.")
        return
    
    success = True
    
    # Analizar la base de datos primero para entender los problemas
    analyze_database_connectivity()
    
    # Modificar los archivos
    if not modify_deepq_parameters():
        success = False
    
    if not modify_datafeeder_parameters():
        success = False
    
    # Actualizar el changelog y versión
    if success:
        update_changelog()
        update_version_number()
    
    # Resumen final
    if success:
        print("\n" + "="*50)
        print(" ✅ OPTIMIZACIÓN COMPLETADA EXITOSAMENTE ✅")
        print("="*50)
        print("\nRecomendaciones:")
        print("1. Reinicie NinjaTrader para aplicar los cambios en DataFeeder.cs")
        print("2. Ejecute DeepQ.py con los nuevos parámetros")
        print("3. Para diagnosticar problemas de conexión TCP, utilice:")
        print("   python fix_tcp_connection.py test --port 5591")
    else:
        print("\n" + "="*50)
        print(" ❌ OPTIMIZACIÓN COMPLETADA CON ERRORES ❌")
        print("="*50)
        print("\nRevise los mensajes de error y corrija los problemas manualmente.")

if __name__ == "__main__":
    main()
