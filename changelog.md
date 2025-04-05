# Changelog
[2025-04-04 21:08:00]

## [1.1.22] - 2025-04-04 21:08:00

### Añadido
- Creado archivo README.md con documentación completa del sistema, incluyendo diagramas explicativos, instrucciones de instalación y configuración
- Creado archivo requirements.txt con todas las dependencias necesarias para el proyecto
- Mejorada la documentación general del sistema para facilitar su comprensión y uso


## [1.1.21] - 2025-04-04 19:47:00

### Corregido
- Solucionado definitivamente el error de compilación CS1524 en DataFeeder.cs
- Reestructurado completamente el método ProcessSignalMessage() para eliminar try-catch anidados
- Eliminado bloque catch duplicado que causaba error CS0160
- Reorganizado el código para mayor claridad y mantenibilidad

## [1.1.20] - 2025-04-04 19:42:00

### Corregido
- Solucionado error de compilación CS1524 y CS0160 en DataFeeder.cs
- Reestructurado el método ProcessSignalMessage para eliminar try-catch anidados duplicados
- Corregido problema de bloques catch duplicados que capturaban el mismo tipo de excepción
- Reorganizado el código para mantener la estructura de control de flujo correcta

## [1.1.19] - 2025-04-04 19:40:00

### Corregido
- Solucionado error de compilación CS1524 "Expected catch or finally" en DataFeeder.cs
- Añadido bloque catch faltante en el método ProcessSignalMessage para un bloque try anidado
- Mejorado el manejo de errores en el procesamiento de señales para mayor robustez

## [1.1.18] - 2025-04-04 18:52:15

### Mejorado
- Optimizado significativamente el balance exploración-explotación en el modelo de aprendizaje por refuerzo
- Aumentado el coeficiente de entropía (ent_coef) de 0.05 a 0.25 para fomentar exploración mucho más agresiva
- Implementado mecanismo de forzado de exploración periódica que selecciona acciones aleatorias en el 15% de las decisiones
- Aumentada la cantidad de entrenamiento por ciclo de 10 a 50 timesteps para un aprendizaje más profundo
- Mejorado el procesamiento de mensajes de resultado de operaciones con manejo de errores extenso y detallado
- Implementado soporte para distintos formatos de fecha/hora en los mensajes de resultado para mayor robustez
- Añadido logging detallado en el procesamiento de operaciones para facilitar la depuración

### Corregido
- Solucionado problema donde el formato incorrecto de mensajes impedía la comunicación bidireccional
- Mejorada la validación de formato de mensajes en DataFeeder.cs con comprobaciones exhaustivas
- Corregido manejo de strings en Operation.ToResultMessage para asegurar formato consistente en fechas y valores numéricos
- Resuelto problema de interpretación incorrecta de los campos de señal en DataFeeder.cs

## [1.1.17] - 2025-04-04 17:02:00

### Corregido
- Solucionado problema crítico de comunicación entre DeepQ.py y DataFeeder.cs
- Mejorado el procesamiento de mensajes en DataFeeder.cs con validación de formato más robusta
- Corregida la interpretación del formato de mensajes para identificar correctamente el SignalId
- Añadidos logs detallados en DeepQ.py para verificar el envío de señales
- Implementada verificación de formato UUID para SignalId en DataFeeder.cs

## [1.1.16] - 2025-04-04 16:39:00

### Optimizado
- Completado el archivo DeepQ.py con todas las funcionalidades necesarias para el sistema de retroalimentación
- Mejorada la función receive_and_process_data para manejar resultados de operaciones
- Reforzado el sistema de procesamiento de resultados con mejor manejo de errores
- Perfeccionado el sistema de actualización de experiencias pendientes con recompensas reales

## [1.1.15] - 2025-04-04 16:03:00

### Añadido
- Implementado sistema completo de retroalimentación para aprendizaje por refuerzo en trading
- Creado servidor TCP en puerto 5591 en DataFeeder.cs para enviar resultados de operaciones
- Añadido seguimiento detallado de operaciones con IDs únicos para señales y operaciones
- Implementado cliente TCP en DeepQ.py para recibir resultados de operaciones
- Ampliado el esquema de la base de datos SQLite con nuevas tablas:
  - Tabla 'signals' para almacenar señales emitidas
  - Tabla 'operation_results' para guardar resultados reales de operaciones
  - Tabla 'real_experiences' para experiencias con recompensas híbridas
- Desarrollado sistema de recompensas híbrido que combina simuladas y reales
- Añadido mecanismo de ponderación adaptativa que da más peso a recompensas reales con el tiempo
- Implementado sistema de experiencias pendientes mientras se esperan resultados reales
- Creada función sigmoide para normalizar P&L a recompensas en rango [-1, 1]

### Cambiado
- Requerido formato de mensajes para incluir SignalId como identificador único
- Modificado el flujo de trabajo del agente para gestionar recompensas diferidas
- Reestructurado el proceso de entrenamiento para incluir experiencias reales
- Actualizada la versión del software a 1.1.15

## [1.1.14] - 2025-04-04 15:28:00

### Mejorado
- Modificados parámetros para optimizar la generación de señales de trading:
  - Reducida la recompensa por hold de 0.1 a 0.01 para evitar que el modelo se atasque
  - Aumentada la recompensa por operaciones exitosas de 1.0 a 1.5
  - Añadida penalización de -0.1 por operar cuando no hay cambio de precio
  - Aumentado el coeficiente de entropía en el modelo PPO de 0.01 a 0.05 para fomentar mayor exploración
  - Reducido el umbral de confianza mínima de 0.7 a 0.6 para permitir más operaciones
- Actualizada la versión del software a 1.1.14

## [1.1.13] - 2025-04-04 14:57:30

### Cambiado
- Eliminados completamente los parámetros StopLossPercent y TakeProfitPercent del panel de configuración
- Reducido el intervalo mínimo entre operaciones de 5 minutos a 2 minutos para permitir más operaciones
- Simplificado el código de ejecución de operaciones eliminando cálculos de niveles SL/TP innecesarios
- Mejorado el rendimiento general al eliminar código redundante relacionado con la gestión de stop loss y take profit


## [1.1.12] - 2025-04-04 14:38:00

### Corregido
- Eliminadas todas las referencias a modificación manual de stop loss y take profit
- Eliminado código que intentaba modificar los niveles de stop loss/take profit en las estrategias ATM
- Se confía completamente en las plantillas ATM preconfiguradas en NinjaTrader
- Corregidos mensajes de log para mostrar información correcta sin referencias a valores SL/TP específicos
- Simplificado el mecanismo para evitar problemas con valores que no se utilizan

## [1.1.11] - 2025-04-04 14:32:00

### Corregido
- Solucionado error "Stop price can't be changed above the market" en órdenes de venta
- Mejorado el cálculo de los precios de stop loss y take profit para respetar las reglas del mercado
- Añadido log detallado del cálculo de precios para facilitar la depuración
- Implementada lógica más clara para diferenciar el cálculo de SL/TP entre operaciones LONG y SHORT
- Corregida la dirección del stop loss para posiciones SHORT (debe estar por encima del precio de entrada)

## [1.1.10] - 2025-04-04 14:10:00

### Corregido
- Solucionado problema de compilación relacionado con la variable 'lastOcoId'
- Mantenida la variable lastOcoId para compatibilidad con métodos existentes
- Eliminadas referencias innecesarias a OCO mientras se mantiene la compatibilidad con código existente

## [1.1.9] - 2025-04-04 14:05:00

### Mejorado
- Implementado sistema ATM Strategy nativo de NinjaTrader para gestión de órdenes
- Reemplazado el enfoque manual de órdenes OCO por ATM Strategies, que gestiona automáticamente los stop loss y take profit
- Añadido seguimiento y monitoreo de estrategias ATM activas
- Mejorada la gestión del ciclo de vida de las estrategias ATM
- Añadido método MonitorAtmStrategies para verificar el estado de las estrategias en tiempo real
- Mejor integración con la arquitectura nativa de NinjaTrader para mayor fiabilidad

## [1.1.8] - 2025-04-04 13:45:00

### Corregido
- Solucionado error de compilación: Reemplazado OrderType.Stop por OrderType.StopMarket en todas las referencias
- Corregido problema de tipo incorrecto en método VerifyStopLossAndTakeProfit
- Mejorada la verificación de órdenes para ambas direcciones (LONG y SHORT)
- Implementada la misma solución para detección de stop loss y take profit en ambos tipos de posiciones

## [1.1.7] - 2025-04-04 13:40:00

### Corregido
- Implementada solución definitiva para problemas de stop loss y take profit
- Mejorado el proceso de creación de SL/TP utilizando nombres únicos para cada orden con timestamps
- Añadida separación de 100ms entre la colocación de SL y TP para evitar problemas de sincronización
- Mejorada la estructura del método ProcessSignalMessage con validaciones exhaustivas
- Agregados mensajes de depuración detallados para cada paso del proceso de órdenes
- Añadidos más detalles al método VerifyStopLossAndTakeProfit para facilitar la identificación de problemas
- Corregido problema donde las referencias a órdenes podían ser inválidas debido a operaciones asíncronas

## [1.1.6] - 2025-04-04 13:30:00

### Corregido
- Identificado y solucionado problema crítico donde las órdenes enviadas desde Python a NinjaTrader no establecían correctamente stop loss y take profit
- Mejorado el sistema de verificación y reintentos para el establecimiento de SL/TP
- Aumentado el tiempo de espera entre la ejecución de la orden de entrada y la colocación de SL/TP
- Añadida validación más estricta para confirmar la existencia de órdenes de SL/TP activas
- Incorporado mecanismo de protección para evitar órdenes sin gestión de riesgo
- Mejorada la sincronización entre Python (DeepQ) y NinjaTrader para asegurar ejecución completa de órdenes

## [1.1.5] - 2025-04-04 13:02:00

### Mejorado
- Implementada verificación robusta de stop loss y take profit para asegurar su colocación correcta
- Añadido método `VerifyStopLossAndTakeProfit` para confirmar que los SL/TP se establecen correctamente
- Agregado sistema de reintentos para establecer SL/TP (5 intentos con esperas entre cada uno)
- Mejorada la validación de posición activa antes de establecer órdenes de salida
- Corregido problema de lógica en el manejo de condicionales para las posiciones SHORT
- Incrementada la espera después de la orden de entrada para dar tiempo a que se procese (300ms)

## [1.1.4] - 2025-04-04 12:47:00

### Corregido
- Solucionado problema donde el script de Python se quedaba trabado al ejecutarse
- Eliminada duplicación de código que causaba inicialización múltiple de componentes
- Añadida variable de control para evitar iniciar el servidor TCP dos veces
- Corregida versión inconsistente en diferentes partes del código
- Eliminada duplicación de definiciones de colores ANSI
- Mejorada la estabilidad general del script DeepQ.py

## [1.1.3] - 2025-04-04 12:38:00

### Corregido
- Solucionados errores de compilación relacionados con métodos ExitLongStop y ExitShortStop
- Reemplazados métodos inexistentes con ExitLongStopMarket y ExitShortStopMarket
- Añadidos parámetros requeridos para los métodos de salida (barsInProgressIndex, isSimulatedStop, quantity)
- Mantenido el mismo enfoque de separación entre entrada al mercado y establecimiento de SL/TP
- Mejorada la compatibilidad con la API de NinjaTrader para órdenes de salida

## [1.1.2] - 2025-04-04 12:35:00

### Corregido
- Solucionado problema crítico donde no se establecían correctamente los stop loss y take profit
- Implementado nuevo enfoque para la ejecución de órdenes que separa la entrada de mercado de los SL/TP
- Añadida verificación de posición activa antes de establecer stop loss y take profit
- Mejorado el proceso de entrada al mercado usando EnterLong/EnterShort directos en lugar de StopMarket
- Añadido tiempo de espera entre la entrada y la colocación de SL/TP para asegurar que la posición esté activa
- Mensajes de log detallados para facilitar la depuración de órdenes y posiciones
- Implementada la misma solución para operaciones LONG y SHORT

## [1.1.1] - 2025-04-04 12:13:00

### Corregido
- Solución definitiva al problema del contador de operaciones diarias (dailyTradeCount)
- Implementado sistema de verificación para confirmar que las órdenes se enviaron correctamente
- Añadidos métodos VerifyOrdersSubmitted y HasAtLeastOneActiveOrder para validar operaciones
- Solo se incrementa el contador cuando se confirma que hay órdenes activas
- Verificación adicional de órdenes pendientes justo antes de enviar nuevas
- Mejor manejo de errores en el proceso de envío de órdenes
- Reducido riesgo de incrementar el contador sin tener operaciones reales

## [1.1.0] - 2025-04-04 09:58:00

### Corregido
- Solucionado problema donde el contador diario de operaciones alcanzaba el límite sin ejecutar operaciones reales
- Implementado sistema de control para evitar intentos repetitivos de crear órdenes
- Añadida variable orderInProgress para evitar solicitudes simultáneas de operaciones
- Implementado intervalo mínimo de tiempo entre operaciones (5 minutos)
- Añadida verificación de órdenes pendientes antes de intentar crear nuevas
- Mejorado sistema de logs para seguimiento del contador de operaciones
- Solucionados errores de referencia nula en el procesamiento de mensajes

## [1.0.9] - 2025-04-04 09:27:00

### Corregido
- Solucionado error de compilación "The name 'GetOrders' does not exist in the current context"
- Reemplazado llamada a GetOrders() con la propiedad Orders de NinjaTrader
- Añadida verificación adicional para el estado de las órdenes (OrderState.Working)
- Mejora en el método CancelOCOOrders para usar correctamente la API de NinjaTrader

## [1.0.8] - 2025-04-04 09:20:00

### Corregido
- Solucionado error de "OCO ID" duplicados que causaba problemas en la gestión de órdenes
- Implementado sistema de generación de IDs OCO únicos basados en timestamp y valor aleatorio
- Añadido método para cancelar órdenes OCO anteriores antes de crear nuevas
- Mejorado el proceso de creación de órdenes usando EnterLongStopMarket/EnterShortStopMarket
- Agregada variable para almacenar y referenciar el último ID OCO utilizado
- Mejorada la trazabilidad de órdenes con logs detallados que incluyen el ID OCO

## [1.0.7] - 2025-04-03 22:47:00

### Corregido
- Solución definitiva para el error "Index was outside the bounds of the array"
- Evitado el uso de IsValidDataPoint para prevenir errores de índice
- Implementada verificación de conteo en colecciones antes de acceder a elementos
- Reducida la frecuencia de mensajes de error para evitar saturación del log
- Mejorada la inicialización del indicador esperando suficientes barras históricas

## [1.0.6] - 2025-04-03 22:42:00

### Corregido
- Implementada verificación robusta de indicador para prevenir errores de referencia nula
- Añadida restauración automática del indicador cuando se detectan problemas
- Mejorado monitoreo del contador de operaciones para prevenir valores inválidos
- Implementadas comprobaciones de datos válidos antes de acceder a propiedades del indicador
- Añadido registro detallado para solucionar problemas de referencia nula

## [1.0.5] - 2025-04-03 22:37:00

### Corregido
- Solucionado problema donde no se abrían operaciones pero mostraba "Límite diario alcanzado"
- Reinicio de contadores en múltiples puntos para evitar estados incorrectos
- Limpieza de caracteres de nueva línea en mensajes recibidos
- Reducidos valores por defecto de stop loss y take profit para menor riesgo
- Evaluación de operaciones solo para señales de Buy/Sell, no para Hold

## [1.0.4] - 2025-04-03 22:32:00

### Corregido
- Solucionado problema de mensajes repetitivos "Límite diario de operaciones alcanzado"
- Añadida verificación para evitar abrir múltiples posiciones simultáneas
- Implementado control para mostrar mensajes de límite solo una vez por día
- Aumentados valores predeterminados de stop loss (50 ticks) y take profit (100 ticks)

## [1.0.3] - 2025-04-03 22:25:00

### Corregido
- Corregido problema donde stop loss y take profit se eliminaban con cada nueva vela
- Implementado sistema de gestión de riesgo basado en ticks en lugar de porcentaje
- Limitado el máximo de ticks para stop loss (20) y take profit (40) para reducir riesgo
- Mejorado el manejo de errores en la comunicación entre DeepQ.py y DataFeeder.cs

## [1.0.2] - 2025-04-03 22:04:00

### Añadido
- Integración con el indicador HFT_TheStrat_ML para sistema de votación
- Sistema de consensus trading para validar señales
- Mejora en el formato de señales para incluir nivel de confianza
- Mayor robustez en la recepción de señales de trading

### Cambiado
- Ampliación del protocolo de comunicación TCP para soportar datos adicionales
- Modificación del formato de mensajes para incluir datos del indicador TheStrat

## [1.0.1] - 2025-04-03 21:28:00

### Corregido
- Migración de OpenAI Gym a Gymnasium para compatibilidad con Stable-Baselines3
- Actualización de la interfaz de TradingEnv para cumplir con la API de Gymnasium
- Corrección del manejo de datos para evitar errores cuando no hay suficientes características
- Eliminación de dependencia de la biblioteca 'fade' que causaba errores
- Mejora en el procesamiento de datos recibidos por TCP para mayor robustez

### Cambiado
- Mejor manejo de errores en la recepción de datos
- Nombres de columnas más descriptivos en el DataFrame de datos
- Valores por defecto para campos vacíos en los datos recibidos

## [1.0.0] - Versión inicial
- Implementación del algoritmo Deep Q-Network para trading
- Conexión TCP para recibir datos de NinjaTrader
- Interfaz de usuario con colores ANSI
- Sistema de almacenamiento de experiencias en base de datos SQLite
