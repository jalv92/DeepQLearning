# Changelog
[2025-04-04 09:27:00]

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
