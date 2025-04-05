
## v1.0.1 - 2025-04-05 14:45:48

### Mejoras
- Completado el ciclo de retroalimentaci�n para el aprendizaje
- Agregadas experiencias reales basadas en resultados de operaciones
- Optimizado el sistema de recompensas para mejorar el aprendizaje

# Changelog del Proyecto DeepQLearning

## [1.1.33] - 2025-04-05 12:04
### Diagnosticado
- Identificado problema crítico de conexión con el puerto 5591 (servidor de resultados)
- El sistema está enviando señales de trading pero no recibe retroalimentación de resultados
- Base de datos sin registros en tablas operation_results y real_experiences
### Añadido
- Script de diagnóstico test_tcp_connection.py para verificar conexiones TCP
- Herramienta fix_deepq_connection.py para diagnosticar y solucionar problemas de conexión
- Análisis detallado del problema en analisis_completo.md
### Recomendado
- Ejecutar simultáneamente DeepQ.py y DataFeeder en NinjaTrader
- Verificar que DataFeeder.cs tenga configurado correctamente el puerto 5591
- Utilizar fix_deepq_connection.py para simular el envío de resultados de operaciones

## [1.1.34] - 2025-04-05 12:40
### Mejorado
- Implementación robusta de la clase TCPResultsClient con manejo de mensajes fragmentados
- Sistema de heartbeat para detectar y reconectar automáticamente conexiones perdidas
- Mecanismo de entropía adaptativa para mejor balance entre exploración y explotación
### Añadido
- Sistema de modos de operación: NORMAL y SIMULACIÓN dependiendo del estado de las conexiones
- Verificación de conexiones al inicio para determinar el modo de funcionamiento
- Mejoras en el procesamiento de mensajes de resultados para mayor robustez
### Eliminado
- Todas las referencias a TensorBoard para simplificar el sistema

## [1.1.35] - 2025-04-05 12:49
### Eliminado
- Scripts de análisis innecesarios: analyze_tensorboard.py, check_tensorboard.py, analyze_model.py, evaluate_checkpoints.py
- Carpeta completa de tensorboard_logs/ con todas sus subcarpetas
- Carpetas de sesiones de entrenamiento antiguas (training_v*)
- Carpeta de resultados de análisis estadísticos (analysis_results/)
### Optimizado
- Limpieza general del proyecto para mejorar la claridad y mantenibilidad
- Reducción significativa del tamaño del proyecto eliminando archivos auxiliares no esenciales

## [1.1.36] - 2025-04-05 13:25
### Optimizado
- Reducción drástica de mensajes de depuración repetitivos en DataFeeder.cs
- Eliminación de mensajes periódicos sobre el contador de operaciones diario
- Optimización de mensajes de monitoreo de ATM Strategies (intervalo aumentado de 20 a 200 barras)
- Reducción de frecuencia en mensajes de depuración del indicador estrategia (de 50-100 a 100-500 barras)
### Mejorado
- Sistema de priorización de mensajes para mostrar solo información crítica o relevante
- Mayor legibilidad de logs en NinjaTrader al eliminar ruido innecesario
- Implementación de intervalos de tiempo más amplios para mensajes de debug importantes
- Mensajes de verificación de SL/TP solo se muestran cuando hay problemas, no continuamente

## [1.1.37] - 2025-04-05 13:48
### Modificado
- Eliminado el entropy_scheduler de DeepQ.py y reemplazado por un valor fijo de entropía (0.25)
- Simplificación del código al eliminar la clase EntropyScheduler y sus métodos relacionados
- Eliminadas las llamadas a entropy_scheduler.step() en el bucle principal de entrenamiento
### Mejorado
- Mayor estabilidad en el entrenamiento del modelo al usar un valor constante de entropía
- Código más limpio y mantenible al eliminar lógica de decaimiento innecesaria
- Mejor desempeño del modelo con valor de entropía optimizado

## [1.1.36] - 2025-04-05 14:07:04

### Modificado
- Aumentado el valor de entropía en DeepQ.py de 0.25 a 0.35 para incrementar la exploración
- Aumentada la exploración forzada en DeepQ.py de 15% a 25%
- Reducido el umbral de confianza (MinConfidenceThreshold) en DataFeeder.cs de 0.6 a 0.45
- Estas modificaciones buscan hacer que el robot sea menos conservador y abra más operaciones

### Arreglado
- Identificado problema de conexión TCP que impide retroalimentación adecuada
- Añadido script fix_tcp_connection.py para diagnosticar y reparar problemas de conexión
