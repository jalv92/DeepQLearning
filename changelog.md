
## v1.0.1 - 2025-04-05 14:45:48

### Mejoras
- Completado el ciclo de retroalimentaci髇 para el aprendizaje
- Agregadas experiencias reales basadas en resultados de operaciones
- Optimizado el sistema de recompensas para mejorar el aprendizaje

# Changelog del Proyecto DeepQLearning

## [1.1.33] - 2025-04-05 12:04
### Diagnosticado
- Identificado problema cr铆tico de conexi贸n con el puerto 5591 (servidor de resultados)
- El sistema est谩 enviando se帽ales de trading pero no recibe retroalimentaci贸n de resultados
- Base de datos sin registros en tablas operation_results y real_experiences
### A帽adido
- Script de diagn贸stico test_tcp_connection.py para verificar conexiones TCP
- Herramienta fix_deepq_connection.py para diagnosticar y solucionar problemas de conexi贸n
- An谩lisis detallado del problema en analisis_completo.md
### Recomendado
- Ejecutar simult谩neamente DeepQ.py y DataFeeder en NinjaTrader
- Verificar que DataFeeder.cs tenga configurado correctamente el puerto 5591
- Utilizar fix_deepq_connection.py para simular el env铆o de resultados de operaciones

## [1.1.34] - 2025-04-05 12:40
### Mejorado
- Implementaci贸n robusta de la clase TCPResultsClient con manejo de mensajes fragmentados
- Sistema de heartbeat para detectar y reconectar autom谩ticamente conexiones perdidas
- Mecanismo de entrop铆a adaptativa para mejor balance entre exploraci贸n y explotaci贸n
### A帽adido
- Sistema de modos de operaci贸n: NORMAL y SIMULACI脫N dependiendo del estado de las conexiones
- Verificaci贸n de conexiones al inicio para determinar el modo de funcionamiento
- Mejoras en el procesamiento de mensajes de resultados para mayor robustez
### Eliminado
- Todas las referencias a TensorBoard para simplificar el sistema

## [1.1.35] - 2025-04-05 12:49
### Eliminado
- Scripts de an谩lisis innecesarios: analyze_tensorboard.py, check_tensorboard.py, analyze_model.py, evaluate_checkpoints.py
- Carpeta completa de tensorboard_logs/ con todas sus subcarpetas
- Carpetas de sesiones de entrenamiento antiguas (training_v*)
- Carpeta de resultados de an谩lisis estad铆sticos (analysis_results/)
### Optimizado
- Limpieza general del proyecto para mejorar la claridad y mantenibilidad
- Reducci贸n significativa del tama帽o del proyecto eliminando archivos auxiliares no esenciales

## [1.1.36] - 2025-04-05 13:25
### Optimizado
- Reducci贸n dr谩stica de mensajes de depuraci贸n repetitivos en DataFeeder.cs
- Eliminaci贸n de mensajes peri贸dicos sobre el contador de operaciones diario
- Optimizaci贸n de mensajes de monitoreo de ATM Strategies (intervalo aumentado de 20 a 200 barras)
- Reducci贸n de frecuencia en mensajes de depuraci贸n del indicador estrategia (de 50-100 a 100-500 barras)
### Mejorado
- Sistema de priorizaci贸n de mensajes para mostrar solo informaci贸n cr铆tica o relevante
- Mayor legibilidad de logs en NinjaTrader al eliminar ruido innecesario
- Implementaci贸n de intervalos de tiempo m谩s amplios para mensajes de debug importantes
- Mensajes de verificaci贸n de SL/TP solo se muestran cuando hay problemas, no continuamente

## [1.1.37] - 2025-04-05 13:48
### Modificado
- Eliminado el entropy_scheduler de DeepQ.py y reemplazado por un valor fijo de entrop铆a (0.25)
- Simplificaci贸n del c贸digo al eliminar la clase EntropyScheduler y sus m茅todos relacionados
- Eliminadas las llamadas a entropy_scheduler.step() en el bucle principal de entrenamiento
### Mejorado
- Mayor estabilidad en el entrenamiento del modelo al usar un valor constante de entrop铆a
- C贸digo m谩s limpio y mantenible al eliminar l贸gica de decaimiento innecesaria
- Mejor desempe帽o del modelo con valor de entrop铆a optimizado

## [1.1.36] - 2025-04-05 14:07:04

### Modificado
- Aumentado el valor de entrop铆a en DeepQ.py de 0.25 a 0.35 para incrementar la exploraci贸n
- Aumentada la exploraci贸n forzada en DeepQ.py de 15% a 25%
- Reducido el umbral de confianza (MinConfidenceThreshold) en DataFeeder.cs de 0.6 a 0.45
- Estas modificaciones buscan hacer que el robot sea menos conservador y abra m谩s operaciones

### Arreglado
- Identificado problema de conexi贸n TCP que impide retroalimentaci贸n adecuada
- A帽adido script fix_tcp_connection.py para diagnosticar y reparar problemas de conexi贸n
