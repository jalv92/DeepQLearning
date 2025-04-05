#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Xml.Serialization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Globalization;
using System.Threading;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;
#endregion

//This namespace holds Strategies in this folder and is required. Do not change it. 
namespace NinjaTrader.NinjaScript.Strategies
{
public class DataFeeder : Strategy
{
	private TcpListener tcpServer;
	private List<TcpClient> clients = new List<TcpClient>();
	private int portNumber = 5555;
	private bool isServerRunning = false;
        
        // Client para recibir señales de DeepQ.py
        private TcpClient signalClient;
        private NetworkStream signalStream;
        private bool isSignalClientConnected = false;
        private int signalPort = 5590;
        
        // Servidor TCP para enviar resultados de operaciones
        private TcpListener resultsServer;
        private List<TcpClient> resultsClients = new List<TcpClient>();
        private int resultsPort = 5591;
        private bool isResultsServerRunning = false;
        
        // Sistema de seguimiento de operaciones
        private class Operation
        {
            public string OperationId { get; set; }
            public string SignalId { get; set; }
            public DateTime EntryTime { get; set; }
            public double EntryPrice { get; set; }
            public int Direction { get; set; } // 1=Long, 2=Short
            public bool IsOpen { get; set; }
            public DateTime? ExitTime { get; set; }
            public double? ExitPrice { get; set; }
            public double? PnL { get; set; }
            public string CloseReason { get; set; }
            
            public Operation(string operationId, string signalId, DateTime entryTime, double entryPrice, int direction)
            {
                OperationId = operationId;
                SignalId = signalId;
                EntryTime = entryTime;
                EntryPrice = entryPrice;
                Direction = direction;
                IsOpen = true;
                ExitTime = null;
                ExitPrice = null;
                PnL = null;
                CloseReason = null;
            }
            
            public void Close(DateTime exitTime, double exitPrice, double pnl, string closeReason)
            {
                IsOpen = false;
                ExitTime = exitTime;
                ExitPrice = exitPrice;
                PnL = pnl;
                CloseReason = closeReason;
            }
            
            public string ToResultMessage()
            {
                if (!IsOpen && ExitTime.HasValue && ExitPrice.HasValue && PnL.HasValue)
                {
                    // Formato: operationId;signalId;entryTime;entryPrice;direction;exitTime;exitPrice;pnl;closeReason
                    // Usar CultureInfo.InvariantCulture para todos los valores numéricos para asegurar formato consistente
                    string resultMessage = string.Format("{0};{1};{2};{3};{4};{5};{6};{7};{8}",
                        OperationId,
                        SignalId,
                        EntryTime.ToString("yyyy-MM-dd HH:mm:ss.fff", CultureInfo.InvariantCulture),
                        EntryPrice.ToString("0.0000", CultureInfo.InvariantCulture),
                        Direction.ToString(CultureInfo.InvariantCulture),
                        ExitTime.Value.ToString("yyyy-MM-dd HH:mm:ss.fff", CultureInfo.InvariantCulture),
                        ExitPrice.Value.ToString("0.0000", CultureInfo.InvariantCulture),
                        PnL.Value.ToString("0.0000", CultureInfo.InvariantCulture),
                        CloseReason
                    );
                    
                    // No usar Print directamente aquí ya que es un método de instancia de NinjaScript
                    // La clase que use este mensaje puede imprimirlo si es necesario
                    return resultMessage;
                }
                return null;
            }
        }
        
        private Dictionary<string, Operation> operations = new Dictionary<string, Operation>();
        
        // Referencia al indicador TheStrat
        private HFT_TheStrat_ML stratIndicator;
        private double[] stratProbabilities = new double[2] { 0.5, 0.5 };
        private int stratBarType = 0;
        
        // Sistema de votación
        private class VotingSystem
        {
            public double StratMLWeight { get; set; } = 0.6;
            public double DeepQWeight { get; set; } = 0.4;
            public double MinConfidenceThreshold { get; set; } = 0.6; // Reducido de 0.7 a 0.6
            public bool RequireConsensus { get; set; } = true;
            public TimeSpan SignalValidityWindow { get; set; } = TimeSpan.FromMinutes(5);
            
            // Señales recibidas desde DeepQ
            public class Signal
            {
                public string SignalId { get; set; }
                public int Action { get; set; } // 0=Hold, 1=Buy, 2=Sell
                public double Confidence { get; set; }
                public DateTime Timestamp { get; set; }
            }
            
            private List<Signal> signalBuffer = new List<Signal>();
            
            public void AddSignal(string signalId, int action, double confidence)
            {
                Signal newSignal = new Signal
                {
                    SignalId = signalId,
                    Action = action,
                    Confidence = confidence,
                    Timestamp = DateTime.Now
                };
                
                lock (signalBuffer)
                {
                    signalBuffer.Add(newSignal);
                    // Limitar tamaño del buffer
                    if (signalBuffer.Count > 100)
                        signalBuffer.RemoveAt(0);
                }
            }
            
            public Signal GetLatestSignal()
            {
                lock (signalBuffer)
                {
                    return signalBuffer.OrderByDescending(s => s.Timestamp).FirstOrDefault();
                }
            }
            
            public List<Signal> GetRecentSignals()
            {
                lock (signalBuffer)
                {
                    DateTime cutoff = DateTime.Now - SignalValidityWindow;
                    return signalBuffer
                        .Where(s => s.Timestamp >= cutoff)
                        .ToList();
                }
            }
            
            public bool EvaluateSignals(int stratAction, double stratConfidence, out int finalAction, out double finalConfidence)
            {
                finalAction = 0; // Default: Hold
                finalConfidence = 0;
                
                Signal deepQSignal = GetLatestSignal();
                if (deepQSignal == null) return false; // No hay señal de DeepQ
                
                // Calcular puntuación ponderada
                double buyScore = 0, sellScore = 0;
                
                // Puntuación de TheStrat
                if (stratAction == 1) // Long
                    buyScore += stratConfidence * StratMLWeight;
                else if (stratAction == 0) // Short
                    sellScore += stratConfidence * StratMLWeight;
                    
                // Puntuación de DeepQ
                if (deepQSignal.Action == 1) // Buy
                    buyScore += deepQSignal.Confidence * DeepQWeight;
                else if (deepQSignal.Action == 2) // Sell
                    sellScore += deepQSignal.Confidence * DeepQWeight;
                    
                // Tomar decisión
                if (buyScore > sellScore && buyScore > MinConfidenceThreshold)
                {
                    finalAction = 1; // Buy
                    finalConfidence = buyScore;
                }
                else if (sellScore > buyScore && sellScore > MinConfidenceThreshold)
                {
                    finalAction = 2; // Sell
                    finalConfidence = sellScore;
                }
                
                // Verificar consensus si es requerido
                bool consensus = !RequireConsensus || 
                                (stratAction == 1 && deepQSignal.Action == 1) || // Ambos long
                                (stratAction == 0 && deepQSignal.Action == 2);   // Ambos short
                
                return finalAction != 0 && consensus;
            }
        }
        
        private VotingSystem votingSystem = new VotingSystem();
        
        // Parámetros de gestión de riesgo
        [NinjaScriptProperty]
        [Display(Name = "Cantidad por defecto", Description = "Número de contratos por operación", Order = 1, GroupName = "Gestión de Riesgo")]
        public int DefaultQuantity { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Máx. operaciones diarias", Description = "Límite de operaciones por día", Order = 4, GroupName = "Gestión de Riesgo")]
        public int MaxDailyTrades { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Consenso requerido", Description = "Requerir consenso entre modelos", Order = 5, GroupName = "Configuración Consensus")]
        public bool RequireConsensus { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Umbral de confianza", Description = "Confianza mínima para operar", Order = 6, GroupName = "Configuración Consensus")]
        public double MinConfidenceThreshold { get; set; }
        
        // Contador de operaciones diarias y control de mensajes
        private int dailyTradeCount = 0;
        private DateTime lastTradeDate = DateTime.MinValue;
        private bool limitMessagePrinted = false;
        private DateTime limitMessageDate = DateTime.MinValue;

        // Método para iniciar el servidor TCP de resultados
        private void StartResultsServer()
        {
            try
            {
                resultsServer = new TcpListener(IPAddress.Loopback, resultsPort);
                resultsServer.Start();
                isResultsServerRunning = true;
                
                // Iniciar un hilo para aceptar conexiones
                Task.Run(() => AcceptResultsClientsAsync());
                
                Print($"DataFeeder: Servidor TCP de resultados iniciado en puerto {resultsPort}");
            }
            catch (Exception ex)
            {
                Print($"DataFeeder: Error al iniciar servidor TCP de resultados: {ex.Message}");
                isResultsServerRunning = false;
            }
        }
        
        // Método para aceptar clientes en el servidor de resultados
        private async Task AcceptResultsClientsAsync()
        {
            while (isResultsServerRunning)
            {
                try
                {
                    TcpClient client = await resultsServer.AcceptTcpClientAsync();
                    lock (resultsClients)
                    {
                        resultsClients.Add(client);
                    }
                    Print($"DataFeeder: Nuevo cliente conectado al servidor de resultados. Total: {resultsClients.Count}");
                }
                catch (Exception ex)
                {
                    if (isResultsServerRunning)
                    {
                        Print($"DataFeeder: Error al aceptar cliente en servidor de resultados: {ex.Message}");
                    }
                }
            }
        }
        
        // Método para enviar resultados a los clientes conectados
        private void SendResultToClients(string message)
        {
            if (!isResultsServerRunning || resultsClients.Count == 0) return;
            
            byte[] data = Encoding.ASCII.GetBytes(message + "\n");
            
            lock (resultsClients)
            {
                // Crear una lista de clientes que se desconectaron
                List<TcpClient> disconnectedClients = new List<TcpClient>();
                
                foreach (var client in resultsClients)
                {
                    try
                    {
                        if (client.Connected)
                        {
                            NetworkStream stream = client.GetStream();
                            if (stream.CanWrite)
                            {
                                stream.Write(data, 0, data.Length);
                            }
                            else
                            {
                                disconnectedClients.Add(client);
                            }
                        }
                        else
                        {
                            disconnectedClients.Add(client);
                        }
                    }
                    catch
                    {
                        disconnectedClients.Add(client);
                    }
                }
                
                // Eliminar clientes desconectados
                foreach (var client in disconnectedClients)
                {
                    resultsClients.Remove(client);
                    client.Close();
                }
                
                if (disconnectedClients.Count > 0)
                {
                    Print($"DataFeeder: {disconnectedClients.Count} cliente(s) de resultados desconectado(s). Restantes: {resultsClients.Count}");
                }
            }
        }
        
        // Método para iniciar el servidor TCP
        private void StartServer()
        {
            try
            {
                // Reiniciar contadores y variables de control
                dailyTradeCount = 0;
                lastTradeDate = DateTime.Now;
                limitMessagePrinted = false;
                limitMessageDate = DateTime.MinValue;
                
                tcpServer = new TcpListener(IPAddress.Loopback, portNumber);
                tcpServer.Start();
                isServerRunning = true;
                
                // Iniciar un hilo para aceptar conexiones
                Task.Run(() => AcceptClientsAsync());
                
                Print($"DataFeeder: Servidor TCP iniciado en puerto {portNumber}");
                
                // Iniciar servidor de resultados
                StartResultsServer();
                
                // Iniciar cliente para recibir señales
                StartSignalClient();
                
                // Inicializar el indicador TheStrat
                stratIndicator = HFT_TheStrat_ML();
                Print("DataFeeder: Indicador HFT_TheStrat_ML inicializado");
                
                // Configurar sistema de votación
                votingSystem.RequireConsensus = RequireConsensus;
                votingSystem.MinConfidenceThreshold = MinConfidenceThreshold;
                
                Print($"DataFeeder: Sistema de votación configurado. Consensus: {RequireConsensus}, Umbral: {MinConfidenceThreshold}");
                Print($"DataFeeder: Contador de operaciones diario iniciado en: {dailyTradeCount} de {MaxDailyTrades}");
            }
            catch (Exception ex)
            {
                Print($"DataFeeder: Error al iniciar servidor TCP: {ex.Message}");
                isServerRunning = false;
            }
        }
        
        // Método para iniciar el cliente que recibe señales
        private void StartSignalClient()
        {
            try
            {
                signalClient = new TcpClient();
                signalClient.BeginConnect(IPAddress.Loopback.ToString(), signalPort, new AsyncCallback(SignalClientConnectCallback), null);
                Print($"DataFeeder: Iniciando conexión con servidor de señales en puerto {signalPort}");
            }
            catch (Exception ex)
            {
                Print($"DataFeeder: Error al iniciar cliente de señales: {ex.Message}");
            }
        }
        
        // Callback para la conexión del cliente de señales
        private void SignalClientConnectCallback(IAsyncResult ar)
        {
            try
            {
                signalClient.EndConnect(ar);
                signalStream = signalClient.GetStream();
                isSignalClientConnected = true;
                
                // Iniciar recepción de datos
                byte[] buffer = new byte[4096];
                signalStream.BeginRead(buffer, 0, buffer.Length, new AsyncCallback(SignalDataReceived), buffer);
                
                Print("DataFeeder: Conectado con servidor de señales");
            }
            catch (Exception ex)
            {
                Print($"DataFeeder: Error al conectar con servidor de señales: {ex.Message}");
                isSignalClientConnected = false;
                
                // Intentar reconectar después de un retraso
                Task.Delay(5000).ContinueWith(t => StartSignalClient());
            }
        }
        
        // Método para procesar datos recibidos del servidor de señales
        private void SignalDataReceived(IAsyncResult ar)
        {
            if (!isSignalClientConnected) return;
            
            try
            {
                byte[] buffer = (byte[])ar.AsyncState;
                int bytesRead = signalStream.EndRead(ar);
                
                if (bytesRead > 0)
                {
                    string message = Encoding.ASCII.GetString(buffer, 0, bytesRead);
                    ProcessSignalMessage(message);
                    
                    // Continuar recibiendo datos
                    signalStream.BeginRead(buffer, 0, buffer.Length, new AsyncCallback(SignalDataReceived), buffer);
                }
                else
                {
                    // Conexión cerrada por el servidor
                    isSignalClientConnected = false;
                    Print("DataFeeder: Conexión con servidor de señales cerrada");
                    
                    // Intentar reconectar
                    Task.Delay(5000).ContinueWith(t => StartSignalClient());
                }
            }
            catch (Exception ex)
            {
                Print($"DataFeeder: Error al recibir datos de señales: {ex.Message}");
                isSignalClientConnected = false;
                
                // Intentar reconectar
                Task.Delay(5000).ContinueWith(t => StartSignalClient());
            }
        }
        
        // Método para procesar mensajes recibidos del servidor de señales con manejo mejorado de errores
        private void ProcessSignalMessage(string message)
        {
            if (string.IsNullOrEmpty(message))
            {
                Print("DataFeeder: Mensaje recibido vacío o nulo, ignorando");
                return;
            }
            
            try
            {
                // Limpiar el mensaje de caracteres no deseados
                message = message.Replace("\n", "").Replace("\r", "").Trim();
                
                // Formato esperado: SignalId;Action;Confidence;Timestamp
                string[] parts = message.Split(';');
                
                if (parts == null)
                {
                    Print("DataFeeder: Error al dividir el mensaje en partes, array nulo");
                    return;
                }
                
                if (parts.Length < 4) // Requerimos exactamente 4 partes
                {
                    Print($"DataFeeder: Formato de mensaje incorrecto, partes insuficientes: {parts.Length}, mensaje: '{message}'");
                    return;
                }
                
                // Extraer cada parte del mensaje con manejo específico
                string signalId = parts[0].Trim();
                string actionStr = parts[1].Trim();
                string confidenceStr = parts[2].Trim();
                string timestampStr = parts[3].Trim();
                
                // Validar que el ID es un GUID
                if (!Guid.TryParse(signalId, out _))
                {
                    Print($"DataFeeder: El ID de señal no es un UUID válido: '{signalId}'");
                    return;
                }
                
                // Convertir la acción a float con manejo detallado de errores
                float actionFloat;
                if (!float.TryParse(actionStr, NumberStyles.Any, CultureInfo.InvariantCulture, out actionFloat))
                {
                    Print($"DataFeeder: No se pudo convertir la acción a float: '{actionStr}'");
                    return;
                }
                
                // Convertir la confianza a double con manejo detallado de errores
                double confidence;
                if (!double.TryParse(confidenceStr, NumberStyles.Any, CultureInfo.InvariantCulture, out confidence))
                {
                    Print($"DataFeeder: No se pudo convertir la confianza a double: '{confidenceStr}'");
                    return;
                }
                
                // Convertir la acción a entero
                int action = (int)Math.Round(actionFloat);
                
                // Validar que la acción esté en el rango válido
                if (action < 0 || action > 2)
                {
                    Print($"DataFeeder: Acción fuera de rango válido (0-2): {action}");
                    return;
                }

                // Validar que la confianza está en el rango válido
                if (confidence < 0 || confidence > 1)
                {
                    Print($"DataFeeder: Confianza fuera de rango válido (0-1): {confidence}");
                    return;
                }
                
                // Mensaje detallado para debugging
                Print($"DataFeeder: Señal procesada correctamente - ID: {signalId}, Acción: {action}, Confianza: {confidence:P2}, Timestamp: {timestampStr}");
                
                try 
                {
                    // Añadir al sistema de votación - verificar que el sistema esté inicializado
                    if (votingSystem != null)
                    {
                        votingSystem.AddSignal(signalId, action, confidence);
                    }
                    else
                    {
                        Print("DataFeeder: ADVERTENCIA - votingSystem es null, reinicializando");
                        votingSystem = new VotingSystem();
                        votingSystem.AddSignal(signalId, action, confidence);
                    }
                    
                    // Solo evaluar operaciones si la acción no es Hold (0)
                    if (action > 0)
                    {
                        try 
                        {
                            // Evaluar si debemos ejecutar una operación
                            EvaluateAndExecuteTrade();
                        }
                        catch (Exception evalEx)
                        {
                            Print($"DataFeeder: Error al evaluar operación: {evalEx.Message}");
                        }
                    }
                }
                catch (Exception innerEx)
                {
                    Print($"DataFeeder: Error interno al procesar señal: {innerEx.Message}");
                }
            }
            catch (Exception ex)
            {
                Print($"DataFeeder: Error al procesar mensaje de señales: {ex.Message} - Stack: {ex.StackTrace}");
            }
        }
        
        // Método para evaluar señales y ejecutar operaciones
        private void EvaluateAndExecuteTrade()
        {
            // Verificar reseteo diario
            if (DateTime.Now.Date > lastTradeDate.Date)
            {
                dailyTradeCount = 0;
                lastTradeDate = DateTime.Now;
                Print("DataFeeder: Reseteo del contador diario de operaciones");
            }
            
                // Verificar si el contador está en un valor inválido
            if (dailyTradeCount < 0 || dailyTradeCount > 1000)
            {
                Print($"DataFeeder: REINICIO DE CONTADOR DEBIDO A VALOR INVÁLIDO ({dailyTradeCount})");
                dailyTradeCount = 0;
            }
            
            // Debug: Mostrar el contador actual
            if (CurrentBar % 20 == 0) // Mostrar solo de vez en cuando para evitar spam
            {
                Print($"DataFeeder: Contador actual = {dailyTradeCount}, Máximo = {MaxDailyTrades}");
            }
                
            // Verificar límite diario - solo imprimir mensaje una vez por día
            if (dailyTradeCount >= MaxDailyTrades)
            {
                // Verificar si ya hemos impreso el mensaje hoy
                if (!limitMessagePrinted || DateTime.Now.Date > limitMessageDate.Date)
                {
                    Print($"DataFeeder: Límite diario de operaciones alcanzado ({dailyTradeCount}/{MaxDailyTrades})");
                    limitMessagePrinted = true;
                    limitMessageDate = DateTime.Now;
                }
                return;
            }
            
            // Verificar si ya tenemos una posición abierta
            if (Position.MarketPosition != MarketPosition.Flat)
            {
                return; // Ya tenemos una posición abierta, no abrir otra
            }
            
            // Obtener datos del indicador TheStrat
            int stratAction = -1;
            if (stratProbabilities[1] > 0.7) stratAction = 1; // Long
            else if (stratProbabilities[0] > 0.7) stratAction = 0; // Short
            
            double stratConfidence = Math.Max(stratProbabilities[0], stratProbabilities[1]);
            
            // Evaluar señales
            int finalAction;
            double finalConfidence;
            bool shouldTrade = votingSystem.EvaluateSignals(stratAction, stratConfidence, out finalAction, out finalConfidence);
            
            if (shouldTrade)
            {
                // Ejecutar operación
                ExecuteTrade(finalAction, finalConfidence);
            }
        }
        
        // Variables para gestión de ATM Strategies
        private List<string> activeAtmStrategies = new List<string>();
        private string lastOcoId = string.Empty; // Mantener para compatibilidad con métodos existentes
        private bool orderInProgress = false;
        private DateTime lastOrderAttemptTime = DateTime.MinValue;
        private TimeSpan minimumOrderInterval = TimeSpan.FromMinutes(2); // Reducido de 5 a 2 minutos

        // Método para ejecutar operaciones utilizando ATM Strategy de NinjaTrader
        private void ExecuteTrade(int action, double confidence, string signalId = null)
        {
            // Verificar si ya hay órdenes pendientes o si ha pasado muy poco tiempo desde el último intento
            if (orderInProgress || HasPendingOrders())
            {
                Print($"DataFeeder: Ya hay órdenes pendientes, omitiendo nueva operación");
                return;
            }

            // Verificar si ha pasado suficiente tiempo desde el último intento
            if (DateTime.Now - lastOrderAttemptTime < minimumOrderInterval)
            {
                Print($"DataFeeder: Esperando intervalo mínimo entre operaciones ({minimumOrderInterval.TotalMinutes} minutos)");
                return;
            }
            
            // Marcar que estamos en proceso de orden
            orderInProgress = true;
            lastOrderAttemptTime = DateTime.Now;
            
            // Generar ID único para la operación si no se proporcionó un signalId
            if (string.IsNullOrEmpty(signalId))
            {
                signalId = Guid.NewGuid().ToString();
            }
            
            try
            {
                // Obtener precio actual
                double currentPrice = Close[0];
                
                // Usar plantillas ATM preconfiguradas que ya tienen los niveles óptimos de SL/TP
                Print($"DataFeeder: Utilizando plantilla ATM predefinida para Acción: {action}, Precio actual: {currentPrice}");
                
                // Generar IDs únicos para la estrategia ATM
                string atmStrategyId = GetAtmStrategyUniqueId();
                string orderId = GetAtmStrategyUniqueId();
                string operationId = Guid.NewGuid().ToString();
                
                bool atmCreated = false;
                
                // Crear la estrategia ATM según la acción
                if (action == 1) // Buy
                {
                    if (Position.MarketPosition == MarketPosition.Short)
                        ExitShort();
                
                    // Crear estrategia ATM para LONG
                    // El nombre de la plantilla ATM debe estar creado previamente en NinjaTrader
                    // Por ejemplo, debe existir una plantilla llamada "ATM-LONG" en NinjaTrader
                    AtmStrategyCreate(OrderAction.Buy, OrderType.Market, 0, 0, TimeInForce.Day, 
                                    orderId, "ATM-LONG", atmStrategyId, (errorCode, callbackId) => {
                        if (errorCode == ErrorCode.NoError && callbackId == atmStrategyId)
                        {
                            atmCreated = true;
                            Print($"DataFeeder: Estrategia ATM LONG creada con éxito");
                            
                            // Las plantillas ATM ya tienen configurados los niveles de SL/TP
                            // No necesitamos modificarlos manualmente
                            
                            // Añadir a la lista de estrategias activas
                            lock (activeAtmStrategies)
                            {
                                activeAtmStrategies.Add(atmStrategyId);
                            }
                            
                            // Registrar la operación en el sistema de seguimiento
                            Operation newOperation = new Operation(
                                operationId,
                                signalId,
                                DateTime.Now,
                                currentPrice,
                                1 // Direction = Long
                            );
                            
                            lock (operations)
                            {
                                operations[operationId] = newOperation;
                            }
                            
                            // Incrementar contador si la estrategia se creó correctamente
                            dailyTradeCount++;
                            lastTradeDate = DateTime.Now;
                            Print($"DataFeeder: Contador incrementado a {dailyTradeCount}/{MaxDailyTrades}");
                            Print($"DataFeeder: ORDEN LONG ejecutada con ATM - Precio: {currentPrice}, Confianza: {confidence:P2}, OperationId: {operationId}, SignalId: {signalId}");
                        }
                        else
                        {
                            Print($"DataFeeder: Error al crear estrategia ATM LONG: {errorCode}");
                        }
                    });
                }
                else if (action == 2) // Sell
                {
                    if (Position.MarketPosition == MarketPosition.Long)
                        ExitLong();
                
                    // Crear estrategia ATM para SHORT
                    // El nombre de la plantilla ATM debe estar creado previamente en NinjaTrader
                    // Por ejemplo, debe existir una plantilla llamada "ATM-SHORT" en NinjaTrader
                    AtmStrategyCreate(OrderAction.Sell, OrderType.Market, 0, 0, TimeInForce.Day, 
                                    orderId, "ATM-SHORT", atmStrategyId, (errorCode, callbackId) => {
                        if (errorCode == ErrorCode.NoError && callbackId == atmStrategyId)
                        {
                            atmCreated = true;
                            Print($"DataFeeder: Estrategia ATM SHORT creada con éxito");
                            
                            // Las plantillas ATM ya tienen configurados los niveles de SL/TP
                            // No necesitamos modificarlos manualmente
                            
                            // Añadir a la lista de estrategias activas
                            lock (activeAtmStrategies)
                            {
                                activeAtmStrategies.Add(atmStrategyId);
                            }
                            
                            // Registrar la operación en el sistema de seguimiento
                            Operation newOperation = new Operation(
                                operationId,
                                signalId,
                                DateTime.Now,
                                currentPrice,
                                2 // Direction = Short
                            );
                            
                            lock (operations)
                            {
                                operations[operationId] = newOperation;
                            }
                            
                            // Incrementar contador si la estrategia se creó correctamente
                            dailyTradeCount++;
                            lastTradeDate = DateTime.Now;
                            Print($"DataFeeder: Contador incrementado a {dailyTradeCount}/{MaxDailyTrades}");
                            Print($"DataFeeder: ORDEN SHORT ejecutada con ATM - Precio: {currentPrice}, Confianza: {confidence:P2}, OperationId: {operationId}, SignalId: {signalId}");
                        }
                        else
                        {
                            Print($"DataFeeder: Error al crear estrategia ATM SHORT: {errorCode}");
                        }
                    });
                }
            }
            catch (Exception ex)
            {
                Print($"DataFeeder: Error al ejecutar operación con ATM Strategy: {ex.Message}");
            }
            finally
            {
                // Importante: garantizar que siempre liberamos el flag de operación en proceso
                orderInProgress = false;
            }
        }
        
        // Método para procesar cuando una posición se cierra (para retroalimentación)
        protected override void OnPositionUpdate(Position position, double averagePrice, int quantity, MarketPosition marketPosition)
        {
            if (marketPosition == MarketPosition.Flat)
            {
                // Una posición se ha cerrado, buscar la operación correspondiente
                string atm = position.Account != null ? position.Account.Name : string.Empty;
                string closeReason = "Unknown";
                
                // Determinar la razón de cierre basado en patrones comunes
                // En NinjaTrader 8 no podemos acceder directamente al tipo de orden que cerró la posición
                // así que usamos lógica alternativa
                
                // Verificar si hay una estrategia ATM activa
                bool foundAtmStrategy = false;
                foreach (string atmId in activeAtmStrategies)
                {
                    try
                    {
                        if (GetAtmStrategyMarketPosition(atmId) == MarketPosition.Flat)
                        {
                            foundAtmStrategy = true;
                            closeReason = "ATM_Strategy_Closed";
                            break;
                        }
                    }
                    catch { }
                }
                
                if (!foundAtmStrategy)
                {
                    // Si no encontramos una estrategia ATM, podemos intentar inferir la razón
                    // por el precio de cierre en relación con el precio de entrada
                    
                    // Por ahora, asumimos "ManualClose" como valor predeterminado
                    closeReason = "ManualClose";
                }
                
                // Buscar todas las operaciones abiertas que correspondan a la posición cerrada
                lock (operations)
                {
                    var operationsToClose = operations.Values
                        .Where(op => op.IsOpen)
                        .ToList();
                    
                    if (operationsToClose.Any())
                    {
                        double exitPrice = Close[0]; // Usar el precio actual como precio de salida
                        DateTime exitTime = Time[0];
                        
                        foreach (var operation in operationsToClose)
                        {
                            // Calcular PnL
                            double pnl = 0;
                            int positionQuantity = Position.Quantity > 0 ? Position.Quantity : 1; // Evitar división por cero
                            
                            if (operation.Direction == 1) // Long
                            {
                                pnl = (exitPrice - operation.EntryPrice) * positionQuantity;
                            }
                            else if (operation.Direction == 2) // Short
                            {
                                pnl = (operation.EntryPrice - exitPrice) * positionQuantity;
                            }
                            
                            // Cerrar la operación
                            operation.Close(exitTime, exitPrice, pnl, closeReason);
                            
                            // Enviar el resultado a los clientes conectados
                            string resultMessage = operation.ToResultMessage();
                            if (!string.IsNullOrEmpty(resultMessage))
                            {
                                SendResultToClients(resultMessage);
                                Print($"DataFeeder: Enviado resultado de operación - {resultMessage}");
                            }
                        }
                    }
                }
            }
        }
        
        // Método para monitorear estrategias ATM activas
        private void MonitorAtmStrategies()
        {
            if (activeAtmStrategies.Count == 0)
                return;
                
            List<string> completedStrategies = new List<string>();
            
            foreach (string atmStrategyId in activeAtmStrategies)
            {
                try
                {
                    // Verificar si la estrategia está cerrada (posición flat)
                    MarketPosition position = GetAtmStrategyMarketPosition(atmStrategyId);
                    
                    if (position == MarketPosition.Flat)
                    {
                        Print($"DataFeeder: Estrategia ATM {atmStrategyId} ha sido cerrada");
                        
                        // Buscar operaciones asociadas a esta estrategia para enviar resultados
                        try
                        {
                            double avgPrice = GetAtmStrategyPositionAveragePrice(atmStrategyId);
                            int quantity = GetAtmStrategyPositionQuantity(atmStrategyId);
                            double pnl = GetAtmStrategyUnrealizedProfitLoss(atmStrategyId);
                            
                            // La estrategia se ha cerrado, buscar operaciones abiertas y cerrarlas
                            lock (operations)
                            {
                                var operationsToClose = operations.Values
                                    .Where(op => op.IsOpen)
                                    .ToList();
                                
                                if (operationsToClose.Any())
                                {
                                    double exitPrice = Close[0]; // Usar el precio actual como precio de salida
                                    DateTime exitTime = Time[0];
                                    
                                    foreach (var operation in operationsToClose)
                                    {
                                        // Cerrar la operación
                                        operation.Close(exitTime, exitPrice, pnl, "ATM_Strategy_Closed");
                                        
                                        // Enviar el resultado a los clientes conectados
                                        string resultMessage = operation.ToResultMessage();
                                        if (!string.IsNullOrEmpty(resultMessage))
                                        {
                                            SendResultToClients(resultMessage);
                                            Print($"DataFeeder: Enviado resultado de operación - {resultMessage}");
                                        }
                                    }
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            Print($"DataFeeder: Error al procesar cierre de ATM {atmStrategyId}: {ex.Message}");
                        }
                        
                        completedStrategies.Add(atmStrategyId);
                        continue;
                    }
                    
                    // Solo mostrar información detallada ocasionalmente para evitar spam
                    if (CurrentBar % 20 == 0)
                    {
                        double avgPrice = GetAtmStrategyPositionAveragePrice(atmStrategyId);
                        int quantity = GetAtmStrategyPositionQuantity(atmStrategyId);
                        double pnl = GetAtmStrategyUnrealizedProfitLoss(atmStrategyId);
                        
                        Print($"DataFeeder: ATM {atmStrategyId} - Posición: {position}, Cantidad: {quantity}, Precio: {avgPrice}, PnL: {pnl}");
                    }
                }
                catch (Exception ex)
                {
                    Print($"DataFeeder: Error al monitorear estrategia ATM {atmStrategyId}: {ex.Message}");
                    
                    // Añadir a la lista para eliminar si hay error en el monitoreo
                    // Esto evita errores continuos por estrategias que ya no existen
                    completedStrategies.Add(atmStrategyId);
                }
            }
            
            // Eliminar las estrategias completadas de la lista de activas
            if (completedStrategies.Count > 0)
            {
                lock (activeAtmStrategies)
                {
                    foreach (string id in completedStrategies)
                    {
                        activeAtmStrategies.Remove(id);
                    }
                }
            }
        }
        
        // Método mejorado para verificar si los SL/TP están correctamente configurados
        private bool VerifyStopLossAndTakeProfit(double stopLoss, double takeProfit)
        {
            try
            {
                // Verificar que estemos en una posición
                if (Position.MarketPosition == MarketPosition.Flat)
                {
                    Print("VerifyStopLossAndTakeProfit: No hay posición activa");
                    return false;
                }

                // Verificación segura de la colección Orders
                if (Orders == null)
                {
                    Print("VerifyStopLossAndTakeProfit: La colección Orders es null");
                    return false;
                }
                
                // Verificar si hay órdenes de SL/TP trabajando
                bool hasStopLoss = false;
                bool hasTakeProfit = false;
                
                // Imprimir información detallada de las órdenes existentes para debug
                Print($"VerifyStopLossAndTakeProfit: Verificando órdenes para posición {Position.MarketPosition}, buscando SL: {stopLoss}, TP: {takeProfit}");
                
                foreach (Order order in Orders)
                {
                    if (order == null) 
                    {
                        Print("VerifyStopLossAndTakeProfit: Orden null encontrada");
                        continue;
                    }
                    
                    // Mostrar información detallada de cada orden para depuración
                    string orderType = "Desconocido";
                    if (order.OrderType == OrderType.Limit) orderType = "Limit";
                    else if (order.OrderType == OrderType.Market) orderType = "Market";
                    else if (order.OrderType == OrderType.StopMarket) orderType = "StopMarket";
                    else if (order.OrderType == OrderType.StopLimit) orderType = "StopLimit";
                    
                    Print($"Orden: ID={order.Id}, Nombre={order.Name}, Tipo={orderType}, Estado={order.OrderState}, " +
                          $"Acción={order.OrderAction}, LimitPrice={order.LimitPrice}, StopPrice={order.StopPrice}");
                    
                    if (order.OrderState != OrderState.Working)
                    {
                        Print($"VerifyStopLossAndTakeProfit: Orden {order.Id} ignorada por estado {order.OrderState}");
                        continue;
                    }
                    
                    // Verificación específica según el tipo de posición
                    if (Position.MarketPosition == MarketPosition.Long)
                    {
                        // Para posición LONG
                        if (order.OrderAction == OrderAction.Sell && order.OrderType == OrderType.StopMarket &&
                            order.StopPrice > 0 && Math.Abs(order.StopPrice - stopLoss) < TickSize * 3)
                        {
                            hasStopLoss = true;
                            Print($"VerifyStopLossAndTakeProfit: Stop Loss LONG encontrado: {order.StopPrice}");
                        }
                        
                        if (order.OrderAction == OrderAction.Sell && order.OrderType == OrderType.Limit &&
                            order.LimitPrice > 0 && Math.Abs(order.LimitPrice - takeProfit) < TickSize * 3)
                        {
                            hasTakeProfit = true;
                            Print($"VerifyStopLossAndTakeProfit: Take Profit LONG encontrado: {order.LimitPrice}");
                        }
                    }
                    else if (Position.MarketPosition == MarketPosition.Short)
                    {
                        // Para posición SHORT
                        if (order.OrderAction == OrderAction.BuyToCover && order.OrderType == OrderType.StopMarket &&
                            order.StopPrice > 0 && Math.Abs(order.StopPrice - stopLoss) < TickSize * 3)
                        {
                            hasStopLoss = true;
                            Print($"VerifyStopLossAndTakeProfit: Stop Loss SHORT encontrado: {order.StopPrice}");
                        }
                        
                        if (order.OrderAction == OrderAction.BuyToCover && order.OrderType == OrderType.Limit &&
                            order.LimitPrice > 0 && Math.Abs(order.LimitPrice - takeProfit) < TickSize * 3)
                        {
                            hasTakeProfit = true;
                            Print($"VerifyStopLossAndTakeProfit: Take Profit SHORT encontrado: {order.LimitPrice}");
                        }
                    }
                }
                
                // Resultado final - Ambos deben estar configurados
                bool result = hasStopLoss && hasTakeProfit;
                Print($"VerifyStopLossAndTakeProfit: Resultado final: hasStopLoss={hasStopLoss}, hasTakeProfit={hasTakeProfit}, resultado={result}");
                return result;
            }
            catch (Exception ex)
            {
                Print($"VerifyStopLossAndTakeProfit: Error en la verificación: {ex.Message}");
                return false;
            }
        }
        
        // Método para verificar si las órdenes se enviaron correctamente
        private bool VerifyOrdersSubmitted(string ocoId)
        {
            if (Orders == null) return false;
            
            // Esperar un breve momento para que las órdenes se procesen
            System.Threading.Thread.Sleep(100);
            
            // Contar órdenes que pertenecen a este grupo OCO
            int countFound = 0;
            foreach (Order order in Orders)
            {
                if (order != null && 
                    (order.Name != null && order.Name.Contains(ocoId)) || 
                    (order.Oco != null && order.Oco.Contains(ocoId)))
                {
                    countFound++;
                }
            }
            
            // Debería haber al menos 2 órdenes (entrada + SL/TP)
            return countFound >= 2;
        }
        
        // Método para verificar si hay al menos una orden activa
        private bool HasAtLeastOneActiveOrder()
        {
            if (Orders == null) return false;
            
            // Esperar un breve momento para verificar
            System.Threading.Thread.Sleep(100);
            
            foreach (Order order in Orders)
            {
                if (order != null && order.OrderState == OrderState.Working)
                {
                    return true;
                }
            }
            return false;
        }
        
        // Verificar si hay órdenes pendientes
        private bool HasPendingOrders()
        {
            if (Orders == null) return false;
            
            foreach (Order order in Orders)
            {
                if (order != null && order.OrderState == OrderState.Working)
                {
                    return true;
                }
            }
            return false;
        }
        
        // Método para cancelar órdenes OCO anteriores
        private void CancelOCOOrders()
        {
            if (!string.IsNullOrEmpty(lastOcoId))
            {
                // Cancelar todas las órdenes pendientes que pertenecen al último grupo OCO
                foreach (Order order in Orders)
                {
                    if (order.OrderState == OrderState.Working && order.Oco == lastOcoId)
                    {
                        CancelOrder(order);
                        Print($"DataFeeder: Cancelada orden previa del grupo OCO: {lastOcoId}, Order ID: {order.Id}");
                    }
                }
            }
        }
        
        // Método para aceptar clientes de forma asíncrona
        private async Task AcceptClientsAsync()
        {
            while (isServerRunning)
            {
                try
                {
                    TcpClient client = await tcpServer.AcceptTcpClientAsync();
                    lock (clients)
                    {
                        clients.Add(client);
                    }
                    Print($"DataFeeder: Nuevo cliente conectado. Total: {clients.Count}");
                }
                catch (Exception ex)
                {
                    if (isServerRunning)
                    {
                        Print($"DataFeeder: Error al aceptar cliente: {ex.Message}");
                    }
                }
            }
        }
        
        // Método para enviar datos a todos los clientes conectados
        private void SendDataToClients(string message)
        {
            if (!isServerRunning || clients.Count == 0) return;
            
                // Obtener datos del indicador TheStrat con manejo de errores mejorado
            try {
                // Primero verificamos si tenemos suficientes barras históricas
                if (CurrentBar < 20) // Necesitamos al menos 20 barras para que el indicador funcione correctamente
                {
                    // No tenemos suficiente historia todavía, enviamos valores por defecto
                    if (CurrentBar % 10 == 0) // Mostrar mensaje solo ocasionalmente para evitar spam
                        Print($"DataFeeder: Esperando suficientes barras históricas. Actual: {CurrentBar}/20");
                }
                else
                {
                    // Evitar llamar a IsValidDataPoint directamente, que es lo que causa el error
                    // En su lugar, usamos verificaciones más básicas
                    if (stratIndicator != null)
                    {
                        bool hasValidData = true;
                        
                        try {
                            // Verificamos si las series tienen valores
                            if (stratIndicator.ProbShort != null && stratIndicator.ProbShort.Count > 0)
                                stratProbabilities[0] = stratIndicator.ProbShort[0];  // Probabilidad de Short
                            
                            if (stratIndicator.ProbLong != null && stratIndicator.ProbLong.Count > 0)
                                stratProbabilities[1] = stratIndicator.ProbLong[0];   // Probabilidad de Long
                                
                            if (stratIndicator.BARTYPE != null && stratIndicator.BARTYPE.Count > 0)
                                stratBarType = stratIndicator.BARTYPE[0];
                        }
                        catch {
                            hasValidData = false;
                        }
                        
                        if (!hasValidData && CurrentBar % 10 == 0)
                            Print($"DataFeeder: Todavía no hay datos válidos del indicador (CurrentBar: {CurrentBar})");
                    }
                    else
                    {
                        if (CurrentBar % 20 == 0) // Reducir frecuencia de mensajes
                        {
                            Print("DataFeeder: stratIndicator es null - intentando reiniciar");
                            stratIndicator = HFT_TheStrat_ML();
                        }
                    }
                }
            } 
            catch (Exception ex) {
                // Evitar mostrar el mismo error repetidamente
                if (CurrentBar % 50 == 0) // Mostrar errores con menor frecuencia
                    Print($"DataFeeder: Error al obtener datos de TheStrat: {ex.Message}");
            }
            
            // Añadir datos de TheStrat al mensaje
            string enhancedMessage = message + ";" + 
                                   stratBarType + ";" +
                                   stratProbabilities[0].ToString(CultureInfo.InvariantCulture) + ";" + 
                                   stratProbabilities[1].ToString(CultureInfo.InvariantCulture);
            
            byte[] data = Encoding.ASCII.GetBytes(enhancedMessage);
            
            lock (clients)
            {
                // Crear una lista de clientes que se desconectaron
                List<TcpClient> disconnectedClients = new List<TcpClient>();
                
                foreach (var client in clients)
                {
                    try
                    {
                        if (client.Connected)
                        {
                            NetworkStream stream = client.GetStream();
                            if (stream.CanWrite)
                            {
                                stream.Write(data, 0, data.Length);
                            }
                            else
                            {
                                disconnectedClients.Add(client);
                            }
                        }
                        else
                        {
                            disconnectedClients.Add(client);
                        }
                    }
                    catch
                    {
                        disconnectedClients.Add(client);
                    }
                }
                
                // Eliminar clientes desconectados
                foreach (var client in disconnectedClients)
                {
                    clients.Remove(client);
                    client.Close();
                }
                
                if (disconnectedClients.Count > 0)
                {
                    Print($"DataFeeder: {disconnectedClients.Count} cliente(s) desconectado(s). Restantes: {clients.Count}");
                }
            }
        }
        
        // Método para cerrar el servidor de resultados
        private void CloseResultsServer()
        {
            isResultsServerRunning = false;
            
            lock (resultsClients)
            {
                foreach (var client in resultsClients)
                {
                    try
                    {
                        client.Close();
                    }
                    catch { }
                }
                resultsClients.Clear();
            }
            
            if (resultsServer != null)
            {
                try
                {
                    resultsServer.Stop();
                }
                catch { }
                resultsServer = null;
            }
            
            Print("DataFeeder: Servidor TCP de resultados cerrado");
        }
        
        // Método para cerrar el servidor y liberar recursos
        private void CloseServer()
        {
            isServerRunning = false;
            
            lock (clients)
            {
                foreach (var client in clients)
                {
                    try
                    {
                        client.Close();
                    }
                    catch { }
                }
                clients.Clear();
            }
            
            if (tcpServer != null)
            {
                try
                {
                    tcpServer.Stop();
                }
                catch { }
                tcpServer = null;
            }
            
            // Cerrar el servidor de resultados
            CloseResultsServer();
            
            Print("DataFeeder: Servidor TCP cerrado");
        }
		
		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description									= @"Alimenta datos de mercado a sistemas externos vía TCP/IP y ejecuta operaciones basadas en consensus trading";
				Name										= "DataFeeder";
				Calculate									= Calculate.OnEachTick;
				EntriesPerDirection							= 1;
				EntryHandling								= EntryHandling.AllEntries;
				IsExitOnSessionCloseStrategy				= true;
				ExitOnSessionCloseSeconds					= 30;
				IsFillLimitOnTouch							= false;
				MaximumBarsLookBack							= MaximumBarsLookBack.TwoHundredFiftySix;
				OrderFillResolution							= OrderFillResolution.Standard;
				Slippage									= 0;
				StartBehavior								= StartBehavior.WaitUntilFlat;
				TimeInForce									= TimeInForce.Gtc;
				TraceOrders									= false;
				RealtimeErrorHandling						= RealtimeErrorHandling.StopCancelClose;
				StopTargetHandling							= StopTargetHandling.PerEntryExecution;
				BarsRequiredToTrade							= 20;
				IsInstantiatedOnEachOptimizationIteration	= true;
                
                // Valores por defecto para parámetros
                DefaultQuantity                             = 1;
                MaxDailyTrades                              = 10;
                RequireConsensus                            = true;
                MinConfidenceThreshold                      = 0.6; // Reducido de 0.7 a 0.6 para permitir más operaciones
                
                // Inicializar valores
                dailyTradeCount = 0;
                lastTradeDate = DateTime.MinValue;
                limitMessagePrinted = false;
                limitMessageDate = DateTime.MinValue;
			}
			else if (State == State.Configure)
			{
                // Añadir indicadores necesarios
                AddChartIndicator(HFT_TheStrat_ML());
			}
			else if (State == State.DataLoaded)
	        {
				StartServer();
	            Print("DataFeeder: Servidor iniciado y esperando conexiones");
	        }
            else if (State == State.Active)
            {
                // Asegurarse de que el contador está en cero al iniciar trading
                dailyTradeCount = 0;
                lastTradeDate = DateTime.Now;
                limitMessagePrinted = false;
                Print($"DataFeeder: Estado activo - Contador de operaciones: {dailyTradeCount}");
            }
			else if (State == State.Terminated)
	        {
				CloseServer();
	            Print("DataFeeder: Servidor cerrado");
	        }
		}

		protected override void OnBarUpdate()
		{
			if(State != State.Realtime)
			{
				return;
			}
			
			double feature1 = Close[0];
	        double feature2 = Volume[0];
			string message = feature1.ToString(CultureInfo.InvariantCulture)+";"+feature2.ToString(CultureInfo.InvariantCulture);
			
			// Enviamos los datos a todos los clientes conectados
	        SendDataToClients(message);
            
            // Monitorear estrategias ATM activas
            MonitorAtmStrategies();
            
            // Evaluar señales para posibles operaciones
            if (stratIndicator != null)
            {
                EvaluateAndExecuteTrade();
            }
		}
		
		protected override void OnMarketData(MarketDataEventArgs marketDataUpdate)
		{
		}
	}
}
