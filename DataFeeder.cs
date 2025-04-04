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
        
        // Referencia al indicador TheStrat
        private HFT_TheStrat_ML stratIndicator;
        private double[] stratProbabilities = new double[2] { 0.5, 0.5 };
        private int stratBarType = 0;
        
        // Sistema de votación
        private class VotingSystem
        {
            public double StratMLWeight { get; set; } = 0.6;
            public double DeepQWeight { get; set; } = 0.4;
            public double MinConfidenceThreshold { get; set; } = 0.7;
            public bool RequireConsensus { get; set; } = true;
            public TimeSpan SignalValidityWindow { get; set; } = TimeSpan.FromMinutes(5);
            
            // Señales recibidas desde DeepQ
            public class Signal
            {
                public int Action { get; set; } // 0=Hold, 1=Buy, 2=Sell
                public double Confidence { get; set; }
                public DateTime Timestamp { get; set; }
            }
            
            private List<Signal> signalBuffer = new List<Signal>();
            
            public void AddSignal(int action, double confidence)
            {
                Signal newSignal = new Signal
                {
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
        [Display(Name = "Stop Loss (%)", Description = "Porcentaje para Stop Loss", Order = 2, GroupName = "Gestión de Riesgo")]
        public double StopLossPercent { get; set; }
        
        [NinjaScriptProperty]
        [Display(Name = "Take Profit (%)", Description = "Porcentaje para Take Profit", Order = 3, GroupName = "Gestión de Riesgo")]
        public double TakeProfitPercent { get; set; }
        
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
        
        // Método para procesar mensajes recibidos del servidor de señales
        private void ProcessSignalMessage(string message)
        {
            try
            {
                // Limpiar el mensaje de caracteres no deseados
                message = message.Replace("\n", "").Replace("\r", "").Trim();
                
                // Formato esperado: Action;Confidence;Timestamp
                string[] parts = message.Split(';');
                
                if (parts.Length >= 2)
                {
                    // Limpiar y validar los valores
                    string actionStr = parts[0].Trim();
                    string confidenceStr = parts[1].Trim();
                    
                    // Mostrar mensaje de depuración
                    Print($"DataFeeder: Mensaje recibido: '{message}', Partes: {parts.Length}, Action: '{actionStr}', Confidence: '{confidenceStr}'");
                    
                    // Intentar convertir con manejo de errores
                    if (float.TryParse(actionStr, NumberStyles.Any, CultureInfo.InvariantCulture, out float actionFloat) &&
                        double.TryParse(confidenceStr, NumberStyles.Any, CultureInfo.InvariantCulture, out double confidence))
                    {
                        int action = (int)Math.Round(actionFloat);
                        
                        // Validar rango de acción
                        if (action >= 0 && action <= 2)
                        {
                            Print($"DataFeeder: Señal recibida de DeepQ - Acción: {action}, Confianza: {confidence:P2}");
                            
                            // Añadir al sistema de votación
                            votingSystem.AddSignal(action, confidence);
                            
                            // Solo evaluar operaciones si la acción no es Hold (0)
                            // Esto evita evaluar innecesariamente cuando no hay señal de trading
                            if (action > 0)
                            {
                                // Evaluar si debemos ejecutar una operación
                                EvaluateAndExecuteTrade();
                            }
                        }
                        else
                        {
                            Print($"DataFeeder: Acción fuera de rango: {action}");
                        }
                    }
                    else
                    {
                        Print($"DataFeeder: No se pudo convertir los valores: Action='{actionStr}', Confidence='{confidenceStr}'");
                    }
                }
                else
                {
                    Print($"DataFeeder: Formato de mensaje incorrecto, partes: {parts.Length}");
                }
            }
            catch (Exception ex)
            {
                Print($"DataFeeder: Error al procesar mensaje de señales: {ex.Message}");
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
        
        // Variable para almacenar la última orden OCO ID
        private string lastOcoId = string.Empty;

        // Método para ejecutar operaciones con IDs OCO únicos
        private void ExecuteTrade(int action, double confidence)
        {
            // Calcular niveles de stop loss y take profit en ticks en lugar de porcentaje
            double currentPrice = Close[0];
            
            // Usar ticks para stop loss y take profit (más preciso y menos riesgo)
            int stopLossTicks = 50;  // Valor predeterminado de 50 ticks
            int takeProfitTicks = 100; // Valor predeterminado de 100 ticks
            
            // Convertir StopLossPercent a ticks (si es muy pequeño, usar mínimo 3 ticks)
            if (StopLossPercent > 0)
            {
                double stopLossAmount = currentPrice * (StopLossPercent / 100);
                stopLossTicks = Math.Max(3, (int)(stopLossAmount / TickSize));
            }
            
            // Convertir TakeProfitPercent a ticks (si es muy pequeño, usar mínimo 5 ticks)
            if (TakeProfitPercent > 0)
            {
                double takeProfitAmount = currentPrice * (TakeProfitPercent / 100);
                takeProfitTicks = Math.Max(5, (int)(takeProfitAmount / TickSize));
            }
            
            // Limitar el máximo de ticks para evitar riesgos excesivos
            stopLossTicks = Math.Min(stopLossTicks, 20);
            takeProfitTicks = Math.Min(takeProfitTicks, 40);
            
            // Calcular precios exactos
            double stopLoss = action == 1 ? 
                currentPrice - (stopLossTicks * TickSize) : 
                currentPrice + (stopLossTicks * TickSize);
                
            double takeProfit = action == 1 ? 
                currentPrice + (takeProfitTicks * TickSize) : 
                currentPrice - (takeProfitTicks * TickSize);
            
            // Generar un ID único para esta operación usando un timestamp y un valor aleatorio
            string uniqueOcoId = "OCO_" + DateTime.Now.Ticks.ToString() + "_" + new Random().Next(10000, 99999).ToString();
            lastOcoId = uniqueOcoId; // Almacenar para referencia

            // Cancelar órdenes OCO anteriores si existen
            CancelOCOOrders();
            
            // Ejecutar orden correspondiente con ID OCO único
            if (action == 1) // Buy
            {
                if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort();

                // Generar orden de entrada, stop loss y take profit con OCO ID único
                EnterLongStopMarket(0, false, DefaultQuantity, currentPrice, uniqueOcoId + "_Entry");
                ExitLongStopMarket(0, true, DefaultQuantity, stopLoss, uniqueOcoId + "_SL", uniqueOcoId);
                ExitLongLimit(0, true, DefaultQuantity, takeProfit, uniqueOcoId + "_TP", uniqueOcoId);
                
                Print($"DataFeeder: ORDEN LONG ejecutada - Precio: {currentPrice}, SL: {stopLoss} ({stopLossTicks} ticks), TP: {takeProfit} ({takeProfitTicks} ticks), Confianza: {confidence:P2}, OCO ID: {uniqueOcoId}");
            }
            else if (action == 2) // Sell
            {
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong();
                    
                // Generar orden de entrada, stop loss y take profit con OCO ID único
                EnterShortStopMarket(0, false, DefaultQuantity, currentPrice, uniqueOcoId + "_Entry");
                ExitShortStopMarket(0, true, DefaultQuantity, stopLoss, uniqueOcoId + "_SL", uniqueOcoId);
                ExitShortLimit(0, true, DefaultQuantity, takeProfit, uniqueOcoId + "_TP", uniqueOcoId);
                
                Print($"DataFeeder: ORDEN SHORT ejecutada - Precio: {currentPrice}, SL: {stopLoss} ({stopLossTicks} ticks), TP: {takeProfit} ({takeProfitTicks} ticks), Confianza: {confidence:P2}, OCO ID: {uniqueOcoId}");
            }
            
            // Actualizar contadores
            dailyTradeCount++;
            lastTradeDate = DateTime.Now;
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
                StopLossPercent                             = 0.2;  // Reducido para menor riesgo
                TakeProfitPercent                           = 0.5;  // Reducido para menor riesgo
                MaxDailyTrades                              = 10;
                RequireConsensus                            = true;
                MinConfidenceThreshold                      = 0.7;
                
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
