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
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.DrawingTools;
#endregion
 
namespace NinjaTrader.NinjaScript.Indicators
{
	public class HFT_TheStrat_ML : Indicator
	{
		NinjaTrader.Gui.Tools.SimpleFont title = 
		new NinjaTrader.Gui.Tools.SimpleFont("Agency Fb", 16) { Size = 20, Bold = true };
		
		NinjaTrader.Gui.Tools.SimpleFont title2 = 
		new NinjaTrader.Gui.Tools.SimpleFont("Agency Fb", 16) { Size = 15, Bold = true };
		
		NinjaTrader.Gui.Tools.SimpleFont TextFont = 
		new NinjaTrader.Gui.Tools.SimpleFont("Agency Fb", 18) { Size = 16, Bold = false };
		
		/// machine vars
		double[][] data;
		int dir;
		
		double prior1;
		double prior2;
		double prior3;
		double prior4;
		double prior5;
		double prior6;
		double prior7;
		
		// Series públicas para acceso desde estrategias
		public Series<int> BARTYPE;
		private Series<double> EMA1;
		private Series<double> RSI1;
        
        // Series para exponer las probabilidades para acceso externo
        public Series<double> ProbShort;
        public Series<double> ProbLong;
		
		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description									= @"Enter the description for your new custom Indicator here.";
				Name										= "HFT_TheStrat_ML";
				Calculate									= Calculate.OnBarClose;
				IsOverlay									= true;
				DisplayInDataBox							= true;
				DrawOnPricePanel							= true;
				DrawHorizontalGridLines						= true;
				DrawVerticalGridLines						= true;
				PaintPriceMarkers							= true;
				ScaleJustification							= NinjaTrader.Gui.Chart.ScaleJustification.Right;
				IsSuspendedWhileInactive					= false;
				data = new double[1000][]; /// machine
				MaximumBarsLookBack							= MaximumBarsLookBack.Infinite;
			}
			else if (State == State.Configure)
			{
				BARTYPE = new Series<int>(this);
				EMA1 = new Series<double>(this);
				RSI1 = new Series<double>(this);
				ProbShort = new Series<double>(this);
				ProbLong = new Series<double>(this);
				
				Draw.TextFixed(this, "barType", "HFT The Strat ML", TextPosition.TopRight, Brushes.DeepSkyBlue, title, Brushes.Transparent, Brushes.Transparent, 8);
			}
		}

		protected override void OnBarUpdate()
		{
			if (CurrentBars[0] < 1001)
    			{
					return;
    			}
				
			if(High[0] > High[1] && Low[0] < Low[1])///Outside/ Engulfing
				{
					BARTYPE[0] = 4;
					Draw.Text(this, "type"+CurrentBar, true, "3", 0, High[0] + TickSize * 4, 0,  Brushes.DeepPink, title2, TextAlignment.Center, Brushes.Transparent, Brushes.Transparent, 0);
				}
				else if(High[0] > High[1] || Low[0] < Low[1])///Directional
				{
					if(Close[0] >= Open[0])///Up
					{
						BARTYPE[0] = 3;
						Draw.Text(this, "type"+CurrentBar, true, "2", 0, High[0] + TickSize * 4, 0,  Brushes.LimeGreen, title2, TextAlignment.Center, Brushes.Transparent, Brushes.Transparent, 0);
					}
					else///Down
					{
						BARTYPE[0] = 2;
						Draw.Text(this, "type"+CurrentBar, true, "2", 0, High[0] + TickSize * 4, 0,  Brushes.Red, title2, TextAlignment.Center, Brushes.Transparent, Brushes.Transparent, 0);
					}
				}
				else/// Inside
				{
					BARTYPE[0] = 1;
					Draw.Text(this, "type"+CurrentBar, true, "1", 0, High[0] + TickSize * 4, 0,  Brushes.Yellow, title2, TextAlignment.Center, Brushes.Transparent, Brushes.Transparent, 0);
				}
				
			double ema1 = EMA(10)[0];
			EMA1[0] = ema1;
			
			double rsi1 = RSI(14,3)[0];
			RSI1[0] = rsi1;
				
		/// Machine Learning Code Starts Here
			/// 
			int NUMCLASSES = 2;
			int NUMPREDICTORS = 7;
			int NUMITEMS = 1000; 
			
			for (int i = 0; i < NUMITEMS; i++)
			{
				prior1 = BARTYPE[i];
				prior2 = EMA1[i];
				prior3 = RSI1[i];
				prior4 = Open[i];
				prior5 = High[i];
				prior6 = Low[i];
				prior7 = Close[i];
				
				if(Close[i]> Close[i+1])
			    {
					dir = 1;/// long
				}
				if(Close[i ]< Close[i+1])
		    	{
					dir = 0;/// Short
				}
				
			if(prior1 != null || prior2 != null || prior3 != null || prior4 != null || prior5 != null || prior6 != null || prior7 != null)
			{
			data[i] = new double[] { prior1, prior2, prior3, prior4, prior5, prior6, prior7, dir };
			}
			else
			{
				data[i] = new double[] { 0, 0, 0, 0, 0, 0, 0, dir };
			}

			}
			
		int N = NUMITEMS;
			
	      int[] classCts = new int[NUMCLASSES];
	      for (int i = 0; i < N; ++i)
	      {
	        int c = (int)data[i][NUMPREDICTORS];
	        ++classCts[c];
	      }
	      double[][] means = new double[NUMCLASSES][];
	      for (int c = 0; c < NUMCLASSES; ++c)
	        means[c] = new double[NUMPREDICTORS];
	      
	      for (int i = 0; i < N; ++i)
	      {
	        int c = (int)data[i][NUMPREDICTORS];
	        for (int j = 0; j < NUMPREDICTORS; ++j)
	          means[c][j] += data[i][j];
	      }

	      for (int c = 0; c < NUMCLASSES; ++c)
	      {
	        for (int j = 0; j < NUMPREDICTORS; ++j)
	          means[c][j] /= classCts[c];
	      }

	      double[][] variances = new double[NUMCLASSES][];
	      for (int c = 0; c < NUMCLASSES; ++c)
	        variances[c] = new double[NUMPREDICTORS];

	      for (int i = 0; i < N; ++i)
	      {
	        int c = (int)data[i][NUMPREDICTORS];
	        for (int j = 0; j < NUMPREDICTORS; ++j)
	        {
	          double x = data[i][j];
	          double u = means[c][j];
	          variances[c][j] += (x - u) * (x - u);
	        }
	      }
	      for (int c = 0; c < NUMCLASSES; ++c)
	      {
	        for (int j = 0; j < NUMPREDICTORS; ++j)
	          variances[c][j] /= classCts[c] - 1;
	      }
	      double[] unk = new double[] { BARTYPE[0], EMA1[0], RSI1[0], Open[0], High[0], Low[0], Close[0]};
	      double[][] condProbs = new double[NUMCLASSES][];
	      for (int c = 0; c < NUMCLASSES; ++c)
	        condProbs[c] = new double[NUMPREDICTORS];

	      for (int c = 0; c < NUMCLASSES; ++c)
	      {
	        for (int j = 0; j < NUMPREDICTORS; ++j)
	        {
	          double u = means[c][j];
	          double v = variances[c][j];
	          double x = unk[j];
	          condProbs[c][j] = ProbDensFunc(u, v, x);
	        }
	      }
	      double[] classProbs = new double[NUMCLASSES];
	      for (int c = 0; c < NUMCLASSES; ++c)
	        classProbs[c] = (classCts[c] * 1.0) / N;

	      double[] evidenceTerms = new double[NUMCLASSES];
	      for (int c = 0; c < NUMCLASSES; ++c)
	      {
	        evidenceTerms[c] = classProbs[c];
	        for (int j = 0; j < NUMPREDICTORS; ++j)
	          evidenceTerms[c] *= condProbs[c][j];
	      }
	      double sumEvidence = 0.0;
	      for (int c = 0; c < NUMCLASSES; ++c)
	        sumEvidence += evidenceTerms[c];

	          double[] predictProbs = new double[NUMCLASSES];
	      for (int c = 0; c < NUMCLASSES; ++c)
	        predictProbs[c] = evidenceTerms[c] / sumEvidence;
	      
	      // Guardar las probabilidades en series para acceso externo
	      ProbShort[0] = predictProbs[0];
	      ProbLong[0] = predictProbs[1];
		  
		  if(predictProbs[1] > .70)
		  {
			Draw.TextFixed(this, "BAlgo","Direction Probability \n"+"Short: "+(predictProbs[0]*100).ToString("F2")+"%" 
			  +"\n"+"Long: "+ (predictProbs[1]*100).ToString("F2")+"%", TextPosition.Center, Brushes.LimeGreen, title, Brushes.Black,Brushes.Black,100);
		  }
		  else if(predictProbs[0] > .70)
		  {
		  	Draw.TextFixed(this, "BAlgo","Direction Probability \n"+"Short: "+(predictProbs[0]*100).ToString("F2")+"%" 
			  +"\n"+"Long: "+ (predictProbs[1]*100).ToString("F2")+"%", TextPosition.Center, Brushes.Red, title, Brushes.Black,Brushes.Black,100);
		  }
		  else
		  {
			  Draw.TextFixed(this, "BAlgo","Direction Probability \n"+"Short: "+(predictProbs[0]*100).ToString("F2")+"%" 
			  +"\n"+"Long: "+ (predictProbs[1]*100).ToString("F2")+"%", TextPosition.Center, Brushes.Gold, title, Brushes.Black,Brushes.Black,100);
		  }
			
		}
		
		static double ProbDensFunc(double u, double v, double x)
	    {
	      double left = 1.0 / Math.Sqrt(2 * Math.PI * v);
	      double right = Math.Exp( -(x - u) * (x - u) / (2 * v) );
	      return left * right;
	    }
          
        static double ProbDensFuncStdDev(double u, double v, double x)
          {	      
            double left = ( x - u) > 0 ? (x - u) : (x - u) *-1;
            double right = left / v;
              return right;
          }
	}
}

#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{
	public partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
	{
		private HFT_TheStrat_ML[] cacheHFT_TheStrat_ML;
		public HFT_TheStrat_ML HFT_TheStrat_ML()
		{
			return HFT_TheStrat_ML(Input);
		}

		public HFT_TheStrat_ML HFT_TheStrat_ML(ISeries<double> input)
		{
			if (cacheHFT_TheStrat_ML != null)
				for (int idx = 0; idx < cacheHFT_TheStrat_ML.Length; idx++)
					if (cacheHFT_TheStrat_ML[idx] != null &&  cacheHFT_TheStrat_ML[idx].EqualsInput(input))
						return cacheHFT_TheStrat_ML[idx];
			return CacheIndicator<HFT_TheStrat_ML>(new HFT_TheStrat_ML(), input, ref cacheHFT_TheStrat_ML);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.HFT_TheStrat_ML HFT_TheStrat_ML()
		{
			return indicator.HFT_TheStrat_ML(Input);
		}

		public Indicators.HFT_TheStrat_ML HFT_TheStrat_ML(ISeries<double> input )
		{
			return indicator.HFT_TheStrat_ML(input);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.HFT_TheStrat_ML HFT_TheStrat_ML()
		{
			return indicator.HFT_TheStrat_ML(Input);
		}

		public Indicators.HFT_TheStrat_ML HFT_TheStrat_ML(ISeries<double> input )
		{
			return indicator.HFT_TheStrat_ML(input);
		}
	}
}

#endregion
