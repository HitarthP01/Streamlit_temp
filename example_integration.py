"""
Example script showing how to use the integrated options analysis from backup.py

This demonstrates how to use the OptionsAnalyzer class and analysis functions
that were integrated from temp.py and temp_chart_creator.py
"""

import pandas as pd
import sys
import os

# Add current directory to path to import our modules
sys.path.append(os.getcwd())

# Import the enhanced modules
import temp
import temp_chart_creator

def example_standalone_analysis():
    """Example of running the analysis functions directly"""
    
    print("🔬 STANDALONE OPTIONS ANALYSIS EXAMPLE")
    print("=" * 50)
    
    # Check if options.csv exists
    if not os.path.exists('options.csv'):
        print("❌ options.csv not found. Please run backup.py first to fetch options data.")
        return
    
    # Load options data
    df_options = pd.read_csv('options.csv')
    print(f"📊 Loaded {len(df_options)} options contracts from CSV")
    
    # Run comprehensive analysis from temp.py
    print("\n🔍 RUNNING COMPREHENSIVE ANALYSIS (from temp.py):")
    print("-" * 50)
    
    # This calls the main analysis function from temp.py
    try:
        analysis_results = temp.comprehensive_options_analysis()
        print(f"✅ Analysis complete! Put-Call Ratio: {analysis_results['pc_ratio']:.2f}")
    except Exception as e:
        print(f"❌ Error running temp.py analysis: {e}")
    
    # Run detailed analysis from temp_chart_creator.py
    print("\n📈 RUNNING CHART CREATOR ANALYSIS (from temp_chart_creator.py):")
    print("-" * 50)
    
    try:
        # Create detailed summary
        summary = temp_chart_creator.create_detailed_summary()
        print("📋 DETAILED SUMMARY:")
        print(summary)
        
        # Create enhanced chart
        chart_file = temp_chart_creator.create_sample_options_chart()
        print(f"📊 Enhanced chart created: {chart_file}")
        
    except Exception as e:
        print(f"❌ Error running temp_chart_creator.py analysis: {e}")
    
    # Image analysis if chart exists
    if os.path.exists('options_chain_analysis.png'):
        print("\n🖼️ RUNNING IMAGE ANALYSIS (from temp.py):")
        print("-" * 50)
        
        try:
            # This calls the image analysis function from temp.py
            result = temp.analyze_options_chart_image_simple('options_chain_analysis.png', 'BMNR')
            print(f"✅ Image analysis completed: {result}")
        except Exception as e:
            print(f"❌ Error running image analysis: {e}")
    
    print("\n✅ ALL ANALYSES COMPLETED!")
    print("💡 You can now use these same functions in backup.py through the web interface.")

def example_options_analyzer_class():
    """Example of using the OptionsAnalyzer class from backup.py"""
    
    print("\n🧪 TESTING OPTIONS ANALYZER CLASS")
    print("=" * 50)
    
    if not os.path.exists('options.csv'):
        print("❌ options.csv not found. Please run backup.py first to fetch options data.")
        return
    
    # Load options data
    df_options = pd.read_csv('options.csv')
    
    # Note: The OptionsAnalyzer class is defined in backup.py
    # Here we'll simulate what it does
    
    # Calculate some key metrics
    total_call_vol = df_options[df_options['option_type'] == 'call']['volume'].sum()
    total_put_vol = df_options[df_options['option_type'] == 'put']['volume'].sum()
    pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0
    
    # Get current price
    current_price = 45.57  # fallback
    if 'current_stock_price' in df_options.columns:
        current_prices = df_options['current_stock_price'].dropna()
        if not current_prices.empty:
            current_price = current_prices.iloc[0]
    
    # Near-the-money analysis
    ntm_strikes = df_options[abs(df_options['strike'] - current_price) <= 2.0]
    ntm_volume = ntm_strikes['volume'].sum()
    ntm_percentage = (ntm_volume / df_options['volume'].sum() * 100) if df_options['volume'].sum() > 0 else 0
    
    # Top volume strikes
    strike_volumes = df_options.groupby('strike')['volume'].sum().sort_values(ascending=False)
    
    print("📊 ENHANCED METRICS:")
    print(f"   • Put-Call Volume Ratio: {pc_ratio:.2f}")
    print(f"   • Total Volume: {df_options['volume'].sum():,}")
    print(f"   • Near-the-Money Volume: {ntm_volume:,} ({ntm_percentage:.1f}%)")
    print(f"   • Current Price: ${current_price:.2f}")
    print(f"   • Top Volume Strike: ${strike_volumes.index[0]:.0f} ({strike_volumes.iloc[0]:,} volume)")
    
    sentiment = 'Bearish' if pc_ratio > 1 else 'Bullish'
    print(f"   • Market Sentiment: {sentiment}")

def main():
    """Run all example analyses"""
    
    print("🚀 OPTIONS ANALYSIS INTEGRATION EXAMPLES")
    print("=" * 60)
    print("This script demonstrates how to use the analysis functions")
    print("that have been integrated into backup.py from temp.py and temp_chart_creator.py")
    print()
    
    # Run standalone analysis
    example_standalone_analysis()
    
    # Test enhanced analyzer functionality
    example_options_analyzer_class()
    
    print("\n" + "=" * 60)
    print("🎯 HOW TO USE IN BACKUP.PY:")
    print("1. Run: streamlit run backup.py")
    print("2. Navigate to Options Chain Analysis section")
    print("3. Enter a stock symbol and fetch options data")
    print("4. Explore the enhanced analysis sections:")
    print("   • Comprehensive Analysis (from temp.py)")
    print("   • Enhanced Chart Analysis (from temp_chart_creator.py)")
    print("   • Options Chart Image Analysis")
    print("=" * 60)

if __name__ == "__main__":
    main()
