import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import numpy as np

# Load options data
df_options = pd.read_csv('options.csv')

# Create sample options chain visualization for testing
def create_sample_options_chart():
    """Create a sample options chain chart for testing Florence-2 analysis"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('BMNR Options Chain Analysis - Call and Put Activity', fontsize=16, fontweight='bold')
    
    # Separate calls and puts
    calls_df = df_options[df_options['option_type'] == 'call'].copy()
    puts_df = df_options[df_options['option_type'] == 'put'].copy()
    
    # Chart 1: Call Volume vs Strike
    if not calls_df.empty:
        calls_sorted = calls_df.sort_values('strike')
        ax1.bar(calls_sorted['strike'], calls_sorted['volume'], color='green', alpha=0.7, width=0.5)
        ax1.set_title('Call Volume by Strike Price', fontweight='bold')
        ax1.set_xlabel('Strike Price ($)')
        ax1.set_ylabel('Volume')
        ax1.grid(True, alpha=0.3)
        
        # Add current stock price line (approximately 45.57)
        current_price = 45.57
        ax1.axvline(x=current_price, color='red', linestyle='--', linewidth=2, label=f'Current Price: ${current_price}')
        ax1.legend()
    
    # Chart 2: Put Volume vs Strike
    if not puts_df.empty:
        puts_sorted = puts_df.sort_values('strike')
        ax2.bar(puts_sorted['strike'], puts_sorted['volume'], color='red', alpha=0.7, width=0.5)
        ax2.set_title('Put Volume by Strike Price', fontweight='bold')
        ax2.set_xlabel('Strike Price ($)')
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)
        
        # Add current stock price line
        ax2.axvline(x=current_price, color='red', linestyle='--', linewidth=2, label=f'Current Price: ${current_price}')
        ax2.legend()
    
    # Chart 3: Implied Volatility Smile
    if not df_options.empty:
        # Plot IV vs Strike for both calls and puts
        calls_iv = calls_df.dropna(subset=['implied_volatility', 'strike'])
        puts_iv = puts_df.dropna(subset=['implied_volatility', 'strike'])
        
        if not calls_iv.empty:
            ax3.scatter(calls_iv['strike'], calls_iv['implied_volatility'], 
                       color='green', alpha=0.6, s=calls_iv['volume']/50, label='Calls')
        
        if not puts_iv.empty:
            ax3.scatter(puts_iv['strike'], puts_iv['implied_volatility'], 
                       color='red', alpha=0.6, s=puts_iv['volume']/50, label='Puts')
        
        ax3.set_title('Implied Volatility by Strike (Bubble size = Volume)', fontweight='bold')
        ax3.set_xlabel('Strike Price ($)')
        ax3.set_ylabel('Implied Volatility (%)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.axvline(x=current_price, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Chart 4: Put-Call Ratio by Strike
    if not calls_df.empty and not puts_df.empty:
        # Group by strike and calculate put-call ratios
        call_volume_by_strike = calls_df.groupby('strike')['volume'].sum()
        put_volume_by_strike = puts_df.groupby('strike')['volume'].sum()
        
        # Find common strikes
        common_strikes = set(call_volume_by_strike.index) & set(put_volume_by_strike.index)
        
        if common_strikes:
            strikes = sorted(list(common_strikes))
            pc_ratios = []
            
            for strike in strikes:
                call_vol = call_volume_by_strike.get(strike, 0)
                put_vol = put_volume_by_strike.get(strike, 0)
                pc_ratio = put_vol / call_vol if call_vol > 0 else np.inf
                pc_ratios.append(pc_ratio)
            
            # Filter out infinite values for plotting
            finite_mask = np.isfinite(pc_ratios)
            strikes_finite = np.array(strikes)[finite_mask]
            pc_ratios_finite = np.array(pc_ratios)[finite_mask]
            
            if len(strikes_finite) > 0:
                ax4.plot(strikes_finite, pc_ratios_finite, 'o-', color='purple', linewidth=2, markersize=6)
                ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='PC Ratio = 1.0')
                ax4.set_title('Put-Call Volume Ratio by Strike', fontweight='bold')
                ax4.set_xlabel('Strike Price ($)')
                ax4.set_ylabel('Put-Call Ratio')
                ax4.grid(True, alpha=0.3)
                ax4.legend()
                ax4.axvline(x=current_price, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    # Save the chart
    chart_filename = 'options_chain_analysis.png'
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Created sample options chart: {chart_filename}")
    return chart_filename

def analyze_options_data_textually():
    """Provide detailed text analysis of options data"""
    
    print("\n" + "="*60)
    print("DETAILED OPTIONS CHAIN ANALYSIS")
    print("="*60)
    
    # Basic statistics
    total_contracts = len(df_options)
    call_contracts = len(df_options[df_options['option_type'] == 'call'])
    put_contracts = len(df_options[df_options['option_type'] == 'put'])
    
    print(f"\n📊 OVERVIEW:")
    print(f"   Total Contracts: {total_contracts}")
    print(f"   Call Options: {call_contracts} ({call_contracts/total_contracts*100:.1f}%)")
    print(f"   Put Options: {put_contracts} ({put_contracts/total_contracts*100:.1f}%)")
    
    # Volume analysis
    total_volume = df_options['volume'].sum()
    call_volume = df_options[df_options['option_type'] == 'call']['volume'].sum()
    put_volume = df_options[df_options['option_type'] == 'put']['volume'].sum()
    put_call_ratio = put_volume / call_volume if call_volume > 0 else 0
    
    print(f"\n📈 VOLUME ANALYSIS:")
    print(f"   Total Volume: {total_volume:,}")
    print(f"   Call Volume: {call_volume:,} ({call_volume/total_volume*100:.1f}%)")
    print(f"   Put Volume: {put_volume:,} ({put_volume/total_volume*100:.1f}%)")
    print(f"   Put-Call Ratio: {put_call_ratio:.2f}")
    
    # Volatility analysis
    avg_iv = df_options['implied_volatility'].mean()
    max_iv = df_options['implied_volatility'].max()
    min_iv = df_options['implied_volatility'].min()
    
    print(f"\n⚡ VOLATILITY ANALYSIS:")
    print(f"   Average IV: {avg_iv:.2f}%")
    print(f"   Highest IV: {max_iv:.2f}%")
    print(f"   Lowest IV: {min_iv:.2f}%")
    
    # Strike analysis
    current_price = 45.57  # From the CSV data context
    strikes = df_options['strike'].unique()
    otm_calls = df_options[(df_options['option_type'] == 'call') & (df_options['strike'] > current_price)]
    itm_calls = df_options[(df_options['option_type'] == 'call') & (df_options['strike'] <= current_price)]
    otm_puts = df_options[(df_options['option_type'] == 'put') & (df_options['strike'] < current_price)]
    itm_puts = df_options[(df_options['option_type'] == 'put') & (df_options['strike'] >= current_price)]
    
    print(f"\n🎯 MONEYNESS ANALYSIS (Current Price: ${current_price}):")
    print(f"   OTM Calls: {len(otm_calls)} contracts, Volume: {otm_calls['volume'].sum():,}")
    print(f"   ITM Calls: {len(itm_calls)} contracts, Volume: {itm_calls['volume'].sum():,}")
    print(f"   OTM Puts: {len(otm_puts)} contracts, Volume: {otm_puts['volume'].sum():,}")
    print(f"   ITM Puts: {len(itm_puts)} contracts, Volume: {itm_puts['volume'].sum():,}")
    
    # Expiration analysis
    expirations = df_options['expiration_date'].value_counts().sort_index()
    print(f"\n📅 EXPIRATION ANALYSIS:")
    for exp_date, count in expirations.head(5).items():
        exp_volume = df_options[df_options['expiration_date'] == exp_date]['volume'].sum()
        print(f"   {exp_date}: {count} contracts, {exp_volume:,} volume")
    
    # Top volume contracts
    print(f"\n🔥 TOP 5 HIGHEST VOLUME CONTRACTS:")
    top_volume = df_options.nlargest(5, 'volume')
    for _, row in top_volume.iterrows():
        print(f"   {row['contract_symbol']} | {row['option_type'].upper()} ${row['strike']:.0f} | "
              f"Vol: {row['volume']:,} | IV: {row['implied_volatility']:.1f}% | "
              f"Bid-Ask: ${row['bid']:.2f}-${row['ask']:.2f}")
    
    # Price level analysis
    print(f"\n💰 KEY PRICE LEVELS:")
    volume_by_strike = df_options.groupby('strike')['volume'].sum().sort_values(ascending=False)
    print("   Highest Volume Strikes:")
    for strike, volume in volume_by_strike.head(5).items():
        print(f"     ${strike:.0f}: {volume:,} volume")
    
    return {
        'total_volume': total_volume,
        'put_call_ratio': put_call_ratio,
        'avg_iv': avg_iv,
        'current_price': current_price,
        'top_strikes': volume_by_strike.head(5).to_dict()
    }

def create_detailed_summary():
    """Create a detailed summary for image analysis"""
    
    analysis = analyze_options_data_textually()
    
    summary = f"""
    BMNR OPTIONS CHAIN SUMMARY FOR ANALYSIS:
    
    🏢 UNDERLYING: BMNR (Current Price: $45.57)
    📊 Total Volume: {analysis['total_volume']:,} contracts
    ⚖️ Put-Call Ratio: {analysis['put_call_ratio']:.2f}
    ⚡ Average Implied Volatility: {analysis['avg_iv']:.1f}%
    
    🎯 KEY OBSERVATIONS:
    • High call volume at $48-50 strikes (above current price)
    • Significant put activity around $44-45 strikes (near current price) 
    • Most active expiration: 2025-08-29 (very near-term)
    • Volatility appears compressed across strikes
    
    📈 MARKET SENTIMENT INDICATORS:
    • Heavy call buying above current price suggests bullish positioning
    • Put activity concentrated near current price indicates hedging
    • High volume in near-term expiry suggests event-driven activity
    """
    
    return summary

# Main execution
if __name__ == "__main__":
    print("🔬 COMPREHENSIVE OPTIONS ANALYSIS")
    print("=" * 50)
    
    # Create sample chart
    chart_file = create_sample_options_chart()
    
    # Analyze data textually
    summary = create_detailed_summary()
    print(summary)
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Chart saved as: {chart_file}")
    print(f"🖼️ You can now use this image with Florence-2 for visual analysis!")
