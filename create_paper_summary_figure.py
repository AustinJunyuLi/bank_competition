#!/usr/bin/env python3
"""
Create a comprehensive summary figure for the paper that combines key results
in a single publication-ready layout.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Output directory
OUT = os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(OUT, exist_ok=True)

# Professional color scheme
COLORS = {
    'line': '#2E86C1',      # Professional blue for main lines
    'accent': '#E74C3C',    # Red for secondary lines  
    'mixed': '#3498DB',     # Light blue for mixed regime
    'boundary': '#F39C12',  # Orange for boundary regime
    'slack': '#27AE60',     # Green for slack regime
    'grid': '#ECF0F1',      # Very light gray for grids
    'text': '#2C3E50'       # Dark gray for text
}

def setup_axis(ax, xlabel, ylabel, title=None):
    """Setup axis with consistent styling"""
    ax.set_xlabel(xlabel, fontsize=11, color=COLORS['text'])
    ax.set_ylabel(ylabel, fontsize=11, color=COLORS['text'])
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', color=COLORS['text'])
    
    # Clean styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(colors=COLORS['text'], labelsize=9)

def add_regime_shading(ax, data, x_col, alpha=0.1):
    """Add regime-based background shading"""
    if 'regime' not in data.columns:
        return
    
    regimes = data['regime'].unique()
    for regime in regimes:
        regime_data = data[data['regime'] == regime]
        if len(regime_data) > 0:
            x_min, x_max = regime_data[x_col].min(), regime_data[x_col].max()
            color = COLORS.get(regime, '#CCCCCC')
            ax.axvspan(x_min, x_max, alpha=alpha, color=color, zorder=1)

def create_comprehensive_summary():
    """Create a comprehensive 3x2 summary figure"""
    # Load data
    dw_data = pd.read_csv(os.path.join(OUT, "results_DW_sweep.csv"))
    tau_mixed = pd.read_csv(os.path.join(OUT, "results_tau_mixed.csv"))
    tau_boundary = pd.read_csv(os.path.join(OUT, "results_tau_boundary.csv"))
    tau_slack = pd.read_csv(os.path.join(OUT, "results_tau_slack.csv"))
    
    # Add capital ratios
    dw_data['CR_D'] = 1 - dw_data['x_D']
    dw_data['CR_W'] = 1 - dw_data['x_W']
    
    for data in [tau_mixed, tau_boundary, tau_slack]:
        data['CR_D'] = 1 - data['x_D']
        data['CR_W'] = 1 - data['x_W']
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Bank Competition with Deposit Insurance: Key Results', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Row 1: DW sweep - Total lending and wholesale rate
    ax = axes[0, 0]
    add_regime_shading(ax, dw_data, 'D_W', alpha=0.1)
    ax.plot(dw_data['D_W'], dw_data['X'], color=COLORS['line'], linewidth=2.5, zorder=10)
    setup_axis(ax, r'Deposit Insurance Coverage $D_W$', r'Total Lending $X$',
              'Panel A: Total Lending vs Coverage')
    
    ax = axes[0, 1]
    add_regime_shading(ax, dw_data, 'D_W', alpha=0.1)
    ax.plot(dw_data['D_W'], dw_data['r_w'], color=COLORS['line'], linewidth=2.5, zorder=10)
    setup_axis(ax, r'Deposit Insurance Coverage $D_W$', r'Wholesale Rate $r_w$',
              'Panel B: Wholesale Rate vs Coverage')
    
    # Row 2: DW sweep - Individual bank lending
    ax = axes[1, 0]
    add_regime_shading(ax, dw_data, 'D_W', alpha=0.1)
    ax.plot(dw_data['D_W'], dw_data['x_D'], 
           color=COLORS['line'], linewidth=2.5, label='Deposit Bank', zorder=10)
    ax.plot(dw_data['D_W'], dw_data['x_W'], 
           color=COLORS['accent'], linewidth=2.5, 
           linestyle='--', label='Wholesale Bank', zorder=10)
    setup_axis(ax, r'Deposit Insurance Coverage $D_W$', 'Bank Lending',
              'Panel C: Individual Bank Lending')
    ax.legend(loc='best', frameon=False, fontsize=10)
    
    ax = axes[1, 1]
    add_regime_shading(ax, dw_data, 'D_W', alpha=0.1)
    ax.plot(dw_data['D_W'], dw_data['CR_D'], 
           color=COLORS['line'], linewidth=2.5, label='Deposit Bank', zorder=10)
    ax.plot(dw_data['D_W'], dw_data['CR_W'], 
           color=COLORS['accent'], linewidth=2.5, 
           linestyle='--', label='Wholesale Bank', zorder=10)
    setup_axis(ax, r'Deposit Insurance Coverage $D_W$', 'Capital Ratio',
              'Panel D: Capital Ratios')
    ax.legend(loc='best', frameon=False, fontsize=10)
    
    # Row 3: Tau effects across regimes - Total lending and wholesale rate
    ax = axes[2, 0]
    ax.plot(tau_mixed['tau'], tau_mixed['X'], 
           color=COLORS['mixed'], linewidth=2.5, label='Mixed Regime', zorder=10)
    ax.plot(tau_boundary['tau'], tau_boundary['X'], 
           color=COLORS['boundary'], linewidth=2.5, 
           linestyle='--', label='Boundary Regime', zorder=10)
    ax.plot(tau_slack['tau'], tau_slack['X'], 
           color=COLORS['slack'], linewidth=2.5, 
           linestyle=':', label='Slack Regime', zorder=10)
    setup_axis(ax, r'Premium $\tau$ (bps)', r'Total Lending $X$',
              'Panel E: Premium Effects on Total Lending')
    ax.legend(loc='best', frameon=False, fontsize=10)
    
    ax = axes[2, 1]
    ax.plot(tau_mixed['tau'], tau_mixed['r_w'], 
           color=COLORS['mixed'], linewidth=2.5, label='Mixed Regime', zorder=10)
    ax.plot(tau_boundary['tau'], tau_boundary['r_w'], 
           color=COLORS['boundary'], linewidth=2.5, 
           linestyle='--', label='Boundary Regime', zorder=10)
    ax.plot(tau_slack['tau'], tau_slack['r_w'], 
           color=COLORS['slack'], linewidth=2.5, 
           linestyle=':', label='Slack Regime', zorder=10)
    setup_axis(ax, r'Premium $\tau$ (bps)', r'Wholesale Rate $r_w$',
              'Panel F: Premium Effects on Wholesale Rate')
    ax.legend(loc='best', frameon=False, fontsize=10)
    
    # Add regime legend for DW plots
    regimes = sorted(dw_data['regime'].unique())
    regime_labels = {
        'mixed': 'Mixed Regime',
        'boundary': 'Boundary Regime', 
        'slack': 'Slack Regime'
    }
    
    regime_handles = [plt.Rectangle((0, 0), 1, 1, 
                                   facecolor=COLORS.get(regime, '#CCCCCC'), 
                                   alpha=0.3, label=regime_labels.get(regime, regime.title()))
                     for regime in regimes if len(regimes) > 1]
    
    if regime_handles:
        fig.legend(handles=regime_handles, loc='upper right', bbox_to_anchor=(0.99, 0.95), 
                  frameon=False, fontsize=11, title='Regime Shading')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.25)
    
    plt.savefig(os.path.join(OUT, "paper_summary_comprehensive.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT, "paper_summary_comprehensive.pdf"), bbox_inches='tight')
    plt.close()

def create_key_mechanisms_figure():
    """Create a focused 2x2 figure showing key economic mechanisms"""
    # Load data
    dw_data = pd.read_csv(os.path.join(OUT, "results_DW_sweep.csv"))
    tau_mixed = pd.read_csv(os.path.join(OUT, "results_tau_mixed.csv"))
    
    # Create figure focusing on key mechanisms
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Key Economic Mechanisms in Bank Competition Model', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Panel 1: Competition effect - how deposit access affects lending allocation
    ax = axes[0, 0]
    add_regime_shading(ax, dw_data, 'D_W', alpha=0.1)
    ax.plot(dw_data['D_W'], dw_data['x_D'], 
           color=COLORS['line'], linewidth=3, label='Deposit Bank', zorder=10)
    ax.plot(dw_data['D_W'], dw_data['x_W'], 
           color=COLORS['accent'], linewidth=3, 
           linestyle='--', label='Wholesale Bank', zorder=10)
    setup_axis(ax, r'Deposit Access $D_W$', 'Bank Lending',
              'Competition Effect: Lending Reallocation')
    ax.legend(loc='best', frameon=False, fontsize=11)
    
    # Panel 2: Market discipline - wholesale rate response
    ax = axes[0, 1]
    add_regime_shading(ax, dw_data, 'D_W', alpha=0.1)
    ax.plot(dw_data['D_W'], dw_data['r_w'], 
           color=COLORS['line'], linewidth=3, zorder=10)
    setup_axis(ax, r'Deposit Access $D_W$', r'Wholesale Rate $r_w$',
              'Market Discipline: Wholesale Pricing')
    
    # Panel 3: Insurance effect - premium impact on deposit bank
    ax = axes[1, 0]
    ax.plot(tau_mixed['tau'], tau_mixed['x_D'], 
           color=COLORS['line'], linewidth=3, label='Deposit Bank', zorder=10)
    ax.plot(tau_mixed['tau'], tau_mixed['x_W'], 
           color=COLORS['accent'], linewidth=3, 
           linestyle='--', label='Wholesale Bank', zorder=10)
    setup_axis(ax, r'Insurance Premium $\tau$', 'Bank Lending',
              'Insurance Effect: Premium Impact')
    ax.legend(loc='best', frameon=False, fontsize=11)
    
    # Panel 4: Aggregate efficiency - total lending response
    ax = axes[1, 1]
    ax.plot(dw_data['D_W'], dw_data['X'], 
           color=COLORS['line'], linewidth=3, label='DW Sweep', zorder=10)
    ax.plot(tau_mixed['tau'], tau_mixed['X'], 
           color=COLORS['mixed'], linewidth=3, 
           linestyle='--', label='Premium Effect', zorder=10)
    setup_axis(ax, 'Policy Parameter', r'Total Lending $X$',
              'Aggregate Efficiency: Total Credit')
    ax.legend(loc='best', frameon=False, fontsize=11)
    
    # Add regime legend
    regimes = sorted(dw_data['regime'].unique())
    regime_labels = {
        'mixed': 'Mixed Regime',
        'boundary': 'Boundary Regime', 
        'slack': 'Slack Regime'
    }
    
    regime_handles = [plt.Rectangle((0, 0), 1, 1, 
                                   facecolor=COLORS.get(regime, '#CCCCCC'), 
                                   alpha=0.3, label=regime_labels.get(regime, regime.title()))
                     for regime in regimes if len(regimes) > 1]
    
    if regime_handles:
        fig.legend(handles=regime_handles, loc='upper right', bbox_to_anchor=(0.99, 0.92), 
                  frameon=False, fontsize=11, title='Regime Shading')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.25, wspace=0.25)
    
    plt.savefig(os.path.join(OUT, "key_mechanisms.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT, "key_mechanisms.pdf"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Creating comprehensive summary figures for paper...")
    
    create_comprehensive_summary()
    print("✓ Created comprehensive 6-panel summary figure")
    
    create_key_mechanisms_figure()
    print("✓ Created key mechanisms 4-panel figure")
    
    print(f"\nSummary figures saved to: {OUT}")
    print("\nRecommended usage:")
    print("- Use 'paper_summary_comprehensive' as main results figure")
    print("- Use 'key_mechanisms' to highlight economic intuition")
    print("- Both available in PNG (300 DPI) and PDF formats")
