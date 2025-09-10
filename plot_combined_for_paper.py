#!/usr/bin/env python3
"""
Generate combined publication-ready plots for the numerical analysis section.
Creates multi-panel figures that match the paper's narrative structure.
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
    ax.set_xlabel(xlabel, fontsize=12, color=COLORS['text'])
    ax.set_ylabel(ylabel, fontsize=12, color=COLORS['text'])
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', color=COLORS['text'])
    
    # Clean styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(colors=COLORS['text'], labelsize=10)

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

def create_dw_sweep_combined():
    """Create the main DW sweep figure (Figure 1 in paper)"""
    # Load data
    dw_data = pd.read_csv(os.path.join(OUT, "results_DW_sweep.csv"))
    
    # Classify regimes based on wholesale rate
    dw_data['regime'] = 'mixed'
    dw_data.loc[abs(dw_data['r_w']) < 1e-6, 'regime'] = 'boundary'
    
    # Find slack region (where outcomes are flat)
    x_var_std = dw_data['x_W'].rolling(window=10, center=True).std()
    slack_threshold = 1e-4
    slack_mask = x_var_std < slack_threshold
    dw_data.loc[slack_mask, 'regime'] = 'slack'
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Deposit Access $D_W$ Sweep: Core Variables', fontsize=16, fontweight='bold', y=0.95)
    
    # Plot configurations
    plots = [
        (axes[0,0], 'x_W', r'Wholesale Bank Lending $x_W$'),
        (axes[0,1], 'x_D', r'Deposit Bank Lending $x_D$'), 
        (axes[1,0], 'X', r'Total Lending $X$'),
        (axes[1,1], 'CR_D', r'Capital Ratios')
    ]
    
    # Add capital ratio data
    dw_data['CR_D'] = 1 - dw_data['x_D']
    dw_data['CR_W'] = 1 - dw_data['x_W']
    
    for ax, var, ylabel in plots:
        add_regime_shading(ax, dw_data, 'D_W', alpha=0.1)
        
        if var == 'CR_D':  # Special handling for capital ratios
            ax.plot(dw_data['D_W'], dw_data['CR_D'], 
                   color=COLORS['line'], linewidth=2.5, label='Deposit Bank', zorder=10)
            ax.plot(dw_data['D_W'], dw_data['CR_W'], 
                   color=COLORS['accent'], linewidth=2.5, 
                   linestyle='--', label='Wholesale Bank', zorder=10)
            ax.legend(loc='best', frameon=False, fontsize=10)
        else:
            ax.plot(dw_data['D_W'], dw_data[var], 
                   color=COLORS['line'], linewidth=2.5, zorder=10)
        
        setup_axis(ax, r'Deposit Insurance Coverage $D_W$', ylabel)
        
        # Add kink point if identifiable
        if 'regime' in dw_data.columns:
            kink_idx = dw_data[dw_data['regime'] == 'slack'].index
            if len(kink_idx) > 0:
                kink_dw = dw_data.loc[kink_idx[0], 'D_W']
                ax.axvline(kink_dw, color='red', linestyle=':', alpha=0.7, linewidth=1.5)
    
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
        fig.legend(handles=regime_handles, loc='upper right', bbox_to_anchor=(0.98, 0.88), 
                  frameon=False, fontsize=10, title='Regimes')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    plt.savefig(os.path.join(OUT, "DW_sweep_combined.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT, "DW_sweep_combined.pdf"), bbox_inches='tight')
    plt.close()

def create_wholesale_rate_plot():
    """Create the wholesale rate plot (Figure 2 in paper)"""
    # Load data
    dw_data = pd.read_csv(os.path.join(OUT, "results_DW_sweep.csv"))
    
    # Classify regimes
    dw_data['regime'] = 'mixed'
    dw_data.loc[abs(dw_data['r_w']) < 1e-6, 'regime'] = 'boundary'
    
    x_var_std = dw_data['x_W'].rolling(window=10, center=True).std()
    slack_threshold = 1e-4
    slack_mask = x_var_std < slack_threshold
    dw_data.loc[slack_mask, 'regime'] = 'slack'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    add_regime_shading(ax, dw_data, 'D_W', alpha=0.1)
    
    ax.plot(dw_data['D_W'], dw_data['r_w'], 
           color=COLORS['line'], linewidth=2.5, zorder=10)
    
    setup_axis(ax, r'Deposit Insurance Coverage $D_W$', 
              r'Wholesale Rate $r_w$', 
              'Wholesale Rate vs Deposit Access')
    
    # Add regime legend
    regimes = sorted(dw_data['regime'].unique())
    regime_labels = {
        'mixed': 'Mixed Regime ($r_w > 0$)',
        'boundary': 'Boundary Regime ($r_w = 0$)', 
        'slack': 'Slack Regime ($r_w = 0$)'
    }
    
    regime_handles = [plt.Rectangle((0, 0), 1, 1, 
                                   facecolor=COLORS.get(regime, '#CCCCCC'), 
                                   alpha=0.3, label=regime_labels.get(regime, regime.title()))
                     for regime in regimes if len(regimes) > 1]
    
    ax.legend(handles=regime_handles, bbox_to_anchor=(1.05, 1), 
             loc='upper left', frameon=False, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "DW_wholesale_rate.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT, "DW_wholesale_rate.pdf"), bbox_inches='tight')
    plt.close()

def create_tau_regime_comparison():
    """Create combined tau sweep plots for the three regimes"""
    # Load tau sweep data for all three DW values
    tau_files = [
        ("results_tau_mixed.csv", "Mixed Regime ($D_W=0.12$)", "mixed"),
        ("results_tau_boundary.csv", "Boundary Regime ($D_W=0.235$)", "boundary"), 
        ("results_tau_slack.csv", "Slack Regime ($D_W=0.30$)", "slack")
    ]
    
    # Variables to plot
    variables = [
        ('x_W', r'Wholesale Bank $x_W$'),
        ('x_D', r'Deposit Bank $x_D$'),
        ('X', r'Total Lending $X$'),
        ('r_w', r'Wholesale Rate $r_w$')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Premium $\\tau$ Effects Across Regimes', fontsize=16, fontweight='bold', y=0.95)
    
    axes_flat = axes.flatten()
    
    for var_idx, (var, ylabel) in enumerate(variables):
        ax = axes_flat[var_idx]
        
        for file_name, regime_label, regime_key in tau_files:
            file_path = os.path.join(OUT, file_name)
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                color = COLORS.get(regime_key, COLORS['line'])
                linestyle = '-' if regime_key == 'mixed' else '--' if regime_key == 'boundary' else ':'
                
                ax.plot(data['tau'], data[var], 
                       color=color, linewidth=2.5, 
                       linestyle=linestyle, label=regime_label, zorder=10)
        
        setup_axis(ax, r'Deposit Insurance Premium $\tau$', ylabel)
        ax.legend(loc='best', frameon=False, fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    plt.savefig(os.path.join(OUT, "tau_regime_comparison.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT, "tau_regime_comparison.pdf"), bbox_inches='tight')
    plt.close()

def create_tau_individual_regimes():
    """Create separate detailed plots for each regime (matching paper layout)"""
    tau_configs = [
        ("results_tau_mixed.csv", "Mixed Regime", "mixed", 0.12),
        ("results_tau_boundary.csv", "Boundary Regime", "boundary", 0.235),
        ("results_tau_slack.csv", "Slack Regime", "slack", 0.30)
    ]
    
    variables = [
        ('x_W', r'Wholesale Bank $x_W$'),
        ('x_D', r'Deposit Bank $x_D$'), 
        ('X', r'Total Lending $X$'),
        ('r_w', r'Wholesale Rate $r_w$'),
        ('CR', r'Capital Ratios')
    ]
    
    for file_name, regime_name, regime_key, dw_val in tau_configs:
        file_path = os.path.join(OUT, file_name)
        if not os.path.exists(file_path):
            continue
            
        data = pd.read_csv(file_path)
        data['CR_D'] = 1 - data['x_D']
        data['CR_W'] = 1 - data['x_W']
        
        # Create 2x3 subplot (5 panels + 1 empty)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Premium $\\tau$ Effects: {regime_name} ($D_W={dw_val}$)', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        regime_color = COLORS.get(regime_key, COLORS['line'])
        
        for var_idx, (var, ylabel) in enumerate(variables):
            row, col = divmod(var_idx, 3)
            ax = axes[row, col]
            
            if var == 'CR':  # Capital ratios
                ax.plot(data['tau'], data['CR_D'], 
                       color=COLORS['line'], linewidth=2.5, label='Deposit Bank', zorder=10)
                ax.plot(data['tau'], data['CR_W'], 
                       color=COLORS['accent'], linewidth=2.5, 
                       linestyle='--', label='Wholesale Bank', zorder=10)
                ax.legend(loc='best', frameon=False, fontsize=10)
            else:
                ax.plot(data['tau'], data[var], 
                       color=regime_color, linewidth=2.5, zorder=10)
            
            setup_axis(ax, r'Premium $\tau$ (bps)', ylabel)
        
        # Hide the empty subplot
        axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        safe_regime = regime_key.replace(' ', '_').lower()
        plt.savefig(os.path.join(OUT, f"tau_detailed_{safe_regime}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(OUT, f"tau_detailed_{safe_regime}.pdf"), bbox_inches='tight')
        plt.close()

def create_threshold_plot():
    """Create the A_W threshold plot if data exists"""
    threshold_file = os.path.join(OUT, "results_DW_sweep.csv")
    if not os.path.exists(threshold_file):
        return
        
    dw_data = pd.read_csv(threshold_file)
    if 'A_W_starstar' not in dw_data.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(dw_data['D_W'], dw_data['A_W_starstar'], 
           color=COLORS['line'], linewidth=2.5, zorder=10)
    
    # Add horizontal line at A_min if available
    if 'A_min' in dw_data.columns:
        A_min = dw_data['A_min'].iloc[0]
        ax.axhline(A_min, color=COLORS['accent'], linestyle='--', 
                  linewidth=2, label=f'$A_{{\\min}} = {A_min}$')
        ax.legend(loc='best', frameon=False, fontsize=10)
    
    setup_axis(ax, r'Deposit Insurance Coverage $D_W$', 
              r'Deposit-Loss Threshold $A_W^{**}$',
              'Deposit-Loss Threshold vs Coverage')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "DW_threshold_AWss.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT, "DW_threshold_AWss.pdf"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating combined publication-ready plots...")
    
    # Create all combined plots
    create_dw_sweep_combined()
    print("✓ Created DW sweep combined plot")
    
    create_wholesale_rate_plot() 
    print("✓ Created wholesale rate plot")
    
    create_tau_regime_comparison()
    print("✓ Created tau regime comparison")
    
    create_tau_individual_regimes()
    print("✓ Created detailed tau regime plots")
    
    create_threshold_plot()
    print("✓ Created threshold plot")
    
    print(f"\nAll combined plots saved to: {OUT}")
    print("\nKey improvements for paper inclusion:")
    print("- Multi-panel layouts reduce figure count")
    print("- Consistent styling across all plots") 
    print("- Clear regime identification with shading")
    print("- Professional typography and colors")
    print("- Both PNG (300 DPI) and PDF formats")
