"""
Beautiful plotting script for bank competition model results.
Generates publication-ready figures with clean, professional styling.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple

# Set up matplotlib for publication-quality figures
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'serif'],
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,  # No grids by default
    'grid.alpha': 0,     # Ensure no grid visibility
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'legend.frameon': False,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Color scheme for professional appearance
COLORS = {
    'mixed': '#2E86AB',      # Professional blue
    'slack': '#A23B72',      # Professional magenta
    'boundary': '#F18F01',   # Professional orange
    'line': '#2C3E50',       # Dark blue-gray
    'accent': '#E74C3C'      # Professional red
}

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(OUT, exist_ok=True)

def setup_axis(ax, xlabel: str, ylabel: str, title: str = None):
    """Set up axis with clean, professional styling."""
    ax.set_xlabel(xlabel, fontsize=12, fontweight='normal')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='normal')
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    # Remove grid completely
    ax.grid(False)
    ax.set_axisbelow(False)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10, 
                   direction='out', length=4, width=0.8)
    ax.tick_params(axis='both', which='minor', direction='out', 
                   length=2, width=0.6)

def add_regime_shading(ax, data: pd.DataFrame, x_col: str, alpha: float = 0.15):
    """Add subtle regime shading to plots."""
    regimes = data['regime'].values
    x_vals = data[x_col].values
    
    current_regime = regimes[0]
    start_idx = 0
    
    for i in range(1, len(regimes)):
        if regimes[i] != current_regime or i == len(regimes) - 1:
            end_idx = i if regimes[i] != current_regime else i + 1
            
            x_start = x_vals[start_idx]
            x_end = x_vals[end_idx - 1] if end_idx > start_idx else x_vals[start_idx]
            
            color = COLORS.get(current_regime, '#CCCCCC')
            ax.axvspan(x_start, x_end, alpha=alpha, color=color, zorder=0)
            
            current_regime = regimes[i]
            start_idx = i

def plot_dw_sweep_variable(data: pd.DataFrame, y_col: str, ylabel: str, 
                          filename: str, title: str = None):
    """Plot a single variable from DW sweep with regime shading."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Add regime shading
    add_regime_shading(ax, data, 'D_W', alpha=0.1)
    
    # Main line plot
    ax.plot(data['D_W'], data[y_col], 
           color=COLORS['line'], linewidth=2.5, zorder=10, label='Model Solution')
    
    setup_axis(ax, r'Deposit Insurance Coverage $D_W$', ylabel, title)
    
    # Add regime labels with better descriptions
    regimes = sorted(data['regime'].unique())
    regime_labels = {
        'mixed': 'Mixed Regime',
        'boundary': 'Boundary Regime', 
        'slack': 'Slack Regime'
    }
    
    if len(regimes) > 1:
        legend_elements = [plt.Rectangle((0, 0), 1, 1, 
                                       facecolor=COLORS.get(regime, '#CCCCCC'), 
                                       alpha=0.3, label=regime_labels.get(regime, regime.title()))
                         for regime in regimes]
        
        # Add main line to legend
        legend_elements.insert(0, plt.Line2D([0], [0], color=COLORS['line'], 
                                           linewidth=2.5, label='Model Solution'))
        
        ax.legend(handles=legend_elements, loc='upper right', frameon=False, 
                 fontsize=10, ncol=1)
    else:
        ax.legend(loc='upper right', frameon=False, fontsize=10)
    
    # Save both PNG and PDF
    plt.savefig(os.path.join(OUT, f"{filename}.png"))
    plt.savefig(os.path.join(OUT, f"{filename}.pdf"))
    plt.close()

def plot_tau_sweep_variable(data_dict: Dict[str, pd.DataFrame], y_col: str, 
                           ylabel: str, filename: str, title: str = None):
    """Plot tau sweep for multiple DW values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Use consistent colors and linestyles for each regime
    regime_styles = {
        'Mixed': {'color': COLORS['mixed'], 'linestyle': '-', 'linewidth': 2.5},
        'Boundary': {'color': COLORS['boundary'], 'linestyle': '--', 'linewidth': 2.5},
        'Slack': {'color': COLORS['slack'], 'linestyle': '-.', 'linewidth': 2.5}
    }
    
    for label, data in data_dict.items():
        # Extract regime from label
        regime_type = 'Mixed' if 'Mixed' in label else ('Boundary' if 'Boundary' in label else 'Slack')
        style = regime_styles[regime_type]
        
        ax.plot(data['tau'], data[y_col], 
               color=style['color'], 
               linestyle=style['linestyle'],
               linewidth=style['linewidth'], 
               label=label, zorder=10)
    
    setup_axis(ax, r'Deposit Insurance Premium $\tau$', ylabel, title)
    
    # Position legend to avoid overlap
    if 'r_w' in y_col or 'Rate' in ylabel:
        legend_loc = 'upper right'
    else:
        legend_loc = 'best'
    
    ax.legend(loc=legend_loc, frameon=False, fontsize=10)
    
    # Save both PNG and PDF
    plt.savefig(os.path.join(OUT, f"{filename}.png"))
    plt.savefig(os.path.join(OUT, f"{filename}.pdf"))
    plt.close()

def plot_combined_thresholds_dw(data: pd.DataFrame):
    """Plot A_D_star and A_W_star together for DW sweep."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Add regime shading
    add_regime_shading(ax, data, 'D_W', alpha=0.1)
    
    # Plot thresholds with better styling
    ax.plot(data['D_W'], data['A_D_star'], 
           color=COLORS['line'], linewidth=2.5, label=r'Deposit Bank Threshold ($A_D^*$)', zorder=10)
    
    # Only plot A_W_star where it exists (not NaN)
    mask = ~data['A_W_star'].isna()
    if mask.any():
        ax.plot(data['D_W'][mask], data['A_W_star'][mask], 
               color=COLORS['accent'], linewidth=2.5, 
               linestyle='--', label=r'Wholesale Bank Threshold ($A_W^*$)', zorder=10)
    
    # Plot A_W_starstar where it exists
    mask_ss = ~data['A_W_starstar'].isna()
    if mask_ss.any():
        ax.plot(data['D_W'][mask_ss], data['A_W_starstar'][mask_ss], 
               color=COLORS['accent'], linewidth=2.0, 
               linestyle=':', alpha=0.8, label=r'Alternative Threshold ($A_W^{**}$)', zorder=10)
    
    setup_axis(ax, r'Deposit Insurance Coverage $D_W$', 
              'Default Thresholds', 'Default Thresholds vs Coverage')
    
    # Add regime legend elements
    regimes = sorted(data['regime'].unique())
    regime_labels = {
        'mixed': 'Mixed Regime',
        'boundary': 'Boundary Regime', 
        'slack': 'Slack Regime'
    }
    
    # Create combined legend
    threshold_handles, threshold_labels = ax.get_legend_handles_labels()
    regime_handles = [plt.Rectangle((0, 0), 1, 1, 
                                   facecolor=COLORS.get(regime, '#CCCCCC'), 
                                   alpha=0.3, label=regime_labels.get(regime, regime.title()))
                     for regime in regimes if len(regimes) > 1]
    
    all_handles = threshold_handles + regime_handles
    all_labels = threshold_labels + [h.get_label() for h in regime_handles]
    
    ax.legend(handles=all_handles, labels=all_labels, loc='upper left', 
             frameon=False, fontsize=10, ncol=1)
    
    # Save both PNG and PDF
    plt.savefig(os.path.join(OUT, "DW_AWss.png"))
    plt.savefig(os.path.join(OUT, "DW_AWss.pdf"))
    plt.close()

def main():
    """Generate all publication-ready plots."""
    print("Generating publication-ready plots...")
    
    # Load data
    try:
        dw_data = pd.read_csv(os.path.join(OUT, "results_DW_sweep.csv"))
        tau_mixed = pd.read_csv(os.path.join(OUT, "results_tau_mixed.csv"))
        tau_boundary = pd.read_csv(os.path.join(OUT, "results_tau_boundary.csv"))
        tau_slack = pd.read_csv(os.path.join(OUT, "results_tau_slack.csv"))
    except FileNotFoundError as e:
        print(f"Error: Could not find required CSV files. {e}")
        print("Please run run_all_v4.py first to generate the data.")
        return
    
    # Fix regime classification for DW sweep
    # Mixed: r_w > 0 (Bank W uses wholesale funding)
    # Boundary: r_w = 0 but cap binds (transition point where wholesale funding just stops)
    # Slack: r_w = 0 and cap is truly slack
    print("Correcting regime classifications...")
    dw_data['regime_corrected'] = dw_data.apply(lambda row: 
        'mixed' if row['r_w'] > 1e-6 
        else ('boundary' if abs(row['r_w']) < 1e-6 and row['regime'] == 'mixed'
              else 'slack'), axis=1)
    
    # Replace the original regime column
    dw_data['regime'] = dw_data['regime_corrected']
    dw_data.drop('regime_corrected', axis=1, inplace=True)
    
    # DW sweep plots
    print("Creating DW sweep plots...")
    plot_dw_sweep_variable(dw_data, 'x_D', r'Deposit Bank Lending $x_D$', 
                          'DW_xD_regimes', 'Deposit Bank Lending vs Coverage')
    
    plot_dw_sweep_variable(dw_data, 'x_W', r'Wholesale Bank Lending $x_W$', 
                          'DW_xW_regimes', 'Wholesale Bank Lending vs Coverage')
    
    plot_dw_sweep_variable(dw_data, 'X', r'Total Lending $X$', 
                          'DW_X_regimes', 'Total Lending vs Coverage')
    
    plot_dw_sweep_variable(dw_data, 'r_w', r'Wholesale Rate $r_w$', 
                          'DW_rw_regimes', 'Wholesale Rate vs Coverage')
    
    # Create capital ratio plot (assuming CR is calculated as equity/assets)
    dw_data['CR_D'] = 1 - dw_data['x_D']  # Simplified capital ratio for deposit bank
    dw_data['CR_W'] = 1 - dw_data['x_W']  # Simplified capital ratio for wholesale bank
    
    fig, ax = plt.subplots(figsize=(8, 5))
    add_regime_shading(ax, dw_data, 'D_W', alpha=0.1)
    ax.plot(dw_data['D_W'], dw_data['CR_D'], 
           color=COLORS['line'], linewidth=2.5, label='Deposit Bank', zorder=10)
    ax.plot(dw_data['D_W'], dw_data['CR_W'], 
           color=COLORS['accent'], linewidth=2.5, 
           linestyle='--', label='Wholesale Bank', zorder=10)
    setup_axis(ax, r'Deposit Insurance Coverage $D_W$', 
              'Capital Ratio', 'Capital Ratios vs Coverage')
    
    # Create combined legend with bank types and regimes
    bank_handles, bank_labels = ax.get_legend_handles_labels()
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
    
    all_handles = bank_handles + regime_handles
    all_labels = bank_labels + [h.get_label() for h in regime_handles]
    
    ax.legend(handles=all_handles, labels=all_labels, loc='upper right', 
             frameon=False, fontsize=10, ncol=1)
    
    plt.savefig(os.path.join(OUT, "DW_CR_regimes.png"))
    plt.savefig(os.path.join(OUT, "DW_CR_regimes.pdf"))
    plt.close()
    
    # Combined thresholds plot
    plot_combined_thresholds_dw(dw_data)
    
    # Tau sweep plots
    print("Creating tau sweep plots...")
    tau_data = {
        r'$D_W = 0.12$ (Mixed)': tau_mixed,
        r'$D_W = 0.235$ (Boundary)': tau_boundary,
        r'$D_W = 0.30$ (Slack)': tau_slack
    }
    
    plot_tau_sweep_variable(tau_data, 'x_D', r'Deposit Bank Lending $x_D$',
                           'tau_DW=0.12 (mixed)_xD', 'Deposit Bank Lending vs Premium')
    
    plot_tau_sweep_variable(tau_data, 'x_W', r'Wholesale Bank Lending $x_W$',
                           'tau_DW=0.12 (mixed)_xW', 'Wholesale Bank Lending vs Premium')
    
    plot_tau_sweep_variable(tau_data, 'X', r'Total Lending $X$',
                           'tau_DW=0.12 (mixed)_X', 'Total Lending vs Premium')
    
    plot_tau_sweep_variable(tau_data, 'r_w', r'Wholesale Rate $r_w$',
                           'tau_DW=0.12 (mixed)_rw', 'Wholesale Rate vs Premium')
    
    # Individual tau plots for each DW value
    for dw_val, label_suffix in [(0.12, 'mixed'), (0.235, 'boundary'), (0.30, 'slack')]:
        data_key = [k for k in tau_data.keys() if str(dw_val) in k][0]
        data = tau_data[data_key]
        
        # Individual plots for each variable
        for var, ylabel in [('x_D', r'Deposit Bank Lending $x_D$'),
                           ('x_W', r'Wholesale Bank Lending $x_W$'),
                           ('X', r'Total Lending $X$'),
                           ('r_w', r'Wholesale Rate $r_w$')]:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(data['tau'], data[var], 
                   color=COLORS['line'], linewidth=2.5, zorder=10)
            setup_axis(ax, r'Deposit Insurance Premium $\tau$', ylabel)
            plt.savefig(os.path.join(OUT, f"tau_DW={dw_val} ({label_suffix})_{var}.png"))
            plt.savefig(os.path.join(OUT, f"tau_DW={dw_val} ({label_suffix})_{var}.pdf"))
            plt.close()
        
        # Capital ratio plot for each DW value
        data['CR_D'] = 1 - data['x_D']
        data['CR_W'] = 1 - data['x_W']
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(data['tau'], data['CR_D'], 
               color=COLORS['line'], linewidth=2.5, label='Deposit Bank', zorder=10)
        ax.plot(data['tau'], data['CR_W'], 
               color=COLORS['accent'], linewidth=2.5, 
               linestyle='--', label='Wholesale Bank', zorder=10)
        setup_axis(ax, r'Deposit Insurance Premium $\tau$', 'Capital Ratio')
        ax.legend(loc='best', frameon=False)
        plt.savefig(os.path.join(OUT, f"tau_DW={dw_val} ({label_suffix})_CR.png"))
        plt.savefig(os.path.join(OUT, f"tau_DW={dw_val} ({label_suffix})_CR.pdf"))
        plt.close()
    
    print(f"All plots generated successfully in {OUT}")
    print("Features:")
    print("- No grids for clean presentation")
    print("- Professional color scheme")
    print("- Publication-ready fonts and sizing")
    print("- Both PNG and PDF formats")
    print("- Regime shading for DW sweep plots")
    print("- Clean axis styling with minimal visual clutter")

if __name__ == "__main__":
    main()
