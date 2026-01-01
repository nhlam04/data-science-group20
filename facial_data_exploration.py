"""
Facial Data Exploration Script

This script explores the Facial_data dataset which contains CSV files for different facial expressions.
Each CSV file likely contains features or information about facial expressions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
FACIAL_DATA_DIR = Path("Facial_data")
OUTPUT_DIR = Path("facial_data_exploration_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Expression categories
EXPRESSION_FILES = {
    'boring': FACIAL_DATA_DIR / 'boring.csv',
    'confused': FACIAL_DATA_DIR / 'confused.csv',
    'happiness': FACIAL_DATA_DIR / 'happiness.csv',
    'neutral': FACIAL_DATA_DIR / 'neutral.csv',
    'surprise': FACIAL_DATA_DIR / 'surprise.csv'
}

print("="*80)
print("FACIAL DATA EXPLORATION")
print("="*80)

# Load and explore each expression dataset
datasets = {}
summaries = []

for expression, filepath in EXPRESSION_FILES.items():
    if not filepath.exists():
        print(f"\n❌ File not found: {filepath}")
        continue
    
    print(f"\n{'='*80}")
    print(f"Loading: {expression.upper()}")
    print(f"{'='*80}")
    
    # Load dataset
    df = pd.read_csv(filepath)
    datasets[expression] = df
    
    # Basic info
    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    
    # Column info
    print(f"\nColumn Names (first 10):")
    print(df.columns.tolist()[:10])
    if len(df.columns) > 10:
        print(f"... and {len(df.columns) - 10} more columns")
    
    # Data types
    print(f"\nData Types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Missing values
    missing = df.isnull().sum().sum()
    print(f"\nMissing Values: {missing} ({missing / df.size * 100:.2f}% of total)")
    
    # Basic statistics
    print(f"\nBasic Statistics (first 5 columns):")
    print(df.iloc[:, :5].describe())
    
    # Store summary
    summaries.append({
        'Expression': expression,
        'Samples': len(df),
        'Features': len(df.columns),
        'Missing': missing,
        'Missing %': f"{missing / df.size * 100:.2f}%"
    })

# Create summary comparison
print(f"\n{'='*80}")
print("DATASET COMPARISON")
print("="*80)

summary_df = pd.DataFrame(summaries)
print("\n", summary_df.to_string(index=False))

# Save summary
summary_df.to_csv(OUTPUT_DIR / 'dataset_summary.csv', index=False)
print(f"\n✓ Summary saved to: {OUTPUT_DIR / 'dataset_summary.csv'}")

# Visualizations
print(f"\n{'='*80}")
print("CREATING VISUALIZATIONS")
print("="*80)

# 1. Sample counts comparison
if datasets:
    fig, ax = plt.subplots(figsize=(10, 6))
    expressions = list(datasets.keys())
    counts = [len(datasets[exp]) for exp in expressions]
    
    bars = ax.bar(expressions, counts, color=sns.color_palette("husl", len(expressions)))
    ax.set_xlabel('Expression', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Sample Count by Expression Type', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sample_counts.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: sample_counts.png")
    plt.close()

# 2. Feature count comparison
if datasets:
    fig, ax = plt.subplots(figsize=(10, 6))
    expressions = list(datasets.keys())
    feature_counts = [len(datasets[exp].columns) for exp in expressions]
    
    bars = ax.bar(expressions, feature_counts, color=sns.color_palette("muted", len(expressions)))
    ax.set_xlabel('Expression', fontsize=12)
    ax.set_ylabel('Number of Features', fontsize=12)
    ax.set_title('Feature Count by Expression Type', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_counts.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_counts.png")
    plt.close()

# 3. Sample distribution analysis (if datasets have similar structure)
if datasets and len(datasets) > 0:
    # Check if all datasets have the same columns
    first_df = list(datasets.values())[0]
    all_same_columns = all(
        list(df.columns) == list(first_df.columns) 
        for df in datasets.values()
    )
    
    if all_same_columns:
        print("\n✓ All datasets have the same column structure")
        
        # Create a combined dataset with expression labels
        combined_data = []
        for expression, df in datasets.items():
            df_copy = df.copy()
            df_copy['expression'] = expression
            combined_data.append(df_copy)
        
        combined_df = pd.concat(combined_data, ignore_index=True)
        combined_df.to_csv(OUTPUT_DIR / 'combined_facial_data.csv', index=False)
        print(f"✓ Saved combined dataset: {OUTPUT_DIR / 'combined_facial_data.csv'}")
        
        # Analyze first few numeric columns
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'expression' in numeric_cols:
            numeric_cols.remove('expression')
        
        if len(numeric_cols) > 0:
            # Plot distribution of first 4 numeric features by expression
            n_features = min(4, len(numeric_cols))
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Feature Distributions by Expression', fontsize=16, fontweight='bold')
            axes = axes.flatten()
            
            for idx in range(n_features):
                feature = numeric_cols[idx]
                for expression in datasets.keys():
                    expr_data = combined_df[combined_df['expression'] == expression][feature]
                    axes[idx].hist(expr_data, alpha=0.5, label=expression, bins=30)
                
                axes[idx].set_xlabel(feature, fontsize=10)
                axes[idx].set_ylabel('Frequency', fontsize=10)
                axes[idx].set_title(f'{feature} Distribution', fontsize=11, fontweight='bold')
                axes[idx].legend()
                axes[idx].grid(alpha=0.3)
            
            # Hide unused subplots
            for idx in range(n_features, 4):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'feature_distributions.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: feature_distributions.png")
            plt.close()
    else:
        print("\n⚠ Datasets have different column structures - skipping combined analysis")

# 4. Correlation analysis for each expression
print(f"\n{'='*80}")
print("CORRELATION ANALYSIS")
print("="*80)

for expression, df in datasets.items():
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        # Calculate correlation matrix (limit to first 20 features for visibility)
        n_features = min(20, len(numeric_df.columns))
        corr_matrix = numeric_df.iloc[:, :n_features].corr()
        
        # Plot correlation heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title(f'Correlation Matrix - {expression.upper()}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'correlation_{expression}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: correlation_{expression}.png")
        plt.close()

# Final summary
print(f"\n{'='*80}")
print("EXPLORATION COMPLETE")
print("="*80)
print(f"\nTotal expressions analyzed: {len(datasets)}")
print(f"Total samples across all expressions: {sum(len(df) for df in datasets.values())}")
print(f"\nAll outputs saved to: {OUTPUT_DIR}/")

# List all generated files
output_files = list(OUTPUT_DIR.glob('*'))
print(f"\nGenerated files ({len(output_files)}):")
for file in sorted(output_files):
    print(f"  - {file.name}")

print("\n" + "="*80)
