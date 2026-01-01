import pandas as pd
import numpy as np

# Check structure
df = pd.read_csv(r'd:\2025.1\data science\btl\Facial_data\boring.csv', nrows=3, header=None)
print('Data format: Each row is a flattened 256x256x3 RGB image')
print('Shape:', df.shape)
print('\nFirst pixel (R,G,B) from 3 samples:')
for i in range(3):
    r, g, b = df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2]
    print(f'  Sample {i+1}: ({r}, {g}, {b})')

# Verify all files
files = {
    'boring.csv': 'Boredom',
    'confused.csv': 'Confusion', 
    'happiness.csv': 'Engagement',  # Map happiness to engagement
    'neutral.csv': 'Neutral',
    'surprise.csv': 'Surprise'
}

print('\n\nDataset Summary:')
print('='*60)
base_path = r'd:\2025.1\data science\btl\Facial_data'
total = 0
for filename, emotion in files.items():
    df_temp = pd.read_csv(f'{base_path}\\{filename}', nrows=0)
    # Count lines efficiently
    with open(f'{base_path}\\{filename}', 'r') as f:
        rows = sum(1 for _ in f)
    print(f'{emotion:12s} ({filename:15s}): {rows:6,} images')
    total += rows

print('='*60)
print(f'{'Total':12s}                    : {total:6,} images')
print(f'\nImage format: 256x256 RGB (196,608 pixel values per image)')
print(f'No labels column - emotion determined by filename')
