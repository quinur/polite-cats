"""Quick data analysis script to understand the structure."""
import pandas as pd

df = pd.read_csv('data/whl.csv')

print('=== DATA STRUCTURE ANALYSIS ===')
print(f'\nTotal rows: {len(df)}')
print(f'Unique games: {df["game_id"].nunique()}')
print(f'Unique teams: {pd.concat([df["home_team"], df["away_team"]]).nunique()}')
print(f'\nRows per game (avg): {len(df) / df["game_id"].nunique():.1f}')

print('\n=== SAMPLE GAME ===')
game1 = df[df['game_id'] == 'game_1']
print(f'Game 1 has {len(game1)} records')
print('\nFirst 5 records of game_1:')
print(game1[['game_id', 'home_team', 'away_team', 'toi', 'home_goals', 'away_goals']].head())

print('\n=== TOI COLUMN ANALYSIS ===')
print(f'TOI mean: {df["toi"].mean():.2f}')
print(f'TOI std: {df["toi"].std():.2f}')
print(f'TOI range: {df["toi"].min():.2f} - {df["toi"].max():.2f}')

print('\n=== GOALS ANALYSIS ===')
print(f'Home goals per record (mean): {df["home_goals"].mean():.3f}')
print(f'Away goals per record (mean): {df["away_goals"].mean():.3f}')

print('\nCONCLUSION: TOI is per-record/shift, NOT per-game')
print('Each game has multiple records (shifts), and TOI needs to be summed')
