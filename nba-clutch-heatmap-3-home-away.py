import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load and prepare the data
file_path = '~/code/data/kaggle/input/basketball/csv/game.csv'
data = pd.read_csv(file_path)

# Convert game_date to datetime and select relevant columns
data['game_date'] = pd.to_datetime(data['game_date'])
new_table = data[['team_abbreviation_home', 'game_date', 'plus_minus_home', 
                  'matchup_home', 'team_abbreviation_away', 'plus_minus_away']]

# Calculate the season column
def calculate_season(date):
    year = date.year
    if date.month <= 6:
        return f'{year - 1}-{str(year)[-2:]}'
    else:
        return f'{year}-{str(year + 1)[-2:]}'

new_table = new_table.copy()
new_table['season'] = new_table['game_date'].apply(calculate_season)

# Filter data for valid seasons and teams
my_data = new_table[new_table['season'] >= '2002-03']
allowed_teams = [
    'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL',
    'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
]
my_data_filtered = my_data[my_data['team_abbreviation_home'].isin(allowed_teams) | 
                           my_data['team_abbreviation_away'].isin(allowed_teams)]
my_data_filtered = my_data_filtered.copy()
my_data_filtered['team_abbreviation_home'] = my_data_filtered['team_abbreviation_home'].replace({'SEA': 'OKC', 'NOH': 'NOP', 'NJN': 'BKN', 'NOK': 'NOP', 'GS': 'GSW'})
my_data_filtered['team_abbreviation_away'] = my_data_filtered['team_abbreviation_away'].replace({'SEA': 'OKC', 'NOH': 'NOP', 'NJN': 'BKN', 'NOK': 'NOP', 'GS': 'GSW'})

# Filter for clutch games
clutch_games = my_data_filtered[
    ((my_data_filtered['plus_minus_home'] >= -5) & (my_data_filtered['plus_minus_home'] <= 5)) |
    ((my_data_filtered['plus_minus_away'] >= -5) & (my_data_filtered['plus_minus_away'] <= 5))
]

# Add results column for clutch games
def clutch_result(row):
    if row['plus_minus_home'] > 0:
        return 'win_home'
    elif row['plus_minus_home'] < 0:
        return 'loss_home'
    elif row['plus_minus_away'] > 0:
        return 'win_away'
    elif row['plus_minus_away'] < 0:
        return 'loss_away'
    return None

clutch_games['result'] = clutch_games.apply(clutch_result, axis=1)

# Split into wins and losses
wins = clutch_games[clutch_games['result'].str.contains('win')]
losses = clutch_games[clutch_games['result'].str.contains('loss')]

# Create pivot tables
wins_matrix_home = pd.pivot_table(
    wins[wins['result'] == 'win_home'],
    values='game_date',
    index='season',
    columns='team_abbreviation_home',
    aggfunc='count',
    fill_value=0
)

wins_matrix_away = pd.pivot_table(
    wins[wins['result'] == 'win_away'],
    values='game_date',
    index='season',
    columns='team_abbreviation_away',
    aggfunc='count',
    fill_value=0
)

losses_matrix_home = pd.pivot_table(
    losses[losses['result'] == 'loss_home'],
    values='game_date',
    index='season',
    columns='team_abbreviation_home',
    aggfunc='count',
    fill_value=0
)

losses_matrix_away = pd.pivot_table(
    losses[losses['result'] == 'loss_away'],
    values='game_date',
    index='season',
    columns='team_abbreviation_away',
    aggfunc='count',
    fill_value=0
)

# Combine home and away results
wins_matrix_combined = wins_matrix_home.add(wins_matrix_away, fill_value=0)
losses_matrix_combined = losses_matrix_home.add(losses_matrix_away, fill_value=0)

# Filter for allowed teams and sort by average
wins_matrix_combined = wins_matrix_combined[allowed_teams]
losses_matrix_combined = losses_matrix_combined[allowed_teams]

# Add average row
wins_matrix_combined.loc['Average'] = wins_matrix_combined.mean(axis=0).round(1)
losses_matrix_combined.loc['Average'] = losses_matrix_combined.mean(axis=0).round(1)

# Sort columns by average
wins_matrix_combined = wins_matrix_combined.sort_values(by='Average', axis=1, ascending=False)
losses_matrix_combined = losses_matrix_combined.sort_values(by='Average', axis=1, ascending=False)

# Display results
print("Wins Matrix (Combined):\n", wins_matrix_combined)
print("\nLosses Matrix (Combined):\n", losses_matrix_combined)

# Optional: Heatmap visualization
plt.figure(figsize=(10, 6))
sns.heatmap(wins_matrix_combined.T, annot=True, cmap='Blues', cbar_kws={'label': 'Wins'}, fmt='.0f')
plt.title('Heatmap of Wins by Season and Team')
plt.xlabel('Season')
plt.ylabel('Team')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(losses_matrix_combined.T, annot=True, cmap='Reds', cbar_kws={'label': 'Losses'}, fmt='.0f')
plt.title('Heatmap of Losses by Season and Team')
plt.xlabel('Season')
plt.ylabel('Team')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

