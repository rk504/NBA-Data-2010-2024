# %% [code] {"execution":{"iopub.status.busy":"2025-01-05T19:48:27.416706Z","iopub.execute_input":"2025-01-05T19:48:27.417102Z","iopub.status.idle":"2025-01-05T19:48:27.952751Z","shell.execute_reply.started":"2025-01-05T19:48:27.417072Z","shell.execute_reply":"2025-01-05T19:48:27.951721Z"}}
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt

file_path = '~/code/data/kaggle/input/basketball/csv/game.csv'
data = pd.read_csv(file_path)

# Convert game_date to datetime if it's not already
data['game_date'] = pd.to_datetime(data['game_date'])

# Select the desired columns from the 'data' DataFrame
new_table = data[['team_abbreviation_home', 'game_date','plus_minus_home',
                    'matchup_home', 
                  'team_abbreviation_away', 'plus_minus_away']]
# %% [code] {"execution":{"iopub.status.busy":"2025-01-05T19:48:27.416706Z","iopub.execute_input":"2025-01-05T19:48:27.417102Z","iopub.status.idle":"2025-01-05T19:48:27.952751Z","shell.execute_reply.started":"2025-01-05T19:48:27.417072Z","shell.execute_reply":"2025-01-05T19:48:27.951721Z"}}
def calculate_season(date):
    year = date.year
    # If the month is between January (1) and June (6), the season is the previous year to the current year
    if date.month <= 6:
        return f'{year - 1}-{str(year)[-2:]}'
    # If the month is between July (7) and December (12), the season is the current year to the next year
    else:
        return f'{year}-{str(year + 1)[-2:]}'

# Now apply the function to calculate the season safely using .loc
new_table = new_table.copy()  # Ensure you're working with a copy
new_table['season'] = new_table['game_date'].apply(calculate_season)

# Filter the data to keep only rows with the season >= '2002-03'
my_data = new_table[new_table['season'] >= '2002-03']

# List of allowed team abbreviations (merged SEA/OKC, NOP/NOH)
allowed_teams = [
    'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL',
    'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
]

# Filter the my_data table to keep only the rows with the allowed teams
my_data_filtered = my_data[my_data['team_abbreviation_home'].isin(allowed_teams) | my_data['team_abbreviation_away'].isin(allowed_teams)]

# Combine SEA into OKC, NOP into NOH, and NJN into BKN
my_data_filtered = my_data_filtered.copy()
my_data_filtered['team_abbreviation_home'] = my_data_filtered['team_abbreviation_home'].replace({'SEA': 'OKC', 'NOH': 'NOP', 'NJN': 'BKN', 'NOK': 'NOP', 'GS': 'GSW'})
my_data_filtered['team_abbreviation_away'] = my_data_filtered['team_abbreviation_away'].replace({'SEA': 'OKC', 'NOH': 'NOP', 'NJN': 'BKN', 'NOK': 'NOP', 'GS': 'GSW'})

# Reassign the filtered data back to the original variable
my_data = my_data_filtered

# %% [code] {"execution":{"iopub.status.busy":"2025-01-05T19:48:27.416706Z","iopub.execute_input":"2025-01-05T19:48:27.417102Z","iopub.status.idle":"2025-01-05T19:48:27.952751Z","shell.execute_reply.started":"2025-01-05T19:48:27.417072Z","shell.execute_reply":"2025-01-05T19:48:27.951721Z"}}

# Filter for clutch games (games where plus_minus_home or plus_minus_away is between -5 and 5)
clutch_games = my_data[(my_data['plus_minus_home'] >= -5) & (my_data['plus_minus_home'] <= 5) |
                       (my_data['plus_minus_away'] >= -5) & (my_data['plus_minus_away'] <= 5)]

# Function to determine if the home team won or lost (1-5 points)
def clutch_result(row):
    if row['plus_minus_home'] <= -1:
        return 'loss_home'
    elif row['plus_minus_home'] >= 1:
        return 'win_home'
    elif row['plus_minus_away'] <= -1:
        return 'loss_away'
    elif row['plus_minus_away'] >= 1:
        return 'win_away'
    return None  # Exclude other games (non-clutch)

# Apply the function to get the result of the clutch games
# Ensure clutch_games is a copy before modification (to keep pandas happy / to avoid SettingWithCopyWarning)
clutch_games = clutch_games.copy()
clutch_games['result'] = clutch_games.apply(clutch_result, axis=1)

# Filter for wins and losses
wins = clutch_games[clutch_games['result'].isin(['win_home', 'win_away'])]
losses = clutch_games[clutch_games['result'].isin(['loss_home', 'loss_away'])]

# Create a matrix for wins by team abbreviation and year
wins_matrix_home = pd.pivot_table(wins, values='game_date', index=wins['season'], columns=wins['team_abbreviation_home'], aggfunc='count', fill_value=0)
wins_matrix_away = pd.pivot_table(wins, values='game_date', index=wins['season'], columns=wins['team_abbreviation_away'], aggfunc='count', fill_value=0)

# Create a matrix for losses by team abbreviation and year
losses_matrix_home = pd.pivot_table(losses, values='game_date', index=losses['season'], columns=losses['team_abbreviation_home'], aggfunc='count', fill_value=0)
losses_matrix_away = pd.pivot_table(losses, values='game_date', index=losses['season'], columns=losses['team_abbreviation_away'], aggfunc='count', fill_value=0)

# Combine the home and away results for wins and losses
wins_matrix = wins_matrix_home.add(wins_matrix_away, fill_value=0)
losses_matrix = losses_matrix_home.add(losses_matrix_away, fill_value=0)
# Compute the clutch wins minus losses
clutch_diff = wins_matrix - losses_matrix

# Filter the clutch_diff DataFrame for only the allowed teams
clutch_diff_filtered = clutch_diff[allowed_teams]

# Add a row for the average of each column, rounded to 1 decimal place
# Ensure clutch_diff_filtered is a copy before modification (to keep pandas happy / to avoid SettingWithCopyWarning)
clutch_diff_filtered = clutch_diff_filtered.copy()
clutch_diff_filtered.loc['Average'] = clutch_diff_filtered.mean(axis=0).round(1)

# Sort the clutch difference matrix from high to low based on the 'Average' row
clutch_diff_sorted_teams = clutch_diff_filtered.sort_values(by='Average', axis=1, ascending=False)


# %%
# Generate formatted seasons with updated format
seasons_formatted = []
for season in clutch_diff_sorted_teams.index:
    try:
        # Split and format season string
        parts = season.split('-')
        formatted_season = f"'{parts[0][-2:]}-'{parts[1]}"
        seasons_formatted.append(formatted_season)
    except IndexError:
        # Handle case where the season format is not as expected
        seasons_formatted.append(season)  # Keep the original value if it cannot be formatted

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(clutch_diff_sorted_teams.T, annot=True, cmap="coolwarm", center=0, cbar_kws={'label': 'Least Clutch                                              Most Clutch'}, xticklabels=seasons_formatted)

# Remove the bottom axis label
plt.xlabel('')
# Rotate season labels for better readability
plt.xticks(rotation=70, ha='right')

# Add titles and labels
plt.title('Heatmap of Clutch Wins - Losses (Games Decided by <=5 Points)', fontsize=16)

# Show the plot
plt.tight_layout()
plt.show()

# %%
