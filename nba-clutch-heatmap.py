# %% [code] {"execution":{"iopub.status.busy":"2025-01-03T19:48:27.416706Z","iopub.execute_input":"2025-01-03T19:48:27.417102Z","iopub.status.idle":"2025-01-03T19:48:27.932731Z","shell.execute_reply.started":"2025-01-03T19:48:27.417072Z","shell.execute_reply":"2025-01-03T19:48:27.931721Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# Use the full path to the file found in the input directory
file_path = '~/code/data/kaggle/input/basketball/csv/game.csv'
data = pd.read_csv(file_path)

# Convert game_date to datetime if it's not already
data['game_date'] = pd.to_datetime(data['game_date'])

# Select the desired columns from the 'data' DataFrame
new_table = data[['team_abbreviation_home', 'game_date','plus_minus_home',
                    'matchup_home', 
                  'team_abbreviation_away', 'plus_minus_away']]

# Display the new table
print(new_table.head(10))
        
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2025-01-03T19:48:36.669199Z","iopub.execute_input":"2025-01-03T19:48:36.669517Z","iopub.status.idle":"2025-01-03T19:48:36.796416Z","shell.execute_reply.started":"2025-01-03T19:48:36.669493Z","shell.execute_reply":"2025-01-03T19:48:36.795306Z"}}
# Define a function to calculate the season
def calculate_season(date):
    year = date.year
    # If the month is between January (1) and June (6), the season is the previous year to the current year
    if date.month <= 6:
        return f'{year - 1}-{str(year)[-2:]}'
    # If the month is between July (7) and December (12), the season is the current year to the next year
    else:
        return f'{year}-{str(year + 1)[-2:]}'

# Ensure you're working with a copy of the DataFrame when assigning a new column
new_table = new_table.copy()

# Now apply the function to calculate the season safely using .loc
new_table['season'] = new_table['game_date'].apply(calculate_season)

# Filter the data to keep only rows with the season >= '2002-03'
my_data = new_table[new_table['season'] >= '2002-03']

# Display the filtered table
print(my_data.head())


# %% [code] {"execution":{"iopub.status.busy":"2025-01-03T19:48:44.557852Z","iopub.execute_input":"2025-01-03T19:48:44.558188Z","iopub.status.idle":"2025-01-03T19:48:44.714018Z","shell.execute_reply.started":"2025-01-03T19:48:44.558160Z","shell.execute_reply":"2025-01-03T19:48:44.712922Z"}}
import pandas as pd

# Filter for clutch games (games where plus_minus_home or plus_minus_away is between -3 and 3)
clutch_games = my_data[(my_data['plus_minus_home'] >= -3) & (my_data['plus_minus_home'] <= 3) |
                       (my_data['plus_minus_away'] >= -3) & (my_data['plus_minus_away'] <= 3)]

# Function to determine if the home team won or lost (1-3 points)
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

# Add average for each column at the bottom
wins_matrix.loc['Average'] = wins_matrix.mean()
losses_matrix.loc['Average'] = losses_matrix.mean()
clutch_diff.loc['Average'] = clutch_diff.mean()

# Display all three tables
print("Clutch Wins Matrix:")
print(wins_matrix)

print("\nClutch Losses Matrix:")
print(losses_matrix)

print("\nClutch Wins - Losses Matrix:")
print(clutch_diff)


# %% [code] {"execution":{"iopub.status.busy":"2025-01-03T18:00:48.677396Z","iopub.execute_input":"2025-01-03T18:00:48.677782Z","iopub.status.idle":"2025-01-03T18:00:48.784829Z","shell.execute_reply.started":"2025-01-03T18:00:48.677742Z","shell.execute_reply":"2025-01-03T18:00:48.783580Z"}}
import pandas as pd

# Filter for clutch games (games where plus_minus_home or plus_minus_away is between -3 and 3)
clutch_games = my_data[(my_data['plus_minus_home'] >= -3) & (my_data['plus_minus_home'] <= 3) |
                       (my_data['plus_minus_away'] >= -3) & (my_data['plus_minus_away'] <= 3)]

# Function to determine if the home team won or lost (1-3 points)
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

# Filter for just "NOH" and "NOP" results
wins_matrix_noh_nop = wins_matrix[['NOH', 'NOP']]
losses_matrix_noh_nop = losses_matrix[['NOH', 'NOP']]
clutch_diff_noh_nop = clutch_diff[['NOH', 'NOP']]

# Add average for each column at the bottom
wins_matrix_noh_nop.loc['Average'] = wins_matrix_noh_nop.mean()
losses_matrix_noh_nop.loc['Average'] = losses_matrix_noh_nop.mean()
clutch_diff_noh_nop.loc['Average'] = clutch_diff_noh_nop.mean()

# Display the results for just "NOH" and "NOP"
print("Clutch Wins Matrix for NOH and NOP:")
print(wins_matrix_noh_nop)

print("\nClutch Losses Matrix for NOH and NOP:")
print(losses_matrix_noh_nop)

print("\nClutch Wins - Losses Matrix for NOH and NOP:")
print(clutch_diff_noh_nop)


# %% [code] {"execution":{"iopub.status.busy":"2025-01-03T18:00:48.785766Z","iopub.execute_input":"2025-01-03T18:00:48.786054Z","iopub.status.idle":"2025-01-03T18:00:48.953484Z","shell.execute_reply.started":"2025-01-03T18:00:48.786027Z","shell.execute_reply":"2025-01-03T18:00:48.952236Z"}}
import pandas as pd

# List of allowed team abbreviations (merged SEA/OKC, NOP/NOH)
allowed_teams = [
    'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL',
    'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
]

# Filter the my_data table to keep only the rows with the allowed teams
# Combine "SEA" and "OKC" into "OKC" and "NOP" and "NOH" into "NOP"
my_data_filtered = my_data[my_data['team_abbreviation_home'].isin(allowed_teams) | my_data['team_abbreviation_away'].isin(allowed_teams)]

# Combine SEA and OKC into OKC, and NOP and NOH into NOP
my_data_filtered['team_abbreviation_home'] = my_data_filtered['team_abbreviation_home'].replace({'SEA': 'OKC', 'NOH': 'NOP'})
my_data_filtered['team_abbreviation_away'] = my_data_filtered['team_abbreviation_away'].replace({'SEA': 'OKC', 'NOH': 'NOP'})

# Now we can proceed with your original pivot table code with the filtered data

# Filter for clutch games (games where plus_minus_home or plus_minus_away is between -3 and 3)
clutch_games = my_data_filtered[(my_data_filtered['plus_minus_home'] >= -3) & (my_data_filtered['plus_minus_home'] <= 3) |
                                (my_data_filtered['plus_minus_away'] >= -3) & (my_data_filtered['plus_minus_away'] <= 3)]

# Function to determine if the home team won or lost (1-3 points)
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

# Filter for just "NOH"/"NOP" and "SEA"/"OKC" results
wins_matrix_filtered = wins_matrix[allowed_teams]
losses_matrix_filtered = losses_matrix[allowed_teams]
clutch_diff_filtered = clutch_diff[allowed_teams]

# Add average for each column at the bottom
wins_matrix_filtered.loc['Average'] = wins_matrix_filtered.mean()
losses_matrix_filtered.loc['Average'] = losses_matrix_filtered.mean()
clutch_diff_filtered.loc['Average'] = clutch_diff_filtered.mean()

# Display the filtered results
print("Clutch Wins Matrix (Filtered):")
print(wins_matrix_filtered)

print("\nClutch Losses Matrix (Filtered):")
print(losses_matrix_filtered)

print("\nClutch Wins - Losses Matrix (Filtered):")
print(clutch_diff_filtered)


# %% [code] {"execution":{"iopub.status.busy":"2025-01-03T18:00:48.954578Z","iopub.execute_input":"2025-01-03T18:00:48.954964Z","iopub.status.idle":"2025-01-03T18:00:49.125128Z","shell.execute_reply.started":"2025-01-03T18:00:48.954931Z","shell.execute_reply":"2025-01-03T18:00:49.123775Z"}}
import pandas as pd

# List of allowed team abbreviations (merged SEA/OKC, NOP/NOH)
allowed_teams = [
    'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL',
    'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
]

# Filter the my_data table to keep only the rows with the allowed teams
# Combine "SEA" and "OKC" into "OKC" and "NOP" and "NOH" into "NOP"
my_data_filtered = my_data[my_data['team_abbreviation_home'].isin(allowed_teams) | my_data['team_abbreviation_away'].isin(allowed_teams)]

# Combine SEA and OKC into OKC, and NOP and NOH into NOP
my_data_filtered['team_abbreviation_home'] = my_data_filtered['team_abbreviation_home'].replace({'SEA': 'OKC', 'NOH': 'NOP'})
my_data_filtered['team_abbreviation_away'] = my_data_filtered['team_abbreviation_away'].replace({'SEA': 'OKC', 'NOH': 'NOP'})

# Now we can proceed with your original pivot table code with the filtered data

# Filter for clutch games (games where plus_minus_home or plus_minus_away is between -3 and 3)
clutch_games = my_data_filtered[(my_data_filtered['plus_minus_home'] >= -3) & (my_data_filtered['plus_minus_home'] <= 3) |
                                (my_data_filtered['plus_minus_away'] >= -3) & (my_data_filtered['plus_minus_away'] <= 3)]

# Function to determine if the home team won or lost (1-3 points)
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

# Filter for just "NOH"/"NOP" and "SEA"/"OKC" results
wins_matrix_filtered = wins_matrix[allowed_teams]
losses_matrix_filtered = losses_matrix[allowed_teams]
clutch_diff_filtered = clutch_diff[allowed_teams]

# Add average for each column at the bottom
wins_matrix_filtered.loc['Average'] = wins_matrix_filtered.mean()
losses_matrix_filtered.loc['Average'] = losses_matrix_filtered.mean()
clutch_diff_filtered.loc['Average'] = clutch_diff_filtered.mean()

# Sort the clutch difference matrix from high to low
clutch_diff_filtered_sorted = clutch_diff_filtered.sort_values(by='Average', axis=1, ascending=False)

# Display the filtered and sorted results
print("Clutch Wins Matrix (Filtered):")
print(wins_matrix_filtered)

print("\nClutch Losses Matrix (Filtered):")
print(losses_matrix_filtered)

print("\nClutch Wins - Losses Matrix (Filtered and Sorted):")
print(clutch_diff_filtered_sorted)


# %% [code] {"execution":{"iopub.status.busy":"2025-01-03T18:00:49.126306Z","iopub.execute_input":"2025-01-03T18:00:49.126648Z","iopub.status.idle":"2025-01-03T18:00:51.055275Z","shell.execute_reply.started":"2025-01-03T18:00:49.126619Z","shell.execute_reply":"2025-01-03T18:00:51.054143Z"}}
import matplotlib.pyplot as plt

# Assuming clutch_diff_filtered_sorted is already calculated and available

# Plot the data
clutch_diff_filtered_sorted.plot(kind='bar', figsize=(10, 6))

# Set plot title and labels
plt.title('Clutch Wins - Losses by Season (Sorted)', fontsize=16)
plt.xlabel('Season', fontsize=12)
plt.ylabel('Clutch Wins - Losses', fontsize=12)

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2025-01-03T18:00:51.056601Z","iopub.execute_input":"2025-01-03T18:00:51.057002Z","iopub.status.idle":"2025-01-03T18:00:52.815592Z","shell.execute_reply.started":"2025-01-03T18:00:51.056959Z","shell.execute_reply":"2025-01-03T18:00:52.814403Z"}}
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the average clutch wins minus losses for each team
average_clutch_diff = clutch_diff_filtered_sorted.mean(axis=0)

# Sort the teams by their average clutch difference (from lowest to highest)
average_clutch_diff_sorted = average_clutch_diff.sort_values()

# Reorder the data based on the sorted team averages
clutch_diff_sorted_teams = clutch_diff_filtered_sorted[average_clutch_diff_sorted.index]

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(clutch_diff_sorted_teams.T, annot=True, cmap="coolwarm", center=0, cbar_kws={'label': 'Clutch Wins - Losses'}, xticklabels=False)

# Set plot title
plt.title('Heatmap of Clutch Wins - Losses (Teams Sorted by Average)', fontsize=16)

# Show the plot
plt.tight_layout()
plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2025-01-03T18:06:51.039805Z","iopub.execute_input":"2025-01-03T18:06:51.040209Z","iopub.status.idle":"2025-01-03T18:06:52.996508Z","shell.execute_reply.started":"2025-01-03T18:06:51.040175Z","shell.execute_reply":"2025-01-03T18:06:52.995241Z"}}
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the average clutch wins minus losses for each team
average_clutch_diff = clutch_diff_filtered_sorted.mean(axis=0)

# Sort the teams by their average clutch difference (from highest to lowest)
average_clutch_diff_sorted = average_clutch_diff.sort_values(ascending=False)

# Round the averages to 1 decimal place
average_clutch_diff_sorted = average_clutch_diff_sorted.round(1)

# Reorder the data based on the sorted team averages
clutch_diff_sorted_teams = clutch_diff_filtered_sorted[average_clutch_diff_sorted.index]

# %% [code] {"execution":{"iopub.status.busy":"2025-01-03T18:06:51.039805Z","iopub.execute_input":"2025-01-03T18:06:51.040209Z","iopub.status.idle":"2025-01-03T18:06:52.996508Z","shell.execute_reply.started":"2025-01-03T18:06:51.040175Z","shell.execute_reply":"2025-01-03T18:06:52.995241Z"}}
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the average clutch wins minus losses for each team
average_clutch_diff = clutch_diff_filtered_sorted.mean(axis=0)

# Sort the teams by their average clutch difference (from highest to lowest)
average_clutch_diff_sorted = average_clutch_diff.sort_values(ascending=False)

# Round the averages to 1 decimal place
average_clutch_diff_sorted = average_clutch_diff_sorted.round(1)

# Reorder the data based on the sorted team averages
clutch_diff_sorted_teams = clutch_diff_filtered_sorted[average_clutch_diff_sorted.index]

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(clutch_diff_sorted_teams.T, annot=True, cmap="coolwarm", center=0, cbar_kws={'label': 'Clutch Wins - Losses'}, xticklabels=True)

# Set plot title
plt.title('Heatmap of Clutch Wins - Losses (Games Decided by <=3 Points)', fontsize=16)

# Show the plot
plt.tight_layout()
plt.show()
