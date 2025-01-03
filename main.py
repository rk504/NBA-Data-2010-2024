import pandas as pd

# Load the data
file_path = "regular_season_box_scores_2010_2024_part_1.csv"
data = pd.read_csv(file_path)

# Ensure relevant columns exist
required_columns = {'season', 'team', 'opponent', 'team_points', 'opponent_points'}
if not required_columns.issubset(data.columns):
    raise ValueError(f"Missing required columns: {required_columns - set(data.columns)}")

# Calculate point differentials and losses by <3 points
data['point_differential'] = data['team_points'] - data['opponent_points']
close_losses = data[(data['point_differential'] > -3) & (data['point_differential'] < 0)]

# Count losses by team
team_losses = close_losses.groupby('team').size().sort_values(ascending=False)

# Display results
print("Most losses by <3 points (2010-2024):")
print(team_losses)

# Optionally save results to a CSV
output_path = "close_losses_summary.csv"
team_losses.to_csv(output_path, header=["Losses by <3 Points"])
print(f"Results saved to {output_path}")
