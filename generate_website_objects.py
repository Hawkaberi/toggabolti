from datetime import datetime as dt
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import timedelta
import plotly.express as px
import json
import os
from dotenv import load_dotenv
import math

# Elo constants
initial_elo_rating = 1200

# Function to calculate the E value in Elo
def E_elo(r, ri):
    return 1 / (1 + 10 ** ((ri - r) / 400))

def get_k_factor():
    return 30

def calculate_rating_change(player, results):
    K = get_k_factor(player)
    total_delta = 0
    num_opponents = len(results)

    for match in results:
        opponent_rating = match['opponent']['elo_rating']
        score = match['score']
        expected_score = E_elo(player['elo_rating'], opponent_rating)
        base_delta = K * (score - expected_score)

        if score == 1:
            streak = player['win_streak']
        else:
            streak = player['loss_streak']

        streak_multiplier = min(1 + 0.1 * max(streak - 1, 0), 2)
        total_delta += base_delta * streak_multiplier

    normalized_change = total_delta / num_opponents if num_opponents > 0 else 0

    avg_opponent_rating = sum(r['opponent']['elo_rating'] for r in results) / num_opponents if num_opponents > 0 else player['elo_rating']
    avg_score = sum(r['score'] for r in results) / num_opponents if num_opponents > 0 else 0

    return {
        'leikmadur': player['leikmadur'],
        'elo_change': normalized_change,
        'avg_opponent_rating': avg_opponent_rating,
        'k_factor': K,
        'avg_score': avg_score,
        'win_streak': player['win_streak'],
        'loss_streak': player['loss_streak']
    }

def update_all_players(results, inactive_players, date=None):
    updated_players = []
    rating_changes = []

    # First pass: calculate rating changes based on original ratings
    for result in results:
        player = result['player']
        match_results = result['results']
        change_data = calculate_rating_change(player, match_results)
        rating_changes.append((player, change_data))
        player['isactive'] = True

    # Second pass: apply rating changes
    for player, change_data in rating_changes:        
        player['elo_rating'] += change_data['elo_change']
        player['display_elo'] = 1200 + (player['elo_rating'] - 1200) * 2.5
        player['games'] += 1
        updated_players.append(player)

    # Preserve inactive players
    updated_players.extend(inactive_players)

    return updated_players

# Load environment variables from .env file (if running locally)
load_dotenv()

# Get the credentials from the environment variable
credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')

if not credentials_json:
    raise ValueError("The environment variable GOOGLE_APPLICATION_CREDENTIALS_JSON is not set")

# Determine the credentials path
if 'GITHUB_ACTIONS' in os.environ:
    # Write the credentials to a temporary file in GitHub Actions
    credentials_path = '/tmp/google-credentials.json'
    with open(credentials_path, 'w') as f:
        f.write(credentials_json)
else:
    # Use the credentials path directly from the .env file when running locally
    credentials_path = credentials_json

print("Accessing Google Sheets")
# Set up credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
client = gspread.authorize(creds)

spreadsheet_titles = [sheet.title for sheet in client.openall()]
print(f"Accessible spreadsheet titles: {spreadsheet_titles}")

# Open the Google Sheet by title
sheet_title = "Toggaboltinn"
spreadsheet = client.open(sheet_title)

print("Loading data...")
# Select the worksheet by title
worksheet = spreadsheet.get_worksheet(0)  # Use 0 if it's the first worksheet
worksheet2 = spreadsheet.get_worksheet(1)

# Get all values from the worksheet
data = worksheet.get_all_values()
data2 = worksheet2.get_all_values()

# Create a Pandas DataFrame
df = pd.DataFrame(data[1:], columns=data[0])
df2 = pd.DataFrame(data2[1:], columns=data2[0])

print("Cleaning data...")
# Clean dataframes
df.columns = df.loc[1]
df = df.rename_axis(None, axis=1)
df.columns.values[0] = 'dagsetning'
df = df.loc[8:].copy()
df.reset_index(drop=True, inplace=True)
df['dagsetning'] = pd.to_datetime(df['dagsetning'], format='%d.%m.%y')
df = df.iloc[:, :-2]
df.drop(df.columns[1], axis=1, inplace=True)
numeric_columns = df.columns[1:]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

df2.columns.values[0] = 'dagsetning'
df2 = df2.loc[3:].copy()
df2.reset_index(drop=True, inplace=True)
df2['dagsetning'] = pd.to_datetime(df2['dagsetning'], format='%d.%m.%y')
df2.drop(df2.columns[1], axis=1, inplace=True)

print("Generating dictionary...")
# Generate dictionary to store each game
dagadict = {}
base_season_start = dt(2022, 11, 16)  # Base start date for season 1

for index, row in df.iterrows():
    date = row['dagsetning']

    # Calculate the season
    season = (date.year - base_season_start.year) + (1 if date >= base_season_start.replace(year=date.year) else 0)

    lid1 = [col for col in df.columns[1:] if row[col] == 1 and df2.at[index, col] == 'x']
    lid2 = [col for col in df.columns[1:] if row[col] == 0 and df2.at[index, col] == 'x']

    if not lid1:
        sigurvegari = 'tie'
    else:
        sigurvegari = 'lid1'

    dagadict[date] = {'lid1': lid1, 'lid2': lid2, 'sigurvegari': sigurvegari, 'season': season}
    
dagadict[dt(2023,1,4)]['lid1'] = ['Aron Skúli', 'Davíð', 'Ísak', 'Sindri', 'Þórarinn', 'Þórður']
dagadict[dt(2023,1,4)]['lid2'] = ['Alexander', 'Anton', 'Aron Skúli', 'Birgir Magnús', 'Haukur', 'Jóhann']

dagadict[dt(2023,7,26)]['lid1'] = ['Birgir Magnús', 'Einvarður', 'Haukur', 'Lárus', 'Sigurgeir']
dagadict[dt(2023,7,26)]['lid2'] = ['Alexander', 'Daníel Arnar', 'Hafsteinn', 'Jóhann', 'Þórður']

dagadict[dt(2023,12,7)]['lid1'] = ['Daníel Arnar', 'Haukur', 'Alexander', 'Birgir Magnús', 'Jóhann', 'Sigurgeir', 'Davíð']
dagadict[dt(2023,12,7)]['lid2'] = ['Gylfi', 'Þórhallur', 'Hafsteinn', 'Ísak', 'Eggert Georg', 'Þórður', 'Sindri']

dagadict[dt(2024,1,4)]['lid1'] = ['Þórður', 'Hafsteinn', 'Anton', 'Ísak', 'Eggert Georg']
dagadict[dt(2024,1,4)]['lid2'] = ['Daníel Arnar', 'Sindri', 'Birgir Magnús', 'Gylfi', 'Þórhallur']

dagadict[dt(2024,4,18)]['lid1'] = ['Þórður', 'Daníel Arnar', 'Hafsteinn', 'Haukur', 'Lárus']
dagadict[dt(2024,4,18)]['lid2'] = ['Máni Þór', 'Þórarinn', 'Birgir Magnús', 'Jakob Jóhann', 'Anton']

dagadict[dt(2024,8,22)]['lid1'] = ['Jakob Jóhann', 'Gunnar', 'Haukur', 'Eggert Georg', 'Hafsteinn', 'Máni Þór']
dagadict[dt(2024,8,22)]['lid2'] = ['Sigurgeir', 'Þórður', 'Daníel Snorri', 'Davíð', 'Anton', 'Birgir Magnús']

dagadict[dt(2024,9,12)]['lid1'] = ['Anton', 'Símon', 'Sigurgeir', 'Eggert Georg', 'Jakob Jóhann', 'Þórarinn', 'Egill (Atli)?']
dagadict[dt(2024,9,12)]['lid2'] = ['Þórður', 'Daníel Snorri', 'Þórhallur', 'Jón', 'Alexander', 'Grímur', 'Victor']

dagadict[dt(2024,11,7)]['lid1'] = ['Alexander', 'Birgir Magnús', 'Sigurgeir', 'Þórður', 'Davíð', 'Jóhann']
dagadict[dt(2024,11,7)]['lid2'] = ['Haukur', 'Eggert Georg', 'Símon', 'Sindri', 'Máni Þór', 'Daníel Snorri']

dagadict[dt(2024,12,24)]['lid1'] = ['Alexander', 'Davíð', 'Þórður', 'Sindri', 'Daníel Snorri', 'Sigurgeir', 'Anton']
dagadict[dt(2024,12,24)]['lid2'] = ['Birgir Magnús', 'Daði Snær', 'Þórarinn', 'Jakob Jóhann', 'Eggert Georg', 'Máni Þór', 'Haukur']

dagadict[dt(2025,6,12)]['lid1'] = ['Davíð', 'Máni Þór', 'Gunnar', 'Sindri', 'Þórður', 'Símon']
dagadict[dt(2025,6,12)]['lid2'] = ['Sigurgeir', 'Alexander', 'Birgir Magnús', 'Haukur', 'Jakob Jóhann', 'Daníel Snorri']

dagadict[dt(2025,8,21)]['lid1'] = ['Gunnar', 'Gunnar (vinnufélagi Dabba)', 'Alexander', 'Siggi Baddi', 'Ásgeir', 'Sindri', 'Þórður']
dagadict[dt(2025,8,21)]['lid2'] = ['Franz', 'Þórarinn', 'Jón Rúnar', 'Haukur', 'Davíð', 'Birgir Magnús', 'Eggert Georg']

dagadict[dt(2025,8,28)]['lid1'] = ['Tómas Daði', 'Birgir Magnús', 'Jakob Jóhann', 'Sindri', 'Þórður', 'Davíð', 'Siggi Baddi']
dagadict[dt(2025,8,28)]['lid2'] = ['Jón Rúnar', 'Daníel Snorri', 'Gunnar (vinnufélagi Dabba)', 'Alexander', 'Haukur', 'Eggert Georg', 'Gunnar']

print("Calculating ratings...")
leikmannalist = []
for leikmadur in df.columns[1:]:
    player = {
        'leikmadur': leikmadur,
        'elo_rating': initial_elo_rating,
        'display_elo': initial_elo_rating,
        'games': 0,
        'sigrar': 0,
        'jafntefli': 0,
        'top': 0,
        'win_streak': 0,
        'loss_streak': 0,
        'isactive': False
    }
    leikmannalist.append(player)

leikmanna_dict = {player['leikmadur']: player for player in leikmannalist}

for date, values in dagadict.items():
    results = []

    if values['sigurvegari'] == 'lid1':
        winners = values['lid1']
        losers = values['lid2']

        for player_name in winners:
            player = leikmanna_dict[player_name]
            player['sigrar'] += 1
            
            # Update streaks
            player['win_streak'] += 1
            player['loss_streak'] = 0

            results.append({
                'player': player,
                'results': [{'opponent': leikmanna_dict[opp], 'score': 1} for opp in losers]
            })

        for player_name in losers:
            player = leikmanna_dict[player_name]
            player['top'] += 1

            # Update streaks
            player['loss_streak'] += 1
            player['win_streak'] = 0

            results.append({
                'player': player,
                'results': [{'opponent': leikmanna_dict[opp], 'score': 0} for opp in winners]
            })

    elif values['sigurvegari'] == 'lid2':
        winners = values['lid2']
        losers = values['lid1']

        for player_name in winners:
            player = leikmanna_dict[player_name]
            player['sigrar'] += 1

            # Update streaks
            player['win_streak'] += 1
            player['loss_streak'] = 0

            results.append({
                'player': player,
                'results': [{'opponent': leikmanna_dict[opp], 'score': 1} for opp in losers]
            })

        for player_name in losers:
            player = leikmanna_dict[player_name]
            player['top'] += 1

            # Update streaks
            player['loss_streak'] += 1
            player['win_streak'] = 0

            results.append({
                'player': player,
                'results': [{'opponent': leikmanna_dict[opp], 'score': 0} for opp in winners]
            })

    elif values['sigurvegari'] == 'tie':
        team1 = values['lid1']
        team2 = values['lid2']

        for player_name in team1:
            player = leikmanna_dict[player_name]
            player['jafntefli'] += 1

            # Reset streaks on tie
            player['win_streak'] = 0
            player['loss_streak'] = 0

            results.append({
                'player': player,
                'results': [{'opponent': leikmanna_dict[opp], 'score': 0.5} for opp in team2]
            })

        for player_name in team2:
            player = leikmanna_dict[player_name]
            player['jafntefli'] += 1

            # Reset streaks on tie
            player['win_streak'] = 0
            player['loss_streak'] = 0

            results.append({
                'player': player,
                'results': [{'opponent': leikmanna_dict[opp], 'score': 0.5} for opp in team1]
            })

    # Update all players
    # Combine both teams' ids into a single list
    active_player_ids = values['lid1'] + values['lid2']

    # Get the list of player dictionaries who are not on either team
    inactive_players = [player for player in leikmannalist if player['leikmadur'] not in active_player_ids]

    leikmannalist = update_all_players(results, inactive_players, date=date)

    # Convert the list of dictionaries to a DataFrame
    leikmanna_df = pd.DataFrame(leikmannalist)

    # Convert the DataFrame back to a list of dictionaries
    leikmannalist = leikmanna_df.to_dict(orient='records')

    leikmanna_dict = {player['leikmadur']: player for player in leikmannalist}

    players_df = pd.DataFrame(leikmannalist)

    dagadict[date]['leikmenn'] = players_df

players_df['win_percentage'] = (players_df['sigrar'] / players_df['games']) * 100
players_df['tie_percentage'] = (players_df['jafntefli'] / players_df['games']) * 100
players_df['lose_percentage'] = (players_df['top'] / players_df['games']) * 100

players_dict = players_df.reset_index().to_dict(orient='records')

# Save player stats to a JSON file
with open('./player_stats.json', 'w') as f:
    json.dump(players_dict, f)

print("Generating rating plot...")
# Combine all DataFrames into one
combined_df = pd.DataFrame()
for date, values in dagadict.items():
    df = values['leikmenn'].copy()
    df['date'] = date
    combined_df = pd.concat([combined_df, df[df['isactive'] == True]])

# Ensure the date column is in datetime format
combined_df['date'] = pd.to_datetime(combined_df['date'])

# Find the first instance of each leikmadur
first_instances = combined_df.groupby('leikmadur')['date'].min().reset_index()

# Create default rows with date one day earlier than the first instance
default_rows = []
for _, row in first_instances.iterrows():
    default_row = {
        'leikmadur': row['leikmadur'],
        'date': row['date'] - timedelta(days=1),
        'elo_rating': initial_elo_rating,
        'display_elo': initial_elo_rating,
        'games': 0,
        'sigrar': 0,
        'jafntefli': 0,
        'top': 0,
        'isactive': True
    }
    default_rows.append(default_row)

# Convert the list of default rows to a DataFrame
default_df = pd.DataFrame(default_rows)

# Append the default rows to the original DataFrame
combined_df = pd.concat([combined_df, default_df], ignore_index=True)

# Sort the DataFrame by date
combined_df = combined_df.sort_values('date').reset_index(drop=True)

# Specify the list of players to show on the graph
players_to_show = list(set(players_df.loc[players_df['games'] > 0, 'leikmadur'].tolist()))  # Replace with your player names

# Define Icelandic alphabet sorting
icelandic_alphabet = 'aábdðeéfghiíjklmnoópqrstuúvxyýzþæö'
char_order = {char: i for i, char in enumerate(icelandic_alphabet)}

def icelandic_sort_key(name):
    name = name.lower()
    return [char_order.get(c, 1000) for c in name]

# Sort players by Icelandic alphabet
players_to_show = sorted(players_to_show, key=icelandic_sort_key)

# Filter the DataFrame to include only the specified players
filtered_df = combined_df[combined_df['leikmadur'].isin(players_to_show)]

# Sort by player name to control legend order
filtered_df['leikmadur'] = pd.Categorical(
    filtered_df['leikmadur'],
    categories=players_to_show,
    ordered=True
)
filtered_df = filtered_df.sort_values(by=['leikmadur', 'date'])

# Plot the Glicko ratings over time for the specified players using Plotly
fig = px.line(filtered_df, x='date', y='display_elo', color='leikmadur', 
              labels={'date': 'Dagsetning', 'display_elo': 'Togga Stig'}, 
              title='Þróun Togga Stiga')

# Customize hover data to include other relevant columns and remove the default hover label
fig.update_traces(hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Togga Rating: %{y}<extra></extra>')

# Update the layout to add buttons and customize the legend
fig.update_layout(
    height=750,  # Increase the height of the plot to make it more square
    updatemenus=[
        {
            "buttons": [
                {
                    "args": [{"visible": True}],
                    "label": "Velja Allt",
                    "method": "restyle"
                },
                {
                    "args": [{"visible": "legendonly"}],
                    "label": "Velja Ekkert",
                    "method": "restyle"
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 10},
            "showactive": True,
            "type": "buttons",
            "x": 1,
            "xanchor": "left",
            "y": 1.1,
            "yanchor": "top"
        }
    ],
    legend=dict(
        title_text='',  # Remove the title of the legend
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.05,
        itemsizing='constant',
        font=dict(size=12)  # Make the legend text smaller
    ),
    margin=dict(l=10, r=10, t=40, b=20)
)

# Save the figure as an HTML file
#pio.write_html(fig, file='plot.html', auto_open=True)

# Convert Plotly graph to JSON
graph_json = fig.to_json()

import os
if os.path.exists("plotly_graph.json"):
    os.remove("plotly_graph.json")

# Save the JSON to a file
with open('./plotly_graph.json', 'w') as f:
    f.write(graph_json)