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

# Glicko-2 constants
tau = 0.5  # This is a system constant that influences the volatility over time.
initial_glicko_rating = 1500
initial_rd = 100  # Rating deviation
initial_vol = 0.06  # Initial volatility
glicko_weight = 0.5 # Hlutfall Glicko á móts við Elo í Togga Rating

# Elo constants
initial_elo_rating = 1500
K = 10  # K-factor for Elo rating system

# Function to convert rating and deviation to Glicko-2 scale
def glicko2_scale(rating, rd):
    return (rating - 1500) / 173.7178, rd / 173.7178

# Function to convert rating and deviation back to Glicko scale
def glicko_scale(rating, rd):
    return 173.7178 * rating + 1500, 173.7178 * rd

# Function to calculate the g value
def g(rd):
    return 1 / math.sqrt(1 + 3 * (rd ** 2) / (math.pi ** 2))

# Function to calculate the E value in Glicko-2
def E_glicko(r, ri, rdi):
    return 1 / (1 + math.exp(-g(rdi) * (r - ri)))

# Function to calculate the E value in Elo
def E_elo(r, ri):
    return 1 / (1 + 10 ** ((ri - r) / 400))

def vol_func(x, delta, vol, v, a):
    exp_x = math.exp(x)
    return exp_x * (delta ** 2 - vol ** 2 - v - exp_x) / (2 * (vol ** 2 + v + exp_x) ** 2) - (x - a) / (tau ** 2)

# Function to calculate the updated volatility
def updated_volatility(vol, delta, v, tau):
    a = math.log(vol ** 2)
    A = a
    B = None
    if delta ** 2 > v + vol ** 2:
        B = math.log(delta ** 2 - v - vol ** 2)
    else:
        k = 1
        while vol_func(a - k * tau, delta, vol, v, a) < 0:
            k += 1
        B = a - k * tau
    fA = vol_func(A, delta, vol, v, a)
    fB = vol_func(B, delta, vol, v, a)
    
    while abs(B - A) > 1e-6:
        C = A + (A - B) * fA / (fB - fA)
        fC = vol_func(C, delta, vol, v, a)
        if fC * fB < 0:
            A = B
            fA = fB
        else:
            fA = fA / 2
        B = C
        fB = fC

    return math.exp(A / 2)

# Function to update Glicko-2 ratings and deviations
def update_glicko_rating(player, results):
    r, rd = glicko2_scale(player['glicko_rating'], player['rd'])
    vol = player['vol']
    
    v_inv_sum = 0
    delta_sum = 0
    
    for result in results:
        opponent, score = result['opponent'], result['score']
        ri, rdi = glicko2_scale(opponent['glicko_rating'], opponent['rd'])
        
        E_val = E_glicko(r, ri, rdi)
        g_val = g(rdi)
        
        v_inv_sum += g_val ** 2 * E_val * (1 - E_val)
        delta_sum += g_val * (score - E_val)
    
    v = 1 / v_inv_sum
    delta = v * delta_sum
    
    new_vol = updated_volatility(vol, delta, v, tau)
    
    pre_rd = math.sqrt(rd ** 2 + new_vol ** 2)
    
    new_rd = 1 / math.sqrt(1 / (pre_rd ** 2) + 1 / v)
    new_rating = r + new_rd ** 2 * delta_sum
    
    player['glicko_rating'], player['rd'] = glicko_scale(new_rating, new_rd)
    player['vol'] = new_vol
    player['games'] += 1  # Update the number of games played

    player['active_glicko'] = player['glicko_rating'] # Save the last active glicko_rating

    return player

# Function to update Elo ratings in a zero-sum manner
def update_elo_ratings(player1, player2, result):
    expected_score1 = E_elo(player1['elo_rating'], player2['elo_rating'])
    expected_score2 = E_elo(player2['elo_rating'], player1['elo_rating'])
    
    # result is 1 if player1 wins, 0 if player2 wins, 0.5 if draw
    change1 = K * (result - expected_score1)
    change2 = K * ((1 - result) - expected_score2)
    
    player1['elo_rating'] += change1
    player2['elo_rating'] += change2
    
    return player1, player2

# Function to update both Glicko-2 and Elo ratings
def update_ratings(player, results):
    player = update_glicko_rating(player, results)
    for result in results:
        opponent = result['opponent']
        score = result['score']
        player, opponent = update_elo_ratings(player, opponent, score)
    return player

# Function to update all players' ratings
def update_all_players(results, inactive_players, max_rd=initial_rd):
    updated_players = []
    for result in results:
        player = result['player']
        match_results = result['results']
        player['time_factor'] = 0 # Reset time factor for active players
        player['isactive'] = True # Marks that a player has had a rating change and is subject to rate decay
        # if player['leikmadur'] == 'Þórður':
        #         print(f"""Fyrir update: {player}""")
        updated_player = update_ratings(player, match_results)
        # if player['leikmadur'] == 'Þórður':
        #         print(f"""Eftir update: {player}""")
        updated_players.append(updated_player)
    
    # Update RD for inactive players and apply the upper limit
    for player in inactive_players:
        player['time_factor'] += 1  # Increment time factor for inactive players
        player['rd'] = min(math.sqrt(player['rd']**2 + (2 * player['time_factor'])**2), max_rd)
        if player['isactive']:
            # if player['leikmadur'] == 'Þórður':
            #     print(f"""Fyrir decay: {player}""")
            player['glicko_rating'] = max(player['glicko_rating']-math.sqrt(player['rd']), player['active_glicko']-(player['rd']*2)) # Rating decay
            # if player['leikmadur'] == 'Þórður':
            #     print(f"""Eftir decay: {player}""")
        updated_players.append(player)
    
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

for index, row in df.iterrows():
    date = row['dagsetning']
    lid1 = [col for col in df.columns[1:] if row[col] == 1 and df2.at[index, col] == 'x']
    lid2 = [col for col in df.columns[1:] if row[col] == 0 and df2.at[index, col] == 'x']

    if not lid1:
        sigurvegari = 'tie'
    else:
        sigurvegari = 'lid1'

    dagadict[date] = {'lid1': lid1, 'lid2': lid2, 'sigurvegari': sigurvegari}

# Manually populate teams when result is tie
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

print("Calculating ratings...")
leikmannalist = []
for leikmadur in df.columns[1:]:
    player = {
        'leikmadur': leikmadur,
        'glicko_rating': initial_glicko_rating,
        'elo_rating': initial_elo_rating,
        'rd': initial_rd,
        'vol': initial_vol,
        'games': 0,
        'sigrar': 0,
        'jafntefli': 0,
        'top': 0,
        'time_factor': 0,
        'active_glicko': initial_glicko_rating,
        'isactive': False
    }
    leikmannalist.append(player)

leikmanna_dict = {player['leikmadur']: player for player in leikmannalist}

for date, values in dagadict.items():
    #print(dagadict[date])
    if values['sigurvegari'] == 'lid1':
        winners = values['lid1']
        losers = values['lid2']
        results = []
        for player in winners:
            player_dict = leikmanna_dict[player]
            player_dict['sigrar'] += 1  # Increment wins
            results.append({'player': player_dict, 'results': [{'opponent': leikmanna_dict[opponent], 'score': 1} for opponent in losers]})

        for player in losers:
            player_dict = leikmanna_dict[player]
            player_dict['top'] += 1  # Increment losses
            results.append({'player': player_dict, 'results': [{'opponent': leikmanna_dict[opponent], 'score': 0} for opponent in winners]})       
    elif values['sigurvegari'] == 'lid2':
        winners = values['lid2']
        losers = values['lid1']
        results = []
        for player in winners:
            player_dict = leikmanna_dict[player]
            player_dict['sigrar'] += 1  # Increment wins
            results.append({'player': player_dict, 'results': [{'opponent': leikmanna_dict[opponent], 'score': 1} for opponent in losers]})

        for player in losers:
            player_dict = leikmanna_dict[player]
            player_dict['top'] += 1  # Increment losses
            results.append({'player': player_dict, 'results': [{'opponent': leikmanna_dict[opponent], 'score': 0} for opponent in winners]}) 
    elif values['sigurvegari'] == 'tie':
        results = []
        for player in values['lid1']:
            player_dict = leikmanna_dict[player]
            player_dict['jafntefli'] += 1  # Increment ties
            results.append({'player': player_dict, 'results': [{'opponent': leikmanna_dict[opponent], 'score': 1} for opponent in values['lid2']]})

        for player in values['lid2']:
            player_dict = leikmanna_dict[player]
            player_dict['jafntefli'] += 1  # Increment ties
            results.append({'player': player_dict, 'results': [{'opponent': leikmanna_dict[opponent], 'score': 0} for opponent in values['lid1']]})

    # Update all players
    # Combine both teams' ids into a single list
    active_player_ids = values['lid1'] + values['lid2']

    # Get the list of player dictionaries who are not on either team
    inactive_players = [player for player in leikmannalist if player['leikmadur'] not in active_player_ids]

    leikmannalist = update_all_players(results, inactive_players)

    # Convert the list of dictionaries to a DataFrame
    leikmanna_df = pd.DataFrame(leikmannalist)

    # Calculate the average of rating1
    glicko_average = leikmanna_df['glicko_rating'].mean()

    # Calculate the scaling factor
    glicko_scaling = 1500 / glicko_average

    # Apply the scaling factor to rating1
    leikmanna_df['glicko_rating'] = leikmanna_df['glicko_rating'] * glicko_scaling

    # Update active_glicko for active players after scaling
    for player_id in active_player_ids:
        leikmanna_df.loc[leikmanna_df['leikmadur'] == player_id, 'active_glicko'] = leikmanna_df.loc[leikmanna_df['leikmadur'] == player_id, 'glicko_rating']

    # Convert the DataFrame back to a list of dictionaries
    leikmannalist = leikmanna_df.to_dict(orient='records')

    leikmanna_dict = {player['leikmadur']: player for player in leikmannalist}

    players_df = pd.DataFrame(leikmannalist)
    players_df['togga_rating'] = players_df['glicko_rating'] * glicko_weight + players_df['elo_rating'] * (1-glicko_weight)

    dagadict[date]['leikmenn'] = players_df

players_df['win_percentage'] = (players_df['sigrar'] / players_df['games']) * 100
players_df['tie_percentage'] = (players_df['jafntefli'] / players_df['games']) * 100
players_df['lose_percentage'] = (players_df['top'] / players_df['games']) * 100

players_dict = players_df.reset_index().to_dict(orient='records')

print("Saving player stats...")
# Save player stats to a JSON file
with open('player_stats.json', 'w') as f:
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
        'glicko_rating': initial_glicko_rating,
        'elo_rating': initial_elo_rating,
        'rd': initial_rd,
        'vol': initial_vol,
        'games': 0,
        'sigrar': 0,
        'jafntefli': 0,
        'top': 0,
        'time_factor': 0,
        'active_glicko': initial_glicko_rating,
        'isactive': True,
        'togga_rating': initial_elo_rating,
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

# Filter the DataFrame to include only the specified players
filtered_df = combined_df[combined_df['leikmadur'].isin(players_to_show)]

# Plot the Glicko ratings over time for the specified players using Plotly
fig = px.line(filtered_df, x='date', y='togga_rating', color='leikmadur', 
              labels={'date': 'Dagsetning', 'togga_rating': 'Togga Stig'}, 
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

# Convert Plotly graph to JSON
graph_json = fig.to_json()

print("Saving rating plot...")
# Save the JSON to a file
with open('./plotly_graph.json', 'w') as f:
    f.write(graph_json)