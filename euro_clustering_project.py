# This script includes both data prep and modeling for my Euro 2024 clustering project 

# I. Data Preparation

#1. Getting a dataframe of what stage each team reached in major tournaments from 2008 to 2022
    # We will use three csvs to piece this together
    # 1. wc_team_finish.csv - Has the stage a team reached in the World Cup
    # 2. euro_matches_to_16.csv - Has all matches played in the Euros up to 2016 with a column indicating what stage the match was in
    # 3. eurocup_2020_results.csv - Has the stage a team reached in Euro 2020 (played in 2021)

# Load libraries
import pandas as pd
from datetime import datetime
import numpy as np 
from sklearn.cluster import KMeans 

# Read in our csv of World Cup team finishes
wc_team_finish_df = pd.read_csv("data/wc_team_finish.csv")

#Grab the year
wc_team_finish_df["year"] = wc_team_finish_df['tournament_id'].str.extract(r'(\d{4})').astype(int)

# Filter to mens
wc_team_finish_df = wc_team_finish_df[wc_team_finish_df['year'].apply(lambda year: year in [2010, 2014, 2018, 2022])]


# Filter to matches after 2010
wc_team_finish_df = wc_team_finish_df[wc_team_finish_df.year >= 2010]

# We now have a dataframe of World Cup finishes
wc_team_finish_df["comp"] = "World Cup"

wc_team_finish_df = wc_team_finish_df[["team_name", "year", "comp", "performance"]].rename(columns={'team_name': 'team',
                                                                                                    "performance": "max_stage"})

# Now let's piece together the Euro Cup finishes
# Read in csv of euro matches up until 2016
euro_matches_hist_df = pd.read_csv("data/euro_matches_to_16.csv")

# Filter to matches after 08
euro_matches_hist_df = euro_matches_hist_df[euro_matches_hist_df.Year >= 2008]

# Change diff group stage labels to Group Stage  
for i in range(len(euro_matches_hist_df.Stage)):
    if "Group" in euro_matches_hist_df.Stage.iloc[i]:
        euro_matches_hist_df.Stage.iloc[i] = "Group Stage"

# Make stage an ordered factor 
euro_matches_hist_df.Stage = pd.Categorical(euro_matches_hist_df['Stage'], 
                                  categories=['Group Stage', 'Round of 16', 'Quarter-finals', 'Semi-finals', 'Final'],
                                  ordered=True)

# Find the highest stage that each team reached as home
home_stage_reached = euro_matches_hist_df.groupby(["HomeTeamName", "Year"])["Stage"].max()

# Find the highest stage that each team reached as away
away_stage_reached = euro_matches_hist_df.groupby(["AwayTeamName", "Year"])["Stage"].max()

# Convert both of these to dfs
away_stage_reached_df = pd.DataFrame(away_stage_reached).reset_index().rename(columns={'AwayTeamName': 'Team'})
away_stage_reached_df["Team"] = away_stage_reached_df["Team"].apply(lambda team: str(team.strip()))

home_stage_reached_df = pd.DataFrame(home_stage_reached).reset_index().rename(columns={'HomeTeamName': 'Team'})
home_stage_reached_df["Team"] = home_stage_reached_df["Team"].apply(lambda team: str(team.strip()))

# Join the dfs
joined_stage_reached_df = away_stage_reached_df.merge(home_stage_reached_df, on = ["Team", "Year"], how = "left" )

# Grab max stage reached
# Switch back to factor
joined_stage_reached_df.Stage_y = pd.Categorical(joined_stage_reached_df.Stage_y, 
                                  categories=['Group Stage', 'Round of 16', 'Quarter-finals', 'Semi-finals', 'Final'],
                                  ordered=True)

joined_stage_reached_df.Stage_x = pd.Categorical(joined_stage_reached_df.Stage_x, 
                                  categories=['Group Stage', 'Round of 16', 'Quarter-finals','Semi-finals', 'Final'],
                                  ordered=True)

# Get the max stage
joined_stage_reached_df["max_stage"] = joined_stage_reached_df.apply(
        lambda row: max(row["Stage_x"], row["Stage_y"], key=lambda x: joined_stage_reached_df["Stage_x"].cat.categories.get_loc(x)),
        axis=1)

# Add in column indicating this was in the euros
joined_stage_reached_df["comp"] = "Euro"

# Grab the columns we want
euro_team_finish_hist_df = joined_stage_reached_df[["Team", "Year","comp", "max_stage"]]

# Rename columns to match other dfs
euro_team_finish_hist_df = euro_team_finish_hist_df.rename(columns={'Team': 'team',
                                                                    "Year": "year"})
                                                                                                    
# Now we do it for the 2020 Euros
euro_20_results_df = pd.read_csv("data/eurocup_2020_results.csv")


# Change labels to group stage
for i in range(len(euro_20_results_df.stage)):
    if "Group" in euro_20_results_df.stage.iloc[i]:
        euro_20_results_df.stage.iloc[i] = "Group Stage"

# Make stage an ordered factor 
euro_20_results_df.stage = euro_20_results_df.stage.apply(lambda stage: stage.strip())

euro_20_results_df.stage = pd.Categorical(euro_20_results_df['stage'], 
                                  categories=['Group Stage', 'Round of 16', 'Quarter-finals', 'Semi-finals', 'Final'],
                                  ordered=True)


# Add in year
euro_20_results_df["date"] = euro_20_results_df["date"].apply(lambda date_messy: pd.to_datetime(date_messy.strip(), dayfirst=True))
euro_20_results_df["year"] = euro_20_results_df["date"].apply(lambda dt: dt.year)

# Get max stage reached as home and away
home_stage_reached_20 = euro_20_results_df.groupby(["team_name_home"])["stage"].max()
away_stage_reached_20 = euro_20_results_df.groupby(["team_name_away"])["stage"].max()

# Convert both of these to dfs
away_stage_reached_20_df = pd.DataFrame(away_stage_reached_20).reset_index().rename(columns={'team_name_away': 'Team'})
home_stage_reached_20_df = pd.DataFrame(home_stage_reached_20).reset_index().rename(columns={'team_name_home': 'Team'})

# Join the dfs
joined_stage_reached_20_df = away_stage_reached_20_df.merge(home_stage_reached_20_df, on = ["Team"], how = "left" )

# Switch back to factor
joined_stage_reached_20_df.stage_y = pd.Categorical(joined_stage_reached_20_df.stage_y, 
                                  categories=['Group Stage', 'Round of 16', 'Quarter-finals', 'Semi-finals', 'Final'],
                                  ordered=True)

joined_stage_reached_20_df.stage_x = pd.Categorical(joined_stage_reached_20_df.stage_x, 
                                  categories=['Group Stage', 'Round of 16', 'Quarter-finals','Semi-finals', 'Final'],
                                  ordered=True)

# Get the max of these two categories
joined_stage_reached_20_df["max_stage"] = joined_stage_reached_20_df.apply(
        lambda row: max(row["stage_x"], row["stage_y"], key=lambda x: joined_stage_reached_20_df["stage_x"].cat.categories.get_loc(x)),
        axis=1)

# Add in year and comp columns
joined_stage_reached_20_df["year"] = 2020
joined_stage_reached_20_df["comp"] = "Euro"

# Grab columns we need
joined_stage_reached_20_df = joined_stage_reached_20_df[["Team", "year","comp", "max_stage"]].rename(columns={"Team": "team"})

joined_stage_reached_20_df.team = joined_stage_reached_20_df.team.apply(lambda team: team.strip())

# Combine all three dfs together
team_major_finish_df = pd.concat([joined_stage_reached_20_df, wc_team_finish_df, euro_team_finish_hist_df])
team_major_finish_df.max_stage = team_major_finish_df.max_stage.str.capitalize()


# Normalize stage strings
for i in range(len(team_major_finish_df.max_stage)):
    stage = (team_major_finish_df.max_stage.iloc[i])
    if stage == "Quarter-final":
        team_major_finish_df.max_stage.iloc[i] = "Quarter-finals"
    if stage == "Third-place match":
        team_major_finish_df.max_stage.iloc[i] = "Semi-finals"


# Function to clean a couple of team names
def clean_names(name):
    if name == "United States":
        return "USA"
    elif name == "Turkey":
        return "Turkiye"
    elif name == "Republic of Ireland":
        return "Ireland"
    else:
        return name
    
# Apply function
team_major_finish_df.team = team_major_finish_df.team.apply(clean_names)

# We know have our df of team major finishes to be used later on

# 2. Adding in data on previous performance to our dataset of major tournament participants

    # The variables we added were:
    # 1. last_major_finish - Stage reached by a team in their last major
    # 2. last_major_mean_gd - Average goal diff by a team in their last major
    # 3. last_two_yr_mean_gd - Average goal diff by a team in friendlies/qualifying matches in last two years
    # Note: For teams that did not make the last major they were eligible for, I gave them a last major finish of 0 and a goal diff of the worst among teams in majors

# Load libraries
import pandas as pd
import datetime as datetime
import regex as re

# Function to fix the name column in the historical team dataset
def extract_country_name(text):
    match = re.match(r'\d+\.\s*(\w+(\s+\w+)*)', text)
    if match:
        return match.group(1)
    else:
        return None
    
# Join historical and current datasets together
# Read in the dataset of historical teams from 2012 to 2022
hist_euro_data_df = pd.read_csv("data/historical_team_data.csv")

# Convert col names to lower case
hist_euro_data_df.columns = hist_euro_data_df.columns.str.lower()

# Take out the rating col
hist_euro_data_df.drop(columns=['rating'], inplace=True)

# fix the team column
hist_euro_data_df.team = hist_euro_data_df.team.apply(extract_country_name)

# Read in dataset of Euro 24 teams
euro_24_teams_df = pd.read_csv("data/euro_24_teams.csv")

# Join the datasets together
hist_euro_data_df = pd.concat([euro_24_teams_df,hist_euro_data_df])

# Let's read in our matches dataset
intl_matches_df = pd.read_csv("data/results.csv")

# Filter out games that have NA vals and that were before 08
intl_matches_df = intl_matches_df.dropna()

intl_matches_df['date'] = pd.to_datetime(intl_matches_df['date'])

intl_matches_df['year'] = intl_matches_df.date.dt.year

intl_matches_df = intl_matches_df[intl_matches_df['year'] >= 2008]

# Define function to weight home and away goal diff into overall goal differential - we will use this shortly
def get_weighted_gd(row):
    weighted_goal_df = (row.home_mean_goal_gd * (row.home_number_of_games / row.total_games)) + (row.away_mean_goal_gd * (row.away_number_of_games / row.total_games))
    return weighted_goal_df

# Let's get a df of how teams performed in major championships (Euro/WC)
# Filter down to matches in Euro/WC
major_comps_df = intl_matches_df[intl_matches_df.tournament.apply(lambda comp: comp in ["FIFA World Cup", "UEFA Euro"])]

# Make gd columns
major_comps_df["home_gd"] = major_comps_df["home_score"] - major_comps_df["away_score"]
major_comps_df["away_gd"] = major_comps_df["away_score"] - major_comps_df["home_score"]


# Make home and away df with goal differential
home_grouped_df = major_comps_df.groupby(["home_team", "year", "tournament"]).agg(
    home_mean_goal_gd=('home_gd', 'mean'),
    home_number_of_games=('home_gd', 'size')
).reset_index().rename(columns = {"home_team": "team"})

away_grouped_df = major_comps_df.groupby(["away_team", "year", "tournament"]).agg(
    away_mean_goal_gd=('away_gd', 'mean'),
    away_number_of_games=('away_gd', 'size')
).reset_index().rename(columns = {"away_team": "team"})


# Set all na values in the games and diff columns equal to 0
home_grouped_df['home_mean_goal_gd'] = home_grouped_df['home_mean_goal_gd'].fillna(0)
home_grouped_df['home_number_of_games'] = home_grouped_df['home_number_of_games'].fillna(0)
away_grouped_df['away_mean_goal_gd'] = away_grouped_df['away_mean_goal_gd'].fillna(0)
away_grouped_df['away_number_of_games'] = away_grouped_df['away_number_of_games'].fillna(0)

# Join H/A dfs into one df
full_grouped_df = home_grouped_df.merge(away_grouped_df, on = ["team", "year", "tournament"], how = "left")

# Set all na values in the games and diff columns equal to 0
full_grouped_df['home_mean_goal_gd'] = full_grouped_df['home_mean_goal_gd'].fillna(0)
full_grouped_df['home_number_of_games'] = full_grouped_df['home_number_of_games'].fillna(0)
full_grouped_df['away_mean_goal_gd'] = full_grouped_df['away_mean_goal_gd'].fillna(0)
full_grouped_df['away_number_of_games'] = full_grouped_df['away_number_of_games'].fillna(0)

# Add total games
full_grouped_df["total_games"] = full_grouped_df.away_number_of_games + full_grouped_df.home_number_of_games

# Get goal df overall in the tourney
full_grouped_df["total_mean_goal_df"] = full_grouped_df.apply(get_weighted_gd, axis = 1)

# Here we have a dataframe of goal diff for each team in the tournaments from
goal_diff_df = full_grouped_df[["team", "year", "tournament", "total_mean_goal_df"]].rename(columns= {"tournament": "comp"})

# Now let's convert 2021 to 2021
goal_diff_df.year = goal_diff_df.year.apply(lambda year: 2020 if year == 2021 else year)

# Function to fix the names of Euro/WC
def fix_tourney_names(tourney):
    if tourney == "UEFA Euro":
        return "Euro"
    if tourney == "FIFA World Cup":
        return "World Cup"

# Fix tourney names
goal_diff_df.comp = goal_diff_df.comp.apply(fix_tourney_names)

# We have our df of team goal differentials in tournaments

# Now, let's add it into our df of team finishes in tournaments
# Remeber our team_major_finish_df

# Function to convert string stage into numeric
# 1: Group Stage
# 2: L in R16
# 3: L in QF
#4: L in SF
#5: L in Finals
#6: Won
def convert_stage(row):
    # Return 6 for team that won the comp
    if row.year == 2008 and row.team == "Spain":
        return 6
    if row.year == 2010 and row.team == "Spain":
        return 6
    if row.year == 2012 and row.team == "Spain":
        return 6
    if row.year == 2014 and row.team == "Germany":
        return 6
    if row.year == 2016 and row.team == "Portugal":
        return 6
    if row.year == 2018 and row.team == "France":
        return 6
    if row.year == 2020 and row.team == "Italy":
        return 6
    if row.year == 2022 and row.team == "Argentina":
        return 6

    if row.max_stage == "Group stage":
        return 1
    if row.max_stage == "Round of 16":
        return 2
    if row.max_stage == "Quarter-finals":
        return 3
    if row.max_stage == "Semi-finals":
        return 4
    if row.max_stage == "Final":
        return 5

# Change the max stage col to numeric encoding
team_major_finish_df.max_stage = team_major_finish_df.apply(convert_stage, axis = 1)

# Merge the goal diff df with our max stage df
team_performance_df = goal_diff_df.merge(team_major_finish_df, how = "left", on = ["team", "year", "comp"])

# Define function to manually fix 4 rows that have NA for max stage (error)
def fix_na_stage(row):
    if row.team == "Austria" and row.year == 2008:
        return 1
    if row.team == "Greece" and row.year == 2008:
        return 1
    if row.team == "Netherlands" and row.year == 2008:
        return 2
    if row.team == "Switzerland" and row.year == 2008:
        return 1
    else:
        return row.max_stage

# Call function  
team_performance_df.max_stage = team_performance_df.apply(fix_na_stage, axis = 1)

# Let's fix euro 2021 to be Euro 2020 for both df
hist_euro_data_df.year = hist_euro_data_df.year.apply(lambda year: 2020 if year == 2021 else year)
team_performance_df.year = team_performance_df.year.apply(lambda year: 2020 if year == 2021 else year)

# Now we want to add in the team performance information to our dataframe of teams in major comps
# Define a function to get a teams finish in the previous major tournament
def get_last_major_finish(row):
    # Grab a df of majors in the 4 years before the current
    last_four_df = team_performance_df[(team_performance_df.year < row.year) & ((team_performance_df.year >= (row.year - 4))) & (team_performance_df.team == row.team)]
    if len((last_four_df)) == 0:
        return 0
    # Get the last row
    else:
        # Find the maximum year
        max_year = last_four_df['year'].max()

        # Filter the DataFrame to get the row where the year is the maximum
        max_year_row = last_four_df[last_four_df['year'] == max_year]

        max_year_finish_val = max_year_row["max_stage"].iloc[0]

        return max_year_finish_val


# Define a function to get a teams average goal diff in the previous major tournament
# Define worst average goal diff - used for teams that did not make the last major they were eligible for
min_goal_diff = team_performance_df.total_mean_goal_df.min()

# define the function
def get_last_major_gd(row):
    # Grab a df of majors in the 4 years before the current
    last_four_df = team_performance_df[(team_performance_df.year < row.year) & ((team_performance_df.year >= (row.year - 4))) & (team_performance_df.team == row.team)]
    if len((last_four_df)) == 0:
        return min_goal_diff
    # Get the last row
    else:
        # Find the maximum year
        max_year = last_four_df['year'].max()

        # Filter the DataFrame to get the row where the year is the maximum
        max_year_row = last_four_df[last_four_df['year'] == max_year]

        max_year_gd_val = max_year_row["total_mean_goal_df"].iloc[0]

        return max_year_gd_val

# Call the functions and get our new columns 
hist_euro_data_df["last_major_finish"] = hist_euro_data_df.apply(get_last_major_finish, axis = 1)
hist_euro_data_df["last_major_mean_gd"] = hist_euro_data_df.apply(get_last_major_gd, axis = 1)

# All that is left to do is add in team average goal diff in friendlies/qualifiers from the two years before they play a major tourney
# Filter our matches dataset down to non major matches
non_major_matches_df = intl_matches_df[intl_matches_df.tournament.apply(lambda tourney: tourney not in ["FIFA World Cup", "UEFA Euro"])]

# Add in month column
non_major_matches_df["month"] = non_major_matches_df.date.dt.month

# Define our function to iterate through the results data and get mean team goal_diff for games played two years before the majors tournaments
l2_goal_diff_list = []
def get_last_two_goal_df(row):
    goal_diff = 0
    games = 0
    max_bound = pd.to_datetime(f'{row.year}-{7:02d}-01')
    min_bound = pd.to_datetime(f'{row.year - 2}-{7:02d}-01')

    l2_yr_games_df = non_major_matches_df[(non_major_matches_df.date <= max_bound) &  (non_major_matches_df.date >= min_bound)]
    for i in range(len(l2_yr_games_df)):
        if row.team == l2_yr_games_df["home_team"].iloc[i]:
            goal_diff += l2_yr_games_df["home_score"].iloc[i] - l2_yr_games_df["away_score"].iloc[i]
            games += 1
        if row.team == l2_yr_games_df["away_team"].iloc[i]:
            goal_diff += l2_yr_games_df["away_score"].iloc[i] - l2_yr_games_df["home_score"].iloc[i]
            games += 1
    
    if games == 0:
        goal_diff = 0
    else:
        goal_diff = (goal_diff / games)
        
    l2_goal_diff_list.append(goal_diff)

# Call the function to fill out the list
hist_euro_data_df.apply(get_last_two_goal_df, axis = 1)

# Add in the column using the list
hist_euro_data_df.last_two_yr_mean_gd = pd.Series(l2_goal_diff_list)

# Fix typo in czech republic
def fix_czech(name):
    if name == "Czech Rebuplic":
        return "Czech Republic"
    else:
        return name
    
hist_euro_data_df.team = hist_euro_data_df.team.apply(fix_czech)

# We have our dataset that has added in our three variables called hist_euro_data_df

# 3. Before we conduct our modeling, we have to account for the inflation of player values from 2012 to 2024
# Get a dictionary of the average team value in major tournaments across the years
year_mean_mkt_val_dict = dict(hist_euro_data_df.groupby("year")["mkt_val"].mean())

# Define the current average team value 
current_mean_mkt_val = year_mean_mkt_val_dict[2024]

# Iterate through the df and put market values in terms of 2024 Euros (currency)
for i, row in hist_euro_data_df.iterrows():

    # Get adjusted team mkt val
    mean_mkt_val = year_mean_mkt_val_dict[row.year]
    # How much more/less is a dollar worth in 2024 than this row's year
    year_adjustment_factor = current_mean_mkt_val / mean_mkt_val
    # Calculate adjusted market value
    adjusted_mkt_val = row.mkt_val * year_adjustment_factor



    # Adjust average player value in the same way
    adjusted_mean_mkt_val = year_adjustment_factor * row.avg_mkt_val
    
    # Adjust the gk mkt val
    proportion_gk = row.goalie_mkt_val / row.mkt_val
    adj_gk_mkt_val = adjusted_mkt_val * proportion_gk

    # Def
    proportion_def = row.def_mkt_val / row.mkt_val
    adj_def_mkt_val = adjusted_mkt_val * proportion_def

    proportion_mean_def = row.def_mean_mkt_val / row.avg_mkt_val
    adj_mean_def_mkt_val = adjusted_mean_mkt_val * proportion_mean_def

    # Mid
    proportion_mid = row.mid_mkt_val / row.mkt_val
    adj_mid_mkt_val = adjusted_mkt_val * proportion_mid

    proportion_mean_mid = row.mid_mean_mkt_val / row.avg_mkt_val
    adj_mean_mid_mkt_val = adjusted_mean_mkt_val * proportion_mean_mid


    # Att
    proportion_att = row.att_mkt_val / row.mkt_val
    adj_att_mkt_val = adjusted_mkt_val * proportion_att

    proportion_mean_att = row.att_mean_mkt_val / row.avg_mkt_val
    adj_mean_att_mkt_val = adjusted_mean_mkt_val * proportion_mean_att

    # Now we actually alter the df vals
    hist_euro_data_df.iloc[i]["mkt_val"] = adjusted_mkt_val
    hist_euro_data_df.iloc[i]['avg_mkt_val'] = adjusted_mean_mkt_val

    hist_euro_data_df.iloc[i]['goalie_mkt_val'] = adj_gk_mkt_val

    hist_euro_data_df.iloc[i]['def_mkt_val'] = adj_def_mkt_val
    hist_euro_data_df.iloc[i]['def_mean_mkt_val'] = adj_mean_def_mkt_val

    hist_euro_data_df.iloc[i]['mid_mkt_val'] = adj_mid_mkt_val
    hist_euro_data_df.iloc[i]['mid_mean_mkt_val'] = adj_mean_mid_mkt_val
    
    hist_euro_data_df.iloc[i]['att_mkt_val'] = adj_att_mkt_val
    hist_euro_data_df.iloc[i]['att_mean_mkt_val'] = adj_mean_att_mkt_val

#print(hist_euro_data_df)
# We now have our model ready dataset

# II. Apply Clustering Model

# Drop columns that are not numeric
hist_euro_data_df_numeric = hist_euro_data_df.select_dtypes(include='number')
# Drop year this is not a predictor
hist_euro_data_df_numeric.drop(columns=['year'], inplace=True)

#print(hist_euro_data_df_numeric)

# Set seed
np.random.seed(0)

# Define k-means model and fit it to the data
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)

k_means.fit(hist_euro_data_df_numeric)

# Get our array of labels
k_means_labels = k_means.labels_

# Get our cluster centers 
k_means_cluster_centers = k_means.cluster_centers_

# Let's join our dataset with cluster each row belongs to
hist_euro_data_df["cluster"] = list(k_means_labels)

# Add in how teams did performance wise in that year
hist_euro_data_clusters_df = hist_euro_data_df.merge(team_major_finish_df, how = "left", on = ["team", "year", "comp"])

# Now we grab just the historical data to aggregate cluster level stats
hist_euro_data_clusters_df.dropna(inplace=True)

# We want a table of cluster, teams, average stage reached, proportion of winner, average mkt value
cluster_level_data = pd.DataFrame(hist_euro_data_clusters_df.groupby("cluster").agg(
    num_rows=('cluster', 'size'),
    avg_max_stage=('max_stage', 'mean'),
    max_stage = ('max_stage', "max"),
    avg_mkt_val=('mkt_val', 'mean'),
    prop_max_stage_6=('max_stage', lambda x: (x == 6).mean()),
    winners =  ('max_stage', lambda x: (x == 6).sum()))).reset_index()

# Function to add in our cluster names
def get_cluster_names(cluster):
    if cluster == 0:
        return "Dark Horses"
    if cluster == 1: 
        return "Giants"
    if cluster == 2:
        return "Contenders"
    if cluster == 3:
        return "Happy to be Here"
    
# Add in cluster names
cluster_level_data["cluster_name"] = cluster_level_data.cluster.apply(get_cluster_names)

# View Happy to be Here
print("Euro 24: Happy to be Here")
print(hist_euro_data_df[(hist_euro_data_df.cluster == 3) & (hist_euro_data_df.year == 2024)].team)

# View Dark Horses
print("Euro 24: Dark Horses")
print(hist_euro_data_df[(hist_euro_data_df.cluster == 0) & (hist_euro_data_df.year == 2024)].team)

# View Contenders
print("Euro 24: Contenders")
print(hist_euro_data_df[(hist_euro_data_df.cluster == 2) & (hist_euro_data_df.year == 2024)].team)

# View Giants
print("Euro 24: Giants")
print(hist_euro_data_df[(hist_euro_data_df.cluster == 1) & (hist_euro_data_df.year == 2024)].team)
