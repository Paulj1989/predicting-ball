# specify db connection
con <- DBI::dbConnect(duckdb::duckdb(), dbdir = "data/club_football.duckdb")

DBI::dbSendQuery(con, "CREATE SCHEMA IF NOT EXISTS raw")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Player Values ----
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# set countries and seasons
COUNTRIES <- c("Germany")
SEASONS <- c(2020:2024)
crossed <- tidyr::crossing(COUNTRIES, SEASONS)

# specify function for team urls for multiple seasons and leagues
get_tm_team_urls <-
  function(countries, seasons) {
    worldfootballR::tm_league_team_urls(
      country_name = countries,
      start_year = seasons
    )
  }

# get team urls
team_urls <-
  purrr::map2(
    crossed$COUNTRIES,
    crossed$SEASONS,
    .f = get_tm_team_urls
  ) |>
  purrr::list_c()

# specify dataframe to store player values
player_values_raw <- tibble::tibble()

# loop through team urls and scrape player values
for (team in team_urls) {
  # print statement will allow you to see where you were blocked from
  print(paste0("Scraping ", team))
  # specify a pause
  Sys.sleep(5)
  df <- worldfootballR::tm_each_team_player_market_val(each_team_url = team)
  player_values_raw <- dplyr::bind_rows(player_values_raw, df)
}

player_values_raw <-
  player_values_raw |>
  janitor::clean_names()

# write player values to db
DBI::dbWriteTable(
  con,
  DBI::Id(schema = "raw", table = "player_values"),
  player_values_raw,
  overwrite = TRUE
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Match Logs ----
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# get league tables from fb ref
# league_table_raw <-
#   worldfootballR::fb_season_team_stats(
#     country = c("GER"),
#     gender = "M",
#     season_end_year = c(2023:2025),
#     tier = "1st",
#     stat_type = "league_table",
#     time_pause = 10
#   )

match_results_raw <-
  worldfootballR::fb_match_results(
    country = c("GER"),
    gender = "M",
    season_end_year = c(2021:2025),
    tier = "1st"
  ) |>
  janitor::clean_names(
    replace = c(
      "xGD" = "xgd",
      "xGA" = "xga",
      "xG" = "xg"
    )
  )

# write league tables to db
DBI::dbWriteTable(
  con,
  DBI::Id(schema = "raw", table = "match_results"),
  match_results_raw,
  overwrite = TRUE
)

# shut down db connection
DBI::dbDisconnect(con, shutdown = TRUE)
