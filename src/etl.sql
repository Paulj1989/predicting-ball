-- Create Schema for Processed Data
CREATE SCHEMA IF NOT EXISTS club_football.processed;

-- Set Schema
-- USE club_football.processed;
-- Create Team Mappings Table
CREATE
OR REPLACE TABLE team_mapping (raw_name TEXT, standard_name TEXT);

-- Add Team Mappings
INSERT INTO
    team_mapping
VALUES
    ('1.FC Heidenheim 1846', 'Heidenheim'),
    ('1.FC Köln', 'Köln'),
    ('1.FC Union Berlin', 'Union Berlin'),
    ('1.FSV Mainz 05', 'Mainz'),
    ('Mainz 05', 'Mainz'),
    ('Arminia', 'Arminia Bielefeld'),
    ('Bayer 04 Leverkusen', 'Bayer Leverkusen'),
    ('Leverkusen', 'Bayer Leverkusen'),
    ('Gladbach', 'Borussia Mönchengladbach'),
    ('Dortmund', 'Borussia Dortmund'),
    ('Eint Frankfurt', 'Eintracht Frankfurt'),
    ('FC Augsburg', 'Augsburg'),
    ('FC Schalke 04', 'Schalke 04'),
    ('FC St. Pauli', 'St. Pauli'),
    ('SC Freiburg', 'Freiburg'),
    ('SV Darmstadt 98', 'Darmstadt'),
    ('Darmstadt 98', 'Darmstadt'),
    ('SpVgg Greuther Fürth', 'Greuther Fürth'),
    ('SV Werder Bremen', 'Werder Bremen'),
    ('TSG 1899 Hoffenheim', 'Hoffenheim'),
    ('VfB Stuttgart', 'Stuttgart'),
    ('VfL Bochum', 'Bochum'),
    ('VfL Wolfsburg', 'Wolfsburg');

-- Create Match Results Table
CREATE
OR REPLACE TABLE match_results AS
SELECT
    REPLACE(
        mr.competition_name,
        'Fußball-Bundesliga',
        'Bundesliga'
    ) AS comp_name,
    mr.country,
    mr.season_end_year - 1 AS season,
    mr.wk,
    mr.date,
    COALESCE(tm_home.standard_name, mr.home) AS home,
    mr.home_goals,
    mr.home_xg,
    COALESCE(tm_away.standard_name, mr.away) AS away,
    mr.away_goals,
    mr.away_xg
FROM
    raw.match_results mr
    LEFT JOIN team_mapping tm_home ON mr.home = tm_home.raw_name
    LEFT JOIN team_mapping tm_away ON mr.away = tm_away.raw_name
WHERE
    mr.wk IS NOT NULL;

-- Fix Types
ALTER TABLE processed.match_results
ALTER wk TYPE INTEGER;

-- Create Squad Values Table
-- CREATE
-- OR REPLACE TABLE squad_values AS
-- SELECT
--     pv.comp_name,
--     pv.country,
--     pv.season_start_year AS season,
--     COALESCE(tm.standard_name, pv.squad) AS team,
--     SUM(pv.player_market_value_euro) AS market_value
-- FROM
--     raw.player_values pv
--     LEFT JOIN team_mapping tm ON pv.squad = tm.raw_name
-- GROUP BY ALL;

-- -- Inspect Team Names
-- WITH
--     a_teams AS (
--         SELECT
--             ROW_NUMBER() OVER () AS row_id,
--             team
--         FROM
--             (
--                 SELECT DISTINCT
--                     team
--                 FROM
--                     squad_values
--                 ORDER BY team
--             )
--     ),
--     b_teams AS (
--         SELECT
--             ROW_NUMBER() OVER () AS row_id,
--             home
--         FROM
--             (
--                 SELECT DISTINCT
--                     home
--                 FROM
--                     match_results
--                 ORDER BY home
--             )
--     )
-- SELECT
--     a.team,
--     b.home
-- FROM
--     a_teams a
--     FULL OUTER JOIN b_teams b USING (row_id);

-- Create Final Training Data Table
CREATE
OR REPLACE TABLE main.club_seasons AS
SELECT
    mr.comp_name,
    mr.country,
    mr.season,
    mr.date,
    mr.wk,
    mr.home AS home_team,
    sv_home.market_value AS home_value,
    mr.home_goals,
    mr.home_xg,
    mr.away AS away_team,
    sv_away.market_value AS away_value,
    mr.away_goals,
    mr.away_xg
FROM
    match_results mr
    LEFT JOIN squad_values sv_home ON mr.home = sv_home.team
    AND mr.season = sv_home.season
    LEFT JOIN squad_values sv_away ON mr.away = sv_away.team
    AND mr.season = sv_away.season;
