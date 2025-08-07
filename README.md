# 🏙️ Urban Ecology Research Trends

Analyzing keyword trends, publication growth, and thematic similarity in urban ecology literature.

## Project Goals

- Identify key research topics in urban ecology
- Measure co-occurrence and thematic similarity
- Visualize trends across decades

## 📊 Tools

- Python (Pandas, Seaborn, Scikit-learn)
- Data: OpenAlex and Research Organization Registry (ROR), both via API


---

Database schema:
Papers table:

| Column           | Type    | Notes                    |
| ---------------- | ------- | ------------------------ |
| `paperId`        | TEXT    | Primary Key              |
| `title`          | TEXT    |                          |
| `year`           | INTEGER |                          |
| `authors`        | TEXT    | Comma-separated list of author IDs |
| `url`            | TEXT    |                          |
| `search_keyword` | TEXT    | Comma-separated list of keywords |
| `firstAuthorCountryIso` |  TEXT | alpha-2 ISO country code |
| `firstAuthorId`  | TEXT    | Foreign Key connects to Authors table |
| `journal`        | TEXT    |                                    |
| `citationCount`  | INTEGER |                          |

Authors table:

| Column          | Type | Notes                             |
| --------------- | ---- | --------------------------------- |
| `authorId`      | TEXT | Primary Key                       |
| `name`          | TEXT |                                   |
| `last_known_institution_name` | TEXT |                     |
| `last_known_institution_ror` | TEXT | To obtain more geographical data from ROR |
| `country_code`  | TEXT |                                   |
| `country_name`  | TEXT |                                   |



Table output test:

COMPREHENSIVE GROWTH ANALYSIS
================================================================================

MODEL COMPARISON RESULTS:
------------------------------------------------------------
              keyword  r_squared_linear  r_squared_log   better_fit  \
2     urban ecosystem          0.615529       0.952564  Exponential   
3  urban green spaces          0.615049       0.946375  Exponential   
1       urban ecology          0.646892       0.947195  Exponential   
0  urban biodiversity          0.674296       0.939802  Exponential   
4    urban vegetation          0.693672       0.946765  Exponential   
5      urban wildlife          0.719227       0.961730  Exponential   

   annual_growth_rate_percent  r_squared_difference  
2                   12.294559              0.337035  
3                   12.061090              0.331326  
1                   10.617228              0.300303  
0                   17.977592              0.265506  
4                    8.853252              0.253093  
5                    8.589296              0.242503  


LOGARITHMIC REGRESSION RESULTS (Testing Exponential Growth):
--------------------------------------------------------------------------------
              keyword  annual_growth_rate_percent  doubling_time_years  \
0  urban biodiversity                   17.977592             4.192646   
2     urban ecosystem                   12.294559             5.977714   
3  urban green spaces                   12.061090             6.086967   
1       urban ecology                   10.617228             6.869259   
4    urban vegetation                    8.853252             8.170969   
5      urban wildlife                    8.589296             8.411710   

   r_squared       p_value                    trend_interpretation  
0   0.939802  2.080995e-25  Significantly increasing (exponential)  
2   0.952564  4.273582e-36  Significantly increasing (exponential)  
3   0.946375  1.039787e-34  Significantly increasing (exponential)  
1   0.947195  6.963555e-35  Significantly increasing (exponential)  
4   0.946765  8.598278e-35  Significantly increasing (exponential)  
5   0.961730  1.600434e-38  Significantly increasing (exponential)  