## 🧱 Phase 1: **Data Acquisition**

**Goal**: Retrieve urban ecology research papers from Semantic Scholar.

### Key Components:

* 🔑 **API Access**:

  * Understand [Semantic Scholar’s API](https://api.semanticscholar.org/).
  * Handle authentication if needed.
* 🧾 **Query Design**:

  * Define search terms: `"urban ecology"`, `"urban biodiversity"`, etc.
  * Limit by date range, keywords, fields of study.
* 🪢 **Rate Limiting & Pagination**:

  * Implement logic to deal with paginated results and API limits.
* 📦 **Data Storage**:

  * Save as SQLite, then normalize into a pandas dataframe.

---

## 📊 Phase 2: **Exploratory Time Series Analysis**

**Goal**: Understand research output trends over time.

### Key Components:

* 📆 **Extract & Clean Publication Dates**.
* 📈 **Group by Year/Month** and count publications.
* 🔁 **Rolling averages**, highlight spikes, dips.
* 📅 **Optionally normalize by total scientific output per year** (to spot relative trends).

---

## 🌍 Phase 3: **Keyword Correlation Analysis**

**Goal**: Analyze keyword coocurrence among the papers in the database, identifying which appear together more often, and which appear by itslef the most.

### Key Components:

1. Data Preparation
 Extract paperId and search_keyword from the database

 Clean and split comma-separated keywords into lists

 Explode into one keyword per row if needed (optional for some analyses)

 Create a binary keyword–paper matrix (papers x keywords)

📊 2. Keyword Frequency & Solo Ratio
 Count total papers for each keyword

 Count how often a keyword appears alone (solo)

 Calculate solo ratio: solo / total, rounded

🔁 3. Keyword Co-occurrence
 Create a co-occurrence matrix (number of shared papers between each keyword pair)

 Normalize co-occurrence matrix if needed (e.g., by row total, Jaccard index, or PMI)

 Visualize as a heatmap (e.g., using seaborn.heatmap)

📐 4. Jaccard Similarity Matrix
 Build sets of paper IDs for each keyword

 Compute Jaccard index for each pair
​
 
 Store in a matrix or long-form table

 Visualize as a heatmap or network graph (optional)

📈 5. Pearson Correlation of Time Trends
 Create a yearly count of papers per keyword (year x keyword)

 Normalize per year if comparing trends (e.g., keyword frequency ÷ total papers that year)

 Detrend if needed (to remove overall publication growth effect)

 Compute Pearson correlation between keywords across years

🌟 6. Optional/Advanced Analyses
 Network graph of co-occurring keywords (nodes = keywords, edges = co-occurrence or Jaccard)

 Hierarchical clustering or t-SNE/PCA to group keywords by similarity

 Association rules / lift if treating keywords like market basket data

 Keyword centrality in the network: find the most connected/influential topics

 Identify bridge keywords: keywords that co-occur with multiple otherwise disconnected topics

---

## 🌍 Phase 4: **Author/Institution Geographic Mapping**

**Goal**: Visualize the spatial distribution of research activity.
use author endpoint from semantic scholar to obtain institution data
Use this data with Research Organization Registry (ROR) to obtain geographical data

### Key Components:

* 👤 **Author Disambiguation**:

  * Clean and standardize author/institution names.
* 🏫 **Institution Geolocation**:

  * Use an API (e.g., Google Maps, OpenStreetMap) to map locations.
* 🗺️ **Map Visualization**:

  * Use `folium`, `geopandas`, or `plotly` to create interactive or static maps.

---

## 📚 Phase 5: **Presentation & Storytelling**

**Goal**: Deliver insights in a clean, reproducible format.

### Key Components:

* 📓 **Jupyter Notebook Layout**:

  * Clear separation of sections: Intro → Method → Results → Discussion.
* 🖼️ **Visualizations**:

  * Use Seaborn, Plotly, or Altair for expressive plots.
* 🧪 **Reproducibility**:

  * Fix random seeds, modularize functions, optionally include a `requirements.txt`.

---

## 🛠️ Optional Add-ons (Stretch Goals)

* 📥 **Cache API responses** (using `joblib`, `pickle`, or local DB).
* 🧠 **NER or citation network analysis**.
* 📤 **Deploy as a Streamlit app** or export a PDF report.

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
| `search_keyword` | TEXT    | For traceability         |
| `location`    | TEXT | country-code from first authors |
| `firstAuthorId`  | TEXT    | Foreign key to `authors` |

Authors table:

| Column        | Type | Notes                             |
| ------------- | ---- | --------------------------------- |
| `authorId`    | TEXT | Primary Key                       |
| `name`        | TEXT | From author API                   |
| `affiliation` | TEXT | From author API                   |
| `ror`         | TEXT | To obtain city data from ror if desired  |  


OpenAlex citation: 
Priem, J., Piwowar, H., & Orr, R. (2022). OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts. ArXiv. https://arxiv.org/abs/2205.01833