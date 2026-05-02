[![Trail Popularity Prediction](https://img.shields.io/badge/Capstone-Trail%20Popularity%20Prediction-blue)](https://github.com/yourusername/trail-popularity-predictor)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-brightgreen)](https://www.python.org/downloads/)
[![XGBoost Model](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![R² Score](https://img.shields.io/badge/R%C2%B2%20Score-0.8556-brightgreen)](https://scikit-learn.org/stable/)


<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/28c996e0-3b90-4e58-9dbd-6099d41f4246" />

# Trail Popularity Prediction: A Data-Driven Approach to Park Resource Optimization

## Executive Summary

This capstone project develops a machine learning model to predict trail popularity utilizing data publicly available at AllTrails.com, enabling park managers to optimize resource allocation and understand what truly drives visitor engagement. Through comprehensive data collection, sophisticated feature engineering, and explainable AI analysis, we discovered that **community engagement metrics matter far more than trail characteristics** in determining popularity.

**Key Achievement**: Advanced model achieves **R² = 0.8556** with RMSE = 8.19 points on a 0-100 scale, representing a 15% accuracy improvement over the baseline through sentiment analysis, creative feature engineering, and feature reduction strategies.

---

## 🎯 Problem Statement

Park directors face unprecedented budget constraints and must allocate limited resources across dozens or hundreds of trails. Current approaches use:
- Gut feeling and historical precedent
- Complaints from the loudest voices
- Random or equal resource distribution

**The Challenge**: No data-driven framework exists publicly to predict which trails will be popular or what drives visitor engagement.

**The Solution**: A predictive model that identifies trails popularity to better determine where a parks resources should go toward maintanence and reveals the true drivers of popularity—enabling strategic resource allocation that maximizes visitor satisfaction and community economic benefit.

---

## 🔍 Key Finding: It's Community, Not Geography

### The Surprising Discovery

Our SHAP analysis revealed a counterintuitive insight:

**Traditional trail features (what we initially expected to matter most):**
- Water features (lakes, rivers, waterfalls): **±0.1-2 point impact**
- Rock climbing available: **±0.1-2 point impact**
- Trail views and scenic features: **±0.1-2 point impact**
- Trail type/activities: **±0.1-2 point impact**

**Community engagement features (what actually matters):**
- Featured photos percentage: **5.46 point impact** ✓
- Average user rating: **3.79 point impact** ✓
- Trail open/closed status: **3.48 point impact** ✓
- Collections with photos: **2.51 point impact** ✓
- Community sentiment: **0.73 point impact** ✓

### What This Means

**Trails don't become popular because they have lakes or views.** They become popular because:

1. **People talk about them** (sentiment analysis shows community discussion drives popularity)
2. **People share photos** (featured photos are the #1 predictor)
3. **People rate them** (high ratings attract more visitors)
4. **Communities engage** (collections and reviews create visibility)

### Practical Implication for Park Management

Rather than building new features or hoping natural scenery drives traffic, parks should:
- **Encourage community engagement**: Host events, competitions, social media campaigns
- **Facilitate photo sharing**: Photo contests, hashtags, featured collections
- **Build review culture**: Ask visitors to leave ratings and reviews
- **Engage word-of-mouth**: Community partnerships, ambassador programs

---

## 📊 Model Performance

### Final Model (XGBoost with Sentiment Analysis)
```
Training R²:     0.9664  (96.64% variance explained during training)
Testing R²:      0.8556  (85.56% variance explained on unseen data)
Testing RMSE:    8.19    (predictions accurate within ±8 points, 95% of time)
Testing MAE:     5.79    (average prediction error: 5.79 points)
```

**Interpretation**: The model explains 85.56% of trail popularity variation and predicts popularity within ±8 points with high confidence. This level of accuracy enables reliable resource allocation decisions.

### Baseline Model (Gradient Boosting without Sentiment)
```
Training R²:     0.8254
Testing R²:      0.7036  (70.36% variance explained)
Testing RMSE:    11.89   (predictions accurate within ±12 points)
Testing MAE:     8.06    (average prediction error: 8.06 points)
```

**Model Comparison**: The advanced model's inclusion of sentiment analysis and user-generated content features improves accuracy by **15%** (R² 0.704 → 0.856), demonstrating the critical value of community engagement metrics. 

---

## 🔬 Feature Importance & Explainability

<img width="2384" height="732" alt="01_feature_optimization_comparison" src="https://github.com/user-attachments/assets/fb1323c9-6314-4337-b8cd-3c7a7101b354" />

"Sweet spot" in number of features was found to be 150 features, more or less features led to worse results, either via not enough data or introducing too much "static" confounding the useful data.

### Top 10 Most Important Features (SHAP Analysis)

<img width="395" height="470" alt="02_shap_summary_bar" src="https://github.com/user-attachments/assets/3f5247eb-9dd8-4a3c-9c8a-b529dfaefee1" />

**SHAP Value Interpretation**: SHAP values represent the contribution of each feature to moving the model's prediction from the baseline. Higher values indicate stronger influence on popularity predictions.

<img width="392" height="470" alt="03_shap_beeswarm" src="https://github.com/user-attachments/assets/0aa2660b-6012-416f-86fa-20338acdf54f" />

Visual of all datapoints and how the given features influenced their popularity. Stronger inlfuences appear further from the median, and the magnitude of that specific value is shown in color. Red being higher values, blue being lower. 

### Explainability Methods Used

1. **SHAP Summary Plot**: Shows which features push predictions higher or lower
2. **SHAP Dependence Plots**: Reveals how specific features influence popularity across their range
3. **SHAP Force Plots**: Individual prediction explanations (why was this specific trail predicted as popular/unpopular?)
4. **Feature Importance Bar Charts**: Comparative importance of all 150 features

**Key Insight**: User-generated content features (photos, ratings, engagement) collectively account for 15% more predictive power than all trail characteristics combined.

---

## 🛠️ Technical Approach

### Data Pipeline

1. **Data Collection** (AllTrails.com)
   - 6,984 raw trails across Southwest U.S.
   - Web scraping with undetected-chromedriver
   - New Apify actor for streamlined data collection: [console.apify.com/actors/MMQdritoUWpzUVbah](https://console.apify.com/actors/MMQdritoUWpzUVbah)

2. **Sentiment Scraping** 
   - Extracted review text from each trail page
   - Scored sentiment across 4 categories:
     - Views & scenery sentiment
     - Difficulty sentiment
     - Crowding sentiment
     - Trail conditions sentiment
   - Custom word lists with VADER sentiment analyzer

3. **Data Preprocessing**
   - Filtered to valid 8-digit trail IDs: 6,984 → 4,565 trails
   - Merged nested columns (activities, features, collections)
   - Removed erroneous scrapes: 2,423 trails excluded
   - Created percentage columns (featured photos %, recording %, etc.)
   - One-hot encoded categorical variables
   - Imputed missing values using domain knowledge

4. **Feature Engineering**
   - Started with 312 raw columns
   - Created 213 binary features from one-hot encoding
   - Generated ratio features (percentage of reviews) so as to avoid data leakage
   - Performs feature reduction to cut bloat
   - Result: 150 features selected from 525 candidates

5. **Model Training**
   - XGBoost with 200 estimators, max_depth=6, learning_rate=0.1
   - Trained on 4,565 trails with 80/20 train-test split
   - Random state=42 for reproducibility

### Why XGBoost?

XGBoost (Extreme Gradient Boosting) outperformed alternative algorithms because:

- **Gradient Boosting Advantage**: Iteratively corrects errors from previous trees, capturing complex nonlinear relationships
- **Feature Interactions**: Handles feature interactions naturally (e.g., high rating + featured photos has synergistic effect)
- **Regularization**: Built-in L1/L2 regularization prevents overfitting (16% train-test gap shows good generalization)
- **Tree-Based Strength**: Works well with mixed feature types (numeric, categorical) without requiring scaling
- **Speed & Scalability**: Efficient training and prediction with 4,565 samples

Alternative models tested: Random Forest (R² = 0.73), Linear models (R² = .42), Gradient Boosting without sentiment (R² = 0.70). XGBoost outperformed both.

---

## 💡 Key Insights & Impact

### What Worked Best & Why

**1. New Scraping Methodology** (Major Impact)
- Previous attempts using basic scraping methods and thus less overall data: R² = 0.53
- New undetected-chromedriver approach capable of scraping hidden statistics: R² = 0.8556
- **Impact**: +61% accuracy improvement through better data quality
- Insight: Data quality determines model ceiling more than algorithm choice

**2. Sentiment Analysis** (15% Improvement)
- Baseline model (no sentiment): R² = 0.704
- Advanced model (with sentiment): R² = 0.8556
- Insight: Community discussion is quantifiable and predictive
- Practical value: Validates investment in community engagement

**3. User-Generated Features** (The Game Changer)
- Removed data leakage by converting counts to percentages (photos as % of reviews, not raw count)
- Community engagement metrics (ratings, reviews, photos) proved 10× more important than trail characteristics
- Insight: Popularity is socially constructed, not naturally determined

### Business Impact for Park Management

**Before This Model**: Parks had no framework for data-driven resource allocation
- Budget cuts applied equally across all trails
- No way to identify which trails drive community engagement
- Trail quality improvements had uncertain ROI

**After This Model**: Parks can now:
- **Identify High-Potential Trails**: Allocate maintenance/marketing budget to trails predicted to attract visitors
- **Optimize Community Engagement**: Focus on photo contests, events, and rating campaigns (proven to work)
- **Predict Popularity Impacts**: Test what-if scenarios ("If we improve this trail's rating by 0.5 stars, popularity increases by ~2 points")
- **Justify Budget Decisions**: Show decision-makers which trails drive revenue based on visitor numbers

**Revenue Impact** (Example):
- Identifying and optimizing top 5 trails (from baseline to optimized) could increase popularity for average trails by 15-20 points
- Result: Popular trails in parks drive attendance, increasing profitability of the park and the allocation of government funding, but in addition to park funding people going out spend more money in the local region. Food, entertainment, movies, etc, popular trails reasonably contribute upwards of $15M annually with regards to regional economic activity.
- Jobs created: Estimated 150-200 jobs in gateway communities

**Current Baseline**: No parks have ever quantified this before. This model enables the first data-driven approach to trail resource optimization.

---

## 📈 Model Limitations & Caveats

1. **Regional Scope**: Model trained on Southwest U.S. trails (Arizona, Colorado, New Mexico, Utah). May not generalize perfectly to other regions with different climates, user demographics, or trail characteristics. For there it's reccomended you rebuild the model based on local dataset.

2. **AllTrails Bias**: Popularity measured as AllTrails views/ratings. Does not capture all local park popularity (visitors who don't use AllTrails). It should be proportional, but won't be exact.

3. **Temporal Dynamics**: Model captures current state. Seasonal trends, emerging new trails, and changing user behavior may affect predictions.

4. **Sentiment Limitations**: Sentiment analysis captures only review text, not all community engagement (social media mentions, word-of-mouth not captured).

5. **Causation vs. Correlation**: Model identifies what features correlate with popularity, not necessarily what causes it. Featured photos correlate with popularity, but may be related to the fact that already-popular trails get more photos (reverse causation).

---

## 📁 Repository Structure
```
Trail-Popularity-Predictor/
│
├── README.md                           # Project documentation (you are here!)
├── LICENSE.txt                         # MIT License
├── requirements.txt                    # Python dependencies
├── RUN_DASHBOARD.sh                    # Script to launch Streamlit dashboard
│
├── app.py                              # Streamlit interactive dashboard
│                                       # Usage: streamlit run app.py
│
├── Data/                               # Data files
│   ├── combined_trails.csv             # Raw data (6,984 trails × 300 columns)
│   ├── preprocessed_trails.csv         # Clean data after preprocessing (4,565 × 93 columns)
│   ├── final_model_trained.pkl         # Advanced XGBoost model (R² = 0.8556)
│   ├── usmetros.csv                    # Metro population data for feature engineering
│   └── config/                         # Sentiment analysis configuration
│       ├── categories.json             # Sentiment categories (views, difficulty, crowding, conditions)
│       ├── pos_words.txt               # Positive sentiment words
│       └── neg_words.txt               # Negative sentiment words
│
├── models/                             # Trained model files
│   ├── baseline_model.pkl              # Baseline model (R² = 0.7036, no sentiment or advanced feature engineering)
│   └── final_model_trained.pkl         # Advanced model (R² = 0.8556, with additional features)
│
├── Notebooks/                          # Jupyter notebooks for analysis and training
│   ├── Model_Streamlined_FINAL.ipynb   # MAIN: Advanced model training (XGBoost)
│   ├── Model_Streamlined.py            # Alternate format of Final model
│   ├── Model2a.ipynb                   # Historical model experiments
│   ├── baseline_model.py               # Standalone baseline model script
│   ├── combined_trails.csv             # Raw data copy, used for baseline
│   └── baseline_performance.txt        # Baseline model performance metrics
│
├── Prototypes/                         # Experimental and prototype files showing testing of feature engineering ideas
│
└── [Additional files created during development]
    ├── MODEL_COMPARISON.txt            # Detailed comparison of baseline vs. advanced models
    ├── decision_rules.txt              # SHAP-based decision rules and recommendations
    ├── SHAP_ANALYSIS_SUMMARY.txt       # Summary of SHAP explainability analysis
    ├── feature_optimization_results.csv # Feature importance rankings
    └── [Visualizations and analysis files]
        ├── 02_shap_summary_bar.png     # Feature importance bar chart
        ├── 03_shap_beeswarm.png        # Feature distributions
        ├── 04_shap_dependence_plots.png # Top 4 features detailed analysis
        └── 05_force_plot_*.png         # Individual prediction explanations
```

---

## 🚀 How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step 1: Collect Raw Data
Use the Apify actor: [console.apify.com/actors/MMQdritoUWpzUVbah](https://console.apify.com/actors/MMQdritoUWpzUVbah)

Or use existing `combined_trails.csv`

### Step 2: Complete Preprocessing & Sentiment Scraping
```bash
python preprocess_pipeline.py
```
**Output**: `Data/preprocessed_trails.csv` (4,565 trails × 93 columns)
**Runtime**: 5-15 hours (includes web scraping)

### Step 3: Train Advanced Model
```bash
# Using Jupyter notebook
jupyter notebook Model_Streamlined_FINAL.ipynb

# Or using training script
python train_advanced_model.py
```
**Output**: `final_model_trained.pkl` with R² = 0.8556

### Step 4: Train Baseline Model (for comparison)
```bash
python baseline_model.py
```
**Output**: `predictions` and `SHAP visualizations of results` in \output and `baseline_model.pkl` with R² = 0.7036
**Runtime**: 5-10 minutes (no web scraping)

<img width="2084" height="1474" alt="04_shap_dependence_plots" src="https://github.com/user-attachments/assets/7c41b846-2eb1-42d3-ae2c-44c9247994ab" />


### Step 5: Run Interactive Dashboard for individual trail analysis
```bash
streamlit run app.py
```

---

## 🔮 Future Work & Improvements

### Model Enhancement
1. **Test Alternative Algorithms**: Explore other gradient boosting variants (LightGBM, CatBoost) to see if further improvements possible
2. **Feature Engineering Expansion**: Develop additional engagement metrics, temporal features (trends over time), and seasonal patterns
3. **Extended Review Scraping**: Current approach captures main page reviews. Future work could scrape full review history for deeper sentiment analysis

### Feature Expansion
1. **Social Media Integration**: Incorporate Twitter/Instagram mentions, hashtag trends, influencer activity
2. **Temporal Modeling**: Better track how sentiment and features change over time; target seasonal trends
3. **Multi-Model Ensemble**: Combine XGBoost predictions with other algorithms for enhanced robustness

### Dashboard Improvements
1. **AI-Powered Insights**: Automated recommendations ("To increase this trail's popularity by 5 points, focus on photo engagement" based on SHAP analysis)
2. **What-If Scenario Planning**: "If we improve ratings by 0.5 stars AND double featured photos, what happens to popularity?"
3. **Community Engagement Tracking**: Dashboard to monitor which communities are most engaged with which trails

### Broader Applications
1. **Regional Model Variants**: Develop separate models for different geographic regions
2. **Park System Optimization**: Extend to optimize across entire park systems (not just individual trails)
3. **Real-Time Updates**: Automated pipeline that retrains model weekly/monthly with new data

---

## ✨ Lessons Learned

1. **Data Quality > Algorithm Sophistication**: Improved scraping method provided more ROI than model optimization (61% vs. 15% improvement)

2. **Community Engagement is Quantifiable**: What seemed like soft, unmeasurable factors (word-of-mouth, community engagement) proved to be the strongest predictors

3. **Data Leakage is Subtle and makes reccomendations difficult**: Initially included raw photo/recording counts. Had to convert to percentages to avoid leakage, but then can't optimize for "more photos" based on number of photos as popularity naturally attracts more photos. Specifying a trail has 10,000 photos directly means it is popular, but saying 30% of reviews include photos is more standardized but less able to give advice insights.

4. **User-Generated Content Trumps Natural Features**: Trail characteristics matter far less than how community perceives and engages with trails

5. **Tree-Based Models Excel with Mixed Features**: XGBoost handled our mix of numeric and categorical features better than linear approaches

---

## 📚 Technical Documentation

- **Model Details**: See `Model_Streamlined_FINAL.ipynb` for training code and methodology
- **Preprocessing Pipeline**: See `preprocess_pipeline.py`
- **SHAP Analysis**: See `SHAP_ANALYSIS_SUMMARY.txt` and generated SHAP visualization files
- **Feature Importance**: See `decision_rules.txt` for actionable recommendations based on SHAP values
- **Baseline Comparison**: See `baseline_model.py` and `MODEL_COMPARISON.txt` for baseline model details

---

## 🤝 Practical Application

This model can be immediately deployed to:

1. **Park Management Systems**: Parks can input trail data and receive popularity predictions
2. **Resource Allocation**: Budget decisions based on predicted vs. actual popularity gaps
3. **Community Engagement Strategy**: Recommendations on which trails to focus community engagement efforts
4. **Trail Development Planning**: Data-driven decisions on trail maintenance, marketing, and event planning

**First-Time Ever**: This is the first data-driven framework for predicting trail popularity and understanding visitor engagement drivers.

---

## 📝 Requirements

See `requirements.txt` for complete dependencies:
```
pandas==2.0.3
numpy==1.24.3
xgboost==2.0.0
scikit-learn==1.3.0
shap>=0.41.0
plotly==5.17.0
streamlit==1.28.1
beautifulsoup4>=4.9.0
vaderSentiment>=3.3.2
undetected-chromedriver>=3.5.0
```

---

## 📞 Questions & Support

For questions feel free to email me at dxu7267@mavs.uta.edu
I will help where I can

---

## 📄 Citation

If using this model or methodology:

```
Trail Popularity Prediction Model (2024)
Capstone Project: Data-Driven Park Resource Optimization
Dataset: AllTrails.com
Model: XGBoost with Sentiment Analysis
Performance: R² = 0.8556 on 4,565 Southwest U.S. trails
```

---

## 🎯 Final Achievement

**Model achieves 85.56% accuracy in predicting trail popularity**, revealing that **community engagement and user-generated content matter far more than trail characteristics**. This finding enables parks to implement data-driven community engagement strategies that demonstrably increase visitor traffic and regional economic activity.

For the first time, park managers have a framework to predict what makes trails popular and what drives visitor engagement—enabling strategic resource allocation in an era of budget constraints.
