"""
Trail Popularity Prediction Dashboard - FINAL INTERACTIVE VERSION
With all controls and features using corrected defaults from Force Plot 5
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from math import radians, sin, cos, sqrt, atan2
from xgboost import XGBRegressor
import plotly.graph_objects as go
import pickle
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Trail Popularity Predictor", page_icon="🚶‍♂️", layout="wide", initial_sidebar_state="collapsed")

# ============================================================================
# TOP 150 FEATURES (EXACT FROM NOTEBOOK)
# ============================================================================

TOP_150_FEATURES = [
    'isClosed', 'numFeaturedPhotos_pct', 'collections_trending', 'avgRating',
    'collections_with_photos_pct', 'numPOIs', 'numRecordings_pct', 'activities_walking',
    'kidFriendly', 'activities_backpacking', 'elevationGainMeters', 'estimatedTime',
    'areaType_W', 'lengthMeters', 'cityName_Indian Hills', 'dist_to_nearest_metro_km',
    'stateName_Colorado', 'collections_hidden-gems', 'hasAlerts', 'sentiment_views',
    'cityName_Peoria', 'sentence_count_all', 'areaType_C', 'activities_off-road-driving',
    'cityName_Littleton', 'sentence_count_views_pct', 'cityName_Idaho Springs',
    'cityName_Cimarron', 'activities_mountain-biking', 'cityName_Bailey',
    'cityName_Highlands Ranch', 'features_dogs-no', 'cityName_Supai', 'cityName_Payson',
    'sentence_count_crowding_pct', 'highestPoint', 'cityName_Page', 'collections_kids',
    'features_dogs-leash', 'routeType_encoded', 'cityName_Salt Lake City', 'elevationMeters',
    'features_historic-site', 'nearest_metro_population', 'difficultyRating',
    'features_partially-paved', 'sum_population_3_nearest_metros', 'features_cave',
    'activities_rock-climbing', 'features_wildlife', 'features_wild-flowers',
    'activities_paddle-sports', 'stateName_Utah', 'activities_scenic-driving',
    'cityName_Quartzsite', 'cityName_Kanab', 'cityName_Park City', 'cityName_Bluff',
    'cityName_Overton', 'collections_count', 'cityName_Orderville', 'features_paved',
    'numPhotos_pct', 'activities_trail-running', 'cityName_Glorieta', 'strollerFriendly',
    'activities_birding', 'areaType_N', 'areaType_F', 'cityName_Howell', 'cityName_Raton',
    'stateName_Wyoming', 'sentiment_crowding', 'activities_camping', 'cityName_Fredonia',
    'dogFriendly', 'cityName_Red River', 'activities_fishing', 'cityName_Logan',
    'cityName_Moab', 'cityName_Mayer', 'adaAccessible', 'cityName_Dutch John',
    'sentence_count_difficulty_pct', 'cityName_Santa Fe', 'cityName_Meadow', 'cityName_Loma',
    'sentiment_all', 'areaType_M', 'cityName_Hurricane', 'sentence_count_conditions_pct',
    'cityName_Spring City', 'areaType_S', 'features_lake', 'sentiment_conditions',
    'stateName_New Mexico', 'cityName_Willard', 'cityName_Placitas', 'cityName_Kayenta',
    'activities_horseback-riding', 'cityName_Littlefield', 'features_rails-trails',
    'cityName_Englewood', 'areaType_Unknown', 'cityName_Dolan Springs', 'features_city-walk',
    'sentiment_difficulty', 'cityName_Parks', 'cityName_Bountiful', 'features_forest',
    'cityName_Corinne', 'cityName_Flagstaff', 'cityName_Tonopah', 'cityName_Blanding',
    'features_views', 'areaType_H', 'cityName_Hanksville', 'cityName_Saratoga Springs',
    'cityName_Castle Valley', 'cityName_Albuquerque', 'cityName_Eden', 'cityName_Pine Valley',
    'cityName_Enterprise', 'activities_snowshoeing', 'features_river', 'cityName_Provo',
    'features_event', 'cityName_Junction', 'cityName_Rio Rancho', 'cityName_Skull Valley',
    'features_dogs', 'cityName_Saint Johns', 'cityName_Chimayo', 'cityName_Rush Valley',
    'cityName_Roy', 'cityName_Ogden', 'cityName_Kingston', 'cityName_Fayette',
    'cityName_Prescott', 'cityName_New Harmony', 'cityName_Cuba', 'cityName_Springerville',
    'cityName_Emery', 'cityName_Vernal', 'cityName_Smithfield', 'cityName_Denver',
    'cityName_Munds Park', 'cityName_Altonah', 'features_waterfall', 'cityName_Taylorsville',
]

# ============================================================================
# LOAD PRE-TRAINED MODEL
# ============================================================================

@st.cache_resource
def load_pretrained_model():
    """Load the pre-trained model saved from notebook"""
    model_path = './Data/final_model_trained.pkl'
    
    if not os.path.exists(model_path):
        st.error(f"ERROR: Model file not found at {model_path}")
        st.write("You need to save the model from your notebook first!")
        st.write("""
        In your notebook, after training, run:
        
        ```python
        import pickle
        with open('./Data/final_model_trained.pkl', 'wb') as f:
            pickle.dump(final_model, f)
        ```
        """)
        st.stop()
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

try:
    with st.spinner('Loading pre-trained model...'):
        model = load_pretrained_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ============================================================================
# CORRECTED FORCE PLOT 5 DEFAULTS (WITH ALL 150 FEATURES)
# ============================================================================

FORCE_PLOT_5_DEFAULTS = {
    'isClosed': 0,
    'numFeaturedPhotos_pct': 0.0,
    'collections_trending': 0,
    'avgRating': 4.1,
    'collections_with_photos_pct': 0.6666666666666666,
    'numPOIs': 6,
    'numRecordings_pct': 2.2133333333333334,
    'activities_walking': 0,
    'kidFriendly': 1,
    'activities_backpacking': 0,
    'elevationGainMeters': 27.7368,
    'estimatedTime': 1.0,
    'areaType_W': 0,
    'lengthMeters': 6598.294,
    'cityName_Indian Hills': 0,
    'dist_to_nearest_metro_km': 136.95010714170868,
    'stateName_Colorado': 0,
    'collections_hidden-gems': 0,
    'hasAlerts': 0,
    'sentiment_views': 0.6482,
    'cityName_Peoria': 0,
    'sentence_count_all': 20,
    'areaType_C': 0,
    'activities_off-road-driving': 0,
    'cityName_Littleton': 0,
    'sentence_count_views_pct': 0.3,
    'cityName_Idaho Springs': 0,
    'cityName_Cimarron': 0,
    'activities_mountain-biking': 0,
    'cityName_Bailey': 0,
    'cityName_Highlands Ranch': 0,
    'features_dogs-no': 0,
    'cityName_Supai': 0,
    'cityName_Payson': 0,
    'sentence_count_crowding_pct': 0.0,
    'highestPoint': 1454,
    'cityName_Page': 0,
    'collections_kids': 0,
    'features_dogs-leash': 1,
    'routeType_encoded': 2,
    'cityName_Salt Lake City': 0,
    'elevationMeters': 1453,
    'features_historic-site': 1,
    'nearest_metro_population': 202452,
    'difficultyRating': 3,
    'features_partially-paved': 0,
    'sum_population_3_nearest_metros': 570606,
    'features_cave': 0,
    'activities_rock-climbing': 0,
    'features_wildlife': 0,
    'features_wild-flowers': 0,
    'activities_paddle-sports': 0,
    'stateName_Utah': 1,
    'activities_scenic-driving': 0,
    'cityName_Quartzsite': 0,
    'cityName_Kanab': 1,
    'cityName_Park City': 0,
    'cityName_Bluff': 0,
    'cityName_Overton': 0,
    'collections_count': 3,
    'cityName_Orderville': 0,
    'features_paved': 0,
    'numPhotos_pct': 2.36,
    'activities_trail-running': 0,
    'cityName_Glorieta': 0,
    'strollerFriendly': 0,
    'activities_birding': 0,
    'areaType_N': 0,
    'areaType_F': 0,
    'cityName_Howell': 0,
    'cityName_Raton': 0,
    'stateName_Wyoming': 0,
    'sentiment_crowding': 0.0,
    'activities_camping': 0,
    'cityName_Fredonia': 0,
    'dogFriendly': 1,
    'cityName_Red River': 0,
    'activities_fishing': 0,
    'cityName_Logan': 0,
    'cityName_Moab': 0,
    'cityName_Mayer': 0,
    'adaAccessible': 0,
    'cityName_Dutch John': 0,
    'sentence_count_difficulty_pct': 0.0,
    'cityName_Santa Fe': 0,
    'cityName_Meadow': 0,
    'cityName_Loma': 0,
    'sentiment_all': 0.1978,
    'areaType_M': 1,
    'cityName_Hurricane': 0,
    'sentence_count_conditions_pct': 0.1,
    'cityName_Spring City': 0,
    'areaType_S': 0,
    'features_lake': 0,
    'sentiment_conditions': -0.2912,
    'stateName_New Mexico': 0,
    'cityName_Willard': 0,
    'cityName_Placitas': 0,
    'cityName_Kayenta': 0,
    'activities_horseback-riding': 0,
    'cityName_Littlefield': 0,
    'features_rails-trails': 0,
    'cityName_Englewood': 0,
    'areaType_Unknown': 0,
    'cityName_Dolan Springs': 0,
    'features_city-walk': 0,
    'sentiment_difficulty': 0.0,
    'cityName_Parks': 0,
    'cityName_Bountiful': 0,
    'features_forest': 0,
    'cityName_Corinne': 0,
    'cityName_Flagstaff': 0,
    'cityName_Tonopah': 0,
    'cityName_Blanding': 0,
    'features_views': 1,
    'areaType_H': 0,
    'cityName_Hanksville': 0,
    'cityName_Saratoga Springs': 0,
    'cityName_Castle Valley': 0,
    'cityName_Albuquerque': 0,
    'cityName_Eden': 0,
    'cityName_Pine Valley': 0,
    'cityName_Enterprise': 0,
    'activities_snowshoeing': 0,
    'features_river': 1,
    'cityName_Provo': 0,
    'features_event': 0,
    'cityName_Junction': 0,
    'cityName_Rio Rancho': 0,
    'cityName_Skull Valley': 0,
    'features_dogs': 0,
    'cityName_Saint Johns': 0,
    'cityName_Chimayo': 0,
    'cityName_Rush Valley': 0,
    'cityName_Roy': 0,
    'cityName_Ogden': 0,
    'cityName_Kingston': 0,
    'cityName_Fayette': 0,
    'cityName_Prescott': 0,
    'cityName_New Harmony': 0,
    'cityName_Cuba': 0,
    'cityName_Springerville': 0,
    'cityName_Emery': 0,
    'cityName_Vernal': 0,
    'cityName_Smithfield': 0,
    'cityName_Denver': 0,
    'cityName_Munds Park': 0,
    'cityName_Altonah': 0,
    'features_waterfall': 0,
    'cityName_Taylorsville': 0,
}

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if 'show_debug' not in st.session_state:
    st.session_state.show_debug = False

# ============================================================================
# HEADER
# ============================================================================

st.markdown("<h1 style='text-align: center; color: #1F4E78;'>🥾 Trail Popularity Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Predict and improve trail popularity with data-driven insights</p>", unsafe_allow_html=True)

st.divider()

# ============================================================================
# MAIN INPUTS
# ============================================================================

input_col, results_col = st.columns([1, 1.2], gap="large")

with input_col:
    st.subheader("Trail Information")
    
    # Key metrics
    st.markdown("**Rating & Engagement**")
    avg_rating = st.slider(
        "Average Rating (stars):",
        min_value=1.0,
        max_value=5.0,
        value=FORCE_PLOT_5_DEFAULTS['avgRating'],
        step=0.1,
        key="rating"
    )
    
    st.markdown("**Length & Elevation**")
    col1, col2 = st.columns(2)
    with col1:
        length_meters = st.slider(
            "Length (meters):",
            min_value=100,
            max_value=25000,
            value=int(FORCE_PLOT_5_DEFAULTS['lengthMeters']),
            step=10,
            key="length"
        )
    with col2:
        elevation_gain = st.slider(
            "Elevation Gain (m):",
            min_value=0,
            max_value=2500,
            value=int(FORCE_PLOT_5_DEFAULTS['elevationGainMeters']),
            step=10,
            key="elevation"
        )
    
    st.markdown("**Content & Features**")
    st.write("**Featured Photos**")
    col1, col2 = st.columns(2)
    with col1:
        featured_count = st.slider(
            "Featured Count:",
            min_value=0,
            max_value=10,
            value=int(FORCE_PLOT_5_DEFAULTS['numFeaturedPhotos_pct'] * 100),
            step=1,
            key="featured_count"
        )
    with col2:
        total_photos = st.slider(
            "Total Photos:",
            min_value=1,
            max_value=7000,
            value=200,
            step=10,
            key="total_photos"
        )
    
    featured_photos_pct = (featured_count / total_photos * 100) if total_photos > 0 else 0
    
    st.write("**Points of Interest**")
    num_pois = st.slider(
        "Points of Interest:",
        min_value=0,
        max_value=50,
        value=int(FORCE_PLOT_5_DEFAULTS['numPOIs']),
        key="pois"
    )
    
    st.markdown("**Community Sentiment**")
    col1, col2 = st.columns(2)
    with col1:
        sentiment_views = st.slider(
            "Sentiment (Views):",
            min_value=-1.0,
            max_value=1.0,
            value=FORCE_PLOT_5_DEFAULTS['sentiment_views'],
            step=0.01,
            key="sentiment_views"
        )
    with col2:
        sentiment_all = st.slider(
            "Sentiment (All):",
            min_value=-1.0,
            max_value=1.0,
            value=FORCE_PLOT_5_DEFAULTS['sentiment_all'],
            step=0.01,
            key="sentiment_all"
        )
    
    st.markdown("**Amenities & Accessibility**")
    col1, col2 = st.columns(2)
    with col1:
        kid_friendly = st.checkbox("Kid Friendly", value=bool(FORCE_PLOT_5_DEFAULTS['kidFriendly']), key="kid_friendly")
        dog_friendly = st.checkbox("Dog Friendly", value=bool(FORCE_PLOT_5_DEFAULTS['dogFriendly']), key="dog_friendly")
        water_features = st.checkbox("Water Features", value=bool(FORCE_PLOT_5_DEFAULTS.get('features_lake', 0)), key="water_features")
    with col2:
        ada_accessible = st.checkbox("ADA Accessible", value=bool(FORCE_PLOT_5_DEFAULTS['adaAccessible']), key="ada")
        has_alerts = st.checkbox("Has Alerts", value=bool(FORCE_PLOT_5_DEFAULTS['hasAlerts']), key="alerts")
        rock_climbing = st.checkbox("Rock Climbing", value=bool(FORCE_PLOT_5_DEFAULTS.get('activities_rock-climbing', 0)), key="rock_climbing")
    
    with st.expander("⚙️ Advanced Options", expanded=False):
        st.write("**Trail Type**")
        route_type = st.selectbox(
            "Route Type:",
            options=['Point (1)', 'Out-and-back (2)', 'Loop (3)'],
            index=1,
            key="route_type"
        )
        route_encoded = float(route_type[route_type.find('(')+1])
        
        st.write("**Location**")
        state = st.selectbox(
            "State:",
            options=['Utah', 'Colorado', 'Arizona', 'New Mexico', 'Wyoming'],
            index=0,
            key="state"
        )
        
        metro_distance = st.slider(
            "Distance to Metro (km):",
            min_value=0.0,
            max_value=200.0,
            value=FORCE_PLOT_5_DEFAULTS['dist_to_nearest_metro_km'],
            step=1.0,
            key="metro_dist"
        )
        
        st.write("**Collections & Community**")
        col1, col2 = st.columns(2)
        with col1:
            collections_trending = st.slider(
                "Trending Collections:",
                min_value=0,
                max_value=10,
                value=int(FORCE_PLOT_5_DEFAULTS['collections_trending']),
                key="collections_trending"
            )
        with col2:
            collections_count = st.slider(
                "Total Collections:",
                min_value=0,
                max_value=50,
                value=int(FORCE_PLOT_5_DEFAULTS['collections_count']),
                key="collections_count"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            difficulty_rating = st.slider(
                "Difficulty Rating:",
                min_value=0,
                max_value=5,
                value=int(FORCE_PLOT_5_DEFAULTS['difficultyRating']),
                key="difficulty"
            )
        with col2:
            sentence_count_all = st.slider(
                "Avg Review Length (sentences):",
                min_value=1,
                max_value=50,
                value=int(FORCE_PLOT_5_DEFAULTS['sentence_count_all']),
                step=1,
                key="sentence_count"
            )

# ============================================================================
# BUILD FEATURE VECTOR & PREDICT
# ============================================================================

# Start with defaults
feature_values = FORCE_PLOT_5_DEFAULTS.copy()

# Update with user inputs
feature_values['avgRating'] = avg_rating
feature_values['lengthMeters'] = float(length_meters)
feature_values['elevationGainMeters'] = float(elevation_gain)
feature_values['numFeaturedPhotos_pct'] = featured_photos_pct
feature_values['numPOIs'] = num_pois
feature_values['sentiment_views'] = sentiment_views
feature_values['sentiment_all'] = sentiment_all
feature_values['kidFriendly'] = int(kid_friendly)

# Handle dog friendly logic
if dog_friendly:
    feature_values['dogFriendly'] = 1
    feature_values['features_dogs'] = 1
    feature_values['features_dogs-leash'] = 1
    feature_values['features_dogs-no'] = 0
else:
    feature_values['dogFriendly'] = 0
    feature_values['features_dogs'] = 0
    feature_values['features_dogs-leash'] = 0
    feature_values['features_dogs-no'] = 1

feature_values['adaAccessible'] = int(ada_accessible)
feature_values['hasAlerts'] = int(has_alerts)

# Handle water features logic - all three water types together
if water_features:
    feature_values['features_lake'] = 1
    feature_values['features_river'] = 1
    feature_values['features_waterfall'] = 1
else:
    feature_values['features_lake'] = 0
    feature_values['features_river'] = 0
    feature_values['features_waterfall'] = 0

feature_values['activities_rock-climbing'] = int(rock_climbing)
feature_values['dist_to_nearest_metro_km'] = metro_distance
feature_values['collections_trending'] = collections_trending
feature_values['collections_count'] = collections_count
feature_values['difficultyRating'] = difficulty_rating
feature_values['sentence_count_all'] = sentence_count_all
feature_values['routeType_encoded'] = route_encoded

# Handle state one-hot encoding
state_mapping = {
    'Utah': 'stateName_Utah',
    'Colorado': 'stateName_Colorado',
    'Arizona': 'stateName_Arizona',
    'New Mexico': 'stateName_New Mexico',
    'Wyoming': 'stateName_Wyoming'
}
for state_name, col_name in state_mapping.items():
    if col_name in feature_values:
        feature_values[col_name] = 1 if state == state_name else 0

# Create feature vector in correct order
feature_vector = np.array([[feature_values.get(f, 0) for f in TOP_150_FEATURES]])

# Make prediction
try:
    prediction = model.predict(feature_vector)[0]
    prediction = min(100, max(0, prediction))
except Exception as e:
    st.error(f"Error: {e}")
    prediction = 0

# ============================================================================
# RESULTS DISPLAY
# ============================================================================

with results_col:
    st.subheader("Predicted Popularity")
    
    # Determine traffic level
    if prediction < 40:
        traffic_level, color = "Low", "#8B5CF6"
    elif prediction < 70:
        traffic_level, color = "Moderate", "#A78BFA"
    else:
        traffic_level, color = "High", "#7C3AED"
    
    # Display gauge
    fig = go.Figure(data=[go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={'text': "Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 33], 'color': "#E9D5FF"},
                {'range': [33, 66], 'color': "#D8B4FE"},
                {'range': [66, 100], 'color': "#C084FC"}
            ]
        }
    )])
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Score", f"{prediction:.1f}")
    with col2:
        st.metric("Level", traffic_level)
    
    st.divider()
    
    st.subheader("💡 Recommendations")
    
    recommendations = []
    if avg_rating < 4.2:
        recommendations.append("**1. Improve Rating** - Better maintenance & trail conditions")
    if avg_rating > 4.8:
        recommendations.append("**1. Seek more Ratings** - Ironically rating is too high, 4.9 and 5.0 tend to be avoided. More ratings help.")
    if featured_photos_pct == 0:
        recommendations.append("**2. Seek Featured Photos** - High-quality images which become boost popularity")
    if num_pois < 5:
        recommendations.append("**3. Add POIs** - Landmarks and points of interest")
    if total_photos < 1000:
        recommendations.append("**4. Add more Photos in general** Photo competitions or photo driven events")
    
    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("✅ Trail is optimized!")
    
    if st.button("🔄 Reset to Defaults", use_container_width=True):
        st.rerun()

st.divider()
st.markdown("<div style='text-align: center; color: #999; font-size: 12px;'>Trail Popularity Predictor | XGBoost (R² = 0.8556)</div>", unsafe_allow_html=True)
