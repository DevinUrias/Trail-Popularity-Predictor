"""
================================================================================
TRAIL POPULARITY PREDICTION - COMPLETE DATA PREPROCESSING PIPELINE
================================================================================

This script combines two critical preprocessing steps:
1. SENTIMENT SCRAPING: Extract reviews from AllTrails and score sentiment
2. DATA CLEANING: Clean, transform, and prepare data for model training

PIPELINE:
    combined_trails.csv (raw data)
         ↓
    [Step 1: Sentiment Scraping - IDs_to_SENT]
         ↓
    trails_sentiment.csv (with sentiment scores)
         ↓
    [Step 2: Data Preprocessing - Preprocessing]
         ↓
    preprocessed_trails.csv (final clean data, ready for model)

REQUIREMENTS:
    pip install undetected-chromedriver pandas beautifulsoup4 vaderSentiment

CONFIG FILES NEEDED (see setup_config_files() function):
    config/categories.json    — sentiment categories with regex patterns
    config/pos_words.txt      — positive sentiment words
    config/neg_words.txt      — negative sentiment words

USAGE:
    python preprocess_pipeline.py
    
This will:
1. Scrape sentiment from AllTrails reviews
2. Clean and transform the data
3. Output: Data/preprocessed_trails.csv (ready for model training)

================================================================================
"""

import json
import re
import time
import random
import pandas as pd
import numpy as np
import ast
import os
from pathlib import Path
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict, Any
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Sentiment Scraping Config
INPUT_CSV_RAW = "./Data/combined_trails.csv"
INPUT_CSV_SENTIMENT = "./trails_sentiment.csv"
OUTPUT_CSV_SENTIMENT = "./trails_sentiment.csv"
OUTPUT_DIR = "./Data"

# Scraper Settings
BASE_URL = "https://www.alltrails.com/trail/"
DELAY_MIN = 4.0
DELAY_MAX = 9.0
PAGE_LOAD_WAIT = 6
NO_MATCH_VALUE = None
VERBOSE = True

# ============================================================================
# STEP 0: SETUP CONFIG FILES (Run once before scraping)
# ============================================================================

def setup_config_files():
    """
    Create the config/ directory and populate with default sentiment categories,
    positive words, and negative words. Modify these to customize sentiment scoring.
    """
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Create categories.json if it doesn't exist
    categories_file = config_dir / "categories.json"
    if not categories_file.exists():
        categories = {
            "views": [r"view", r"vista", r"scenery", r"scenic"],
            "difficulty": [r"difficult", r"hard", r"challenging", r"steep", r"steep", r"grind"],
            "crowding": [r"crowded", r"busy", r"people", r"traffic"],
            "conditions": [r"muddy", r"wet", r"snow", r"ice", r"rocky", r"erosion", r"trail condition"]
        }
        with open(categories_file, 'w') as f:
            json.dump(categories, f, indent=2)
        print(f"✓ Created {categories_file}")
    
    # Create pos_words.txt if it doesn't exist
    pos_words_file = config_dir / "pos_words.txt"
    if not pos_words_file.exists():
        pos_words = """beautiful
wonderful
amazing
excellent
fantastic
great
good
awesome
perfect
stunning
gorgeous
lovely
nice
enjoyable
fun
rewarding
worth it
highly recommend
must do
easy
well maintained"""
        with open(pos_words_file, 'w') as f:
            f.write(pos_words)
        print(f"✓ Created {pos_words_file}")
    
    # Create neg_words.txt if it doesn't exist
    neg_words_file = config_dir / "neg_words.txt"
    if not neg_words_file.exists():
        neg_words = """bad
terrible
awful
horrible
disappointing
dangerous
crowded
busy
muddy
wet
slippery
steep
exhausting
not worth
waste of time
boring
overhyped
misleading
poorly maintained
difficult
hard"""
        with open(neg_words_file, 'w') as f:
            f.write(neg_words)
        print(f"✓ Created {neg_words_file}")

# ============================================================================
# STEP 1: SENTIMENT SCRAPING (IDs_to_SENT)
# ============================================================================

def load_config():
    """Load sentiment categories and word lists from config files."""
    config_dir = Path("config")
    
    # Load categories.json
    categories_raw = json.loads((config_dir / "categories.json").read_text(encoding="utf-8"))
    categories = {}
    for cat_name, patterns in categories_raw.items():
        compiled = []
        for pat in patterns:
            try:
                compiled.append(re.compile(pat, re.IGNORECASE))
            except re.error as e:
                print(f"  Warning: invalid regex in '{cat_name}': {pat!r} — {e}")
        categories[cat_name] = compiled
    
    # Load word lists
    def load_wordlist(filename):
        lines = (config_dir / filename).read_text(encoding="utf-8").splitlines()
        return {ln.strip().lower() for ln in lines if ln.strip() and not ln.startswith("#")}
    
    pos_words = load_wordlist("pos_words.txt")
    neg_words = load_wordlist("neg_words.txt")
    
    print(f"Config loaded: {len(categories)} categories, "
          f"{len(pos_words)} positive words, {len(neg_words)} negative words")
    return categories, pos_words, neg_words


def make_driver():
    """Initialize undetected Chrome driver."""
    try:
        import undetected_chromedriver as uc
        options = uc.ChromeOptions()
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        return uc.Chrome(options=options, headless=False)
    except Exception as e:
        print(f"Error: Could not start Chrome driver. Make sure undetected-chromedriver is installed.")
        print(f"  pip install undetected-chromedriver")
        raise


def fetch_html(driver, url, wait_seconds=PAGE_LOAD_WAIT):
    """Fetch HTML from URL using Selenium driver."""
    driver.get(url)
    time.sleep(wait_seconds)
    return driver.page_source


def is_blocked(html):
    """Check if page is blocked by CloudFlare or other protection."""
    html_lower = html.lower()
    if any(s in html_lower for s in ["alltrails", "trailhead", "difficulty", "elevation", "data-testid"]):
        return False
    if any(s in html_lower for s in [
        "unusual activity detected", "access is temporarily restricted",
        "cf-browser-verification", "cf_clearance",
        "checking your browser before accessing",
        "enable javascript and cookies to continue",
        "ray id:", "why have i been blocked", "ddos-guard",
    ]):
        return True
    return len(html) < 5000


# Junk text filters
_UI_JUNK = re.compile(
    r"ai.generated|suggest trail update|mark as completed|begin typing to search"
    r"|up and down arrow keys|review trail|sign in|log in|sign up"
    r"|show activity|show more|load more|read more|see all"
    r"|bathrooms available|easy to park|not crowded|great conditions|great views"
    r"|\d+\s*(reviews|activities|photos)",
    re.IGNORECASE
)

_REVIEW_HEADER = re.compile(
    r"^[A-Z][a-z]+ [A-Z][a-z]+ (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}, \d{4}",
    re.IGNORECASE
)


def _is_junk(sentence):
    """Check if sentence is UI noise rather than review content."""
    if _UI_JUNK.search(sentence):
        return True
    if _REVIEW_HEADER.match(sentence) and len(sentence.split()) < 12:
        return True
    tokens = sentence.split()
    if len(tokens) > 4:
        real_words = [t for t in tokens if re.match(r"[a-zA-Z]{2,}", t)]
        if len(real_words) / len(tokens) < 0.5:
            return True
    return False


def extract_sentences(html):
    """Extract clean sentences from trail description and reviews."""
    soup = BeautifulSoup(html, "html.parser")
    texts = []
    
    # Trail description
    for selector in [
        {"attrs": {"data-testid": "trail-detail-overview"}},
        {"class_": "trail-description"},
        {"class_": "MuiTypography-body1"},
    ]:
        el = soup.find("div", **selector)
        if el:
            texts.append(el.get_text(separator=" ", strip=True))
            break
    
    # User reviews
    review_els = (
        soup.find_all(attrs={"data-testid": "review-card"}) or
        soup.find_all("div", class_=re.compile(r"review", re.I)) or
        soup.find_all("p", class_=re.compile(r"review|comment|body", re.I))
    )
    seen_blocks = set()
    for el in review_els:
        text = el.get_text(separator=" ", strip=True)
        if text and text not in seen_blocks and len(text) > 20:
            texts.append(text)
            seen_blocks.add(text)
    
    all_text = " ".join(texts)
    raw_sentences = re.split(r'(?<=[.!?])\s+', all_text)
    
    # Deduplicate and filter junk
    seen_sentences = set()
    clean = []
    for s in raw_sentences:
        s = s.strip()
        if len(s) < 15:
            continue
        if s in seen_sentences:
            continue
        if _is_junk(s):
            continue
        seen_sentences.add(s)
        clean.append(s)
    
    return clean


def filter_by_category(sentences, patterns):
    """Filter sentences by category patterns."""
    results = []
    for s in sentences:
        hits = []
        for p in patterns:
            m = p.search(s)
            if m:
                hits.append(m.group(0).lower())
        if hits:
            results.append((s, sorted(set(hits))))
    return results


def build_analyzer(pos_words, neg_words):
    """Build VADER analyzer with custom word lists."""
    analyzer = SentimentIntensityAnalyzer()
    for word in pos_words:
        if word not in analyzer.lexicon:
            analyzer.lexicon[word] = 3.0
    for word in neg_words:
        if word not in analyzer.lexicon:
            analyzer.lexicon[word] = -3.0
    return analyzer


def score_sentences(matches, analyzer, category_name="", verbose=False):
    """Score sentences using VADER sentiment analyzer."""
    if not matches:
        return None
    
    if verbose and category_name:
        print(f"    -- {category_name} --")
    
    total = 0.0
    for item in matches:
        sentence, keywords = item if isinstance(item, tuple) else (item, [])
        score = analyzer.polarity_scores(sentence)["compound"]
        total += score
        
        if verbose and category_name:
            kw_str = f"[{', '.join(keywords)}]" if keywords else ""
            bar = chr(9619) * int(abs(score) * 10)
            polarity = "+" if score >= 0 else "-"
            print(f"    {polarity}{abs(score):.3f} {bar:<10} {kw_str}")
            preview = sentence if len(sentence) <= 120 else sentence[:117] + "..."
            print(f'           "{preview}"')
    
    return round(total / len(matches), 4)


def score_trail(sentences, categories, analyzer, verbose=False):
    """Score all sentiment categories for a trail."""
    all_tuples = [(s, []) for s in sentences]
    
    if verbose:
        print(f"    -- All ({len(sentences)} sentences) --")
    
    results = {
        "sentiment_all": score_sentences(all_tuples, analyzer),
        "sentence_count_all": len(sentences),
    }
    
    for cat_name, patterns in categories.items():
        matched = filter_by_category(sentences, patterns)
        score = score_sentences(matched, analyzer, category_name=cat_name, verbose=verbose)
        col = cat_name.lower().replace(" ", "_")
        results[f"sentiment_{col}"] = score if score is not None else NO_MATCH_VALUE
        results[f"sentence_count_{col}"] = len(matched)
    
    return results


def scrape_trails(input_csv=INPUT_CSV_RAW, output_csv=OUTPUT_CSV_SENTIMENT):
    """Main sentiment scraping pipeline."""
    print("\n" + "="*80)
    print("STEP 1: SENTIMENT SCRAPING (IDs_to_SENT)")
    print("="*80)
    
    categories, pos_words, neg_words = load_config()
    analyzer = build_analyzer(pos_words, neg_words)
    
    df = pd.read_csv(input_csv, header=0, dtype=str)
    id_col_name = df.columns[0]
    print(f"\nLoaded {len(df)} rows. Trail ID column: '{id_col_name}'")
    
    # Ensure all output columns exist
    score_cols = (
        ["sentiment_all", "sentence_count_all"] +
        [f"sentiment_{c.lower().replace(' ', '_')}" for c in categories] +
        [f"sentence_count_{c.lower().replace(' ', '_')}" for c in categories] +
        ["scrape_status"]
    )
    for col in score_cols:
        if col not in df.columns:
            df[col] = ""
    
    print("Starting Chrome driver for web scraping...")
    driver = make_driver()
    blocked_count = 0
    
    try:
        for idx, row in df.iterrows():
            trail_id = str(row[id_col_name]).strip()
            
            # Skip already-scored rows
            existing = str(row.get("sentiment_all", "")).strip()
            if existing and existing not in ("", "nan", "ERROR", "BLOCKED"):
                print(f"[{idx+1}/{len(df)}] Skipping '{trail_id}' (already scored)")
                continue
            
            url = f"{BASE_URL}{trail_id}"
            print(f"[{idx+1}/{len(df)}] Fetching: {url}")
            
            try:
                html = fetch_html(driver, url)
                
                if is_blocked(html):
                    blocked_count += 1
                    df.at[idx, "scrape_status"] = "BLOCKED"
                    print(f"  ✗ BLOCKED ({len(html):,} chars)")
                    if blocked_count >= 3:
                        print("  WARNING: Blocked 3x. Pausing 60s...")
                        time.sleep(60)
                        blocked_count = 0
                    else:
                        time.sleep(30)
                else:
                    blocked_count = 0
                    sentences = extract_sentences(html)
                    if VERBOSE:
                        print(f"  Extracting sentiment for {len(sentences)} sentences...")
                    scores = score_trail(sentences, categories, analyzer, verbose=VERBOSE)
                    for col, val in scores.items():
                        df.at[idx, col] = val
                    df.at[idx, "scrape_status"] = "OK"
                    
                    cat_summary = "  ".join(
                        f"{c.lower()}={scores.get(f'sentiment_{c.lower().replace(chr(32), chr(95))}', 'n/a')}"
                        for c in categories
                    )
                    print(f"  ✓ all={scores['sentiment_all']:+.3f}  {cat_summary}  ({scores['sentence_count_all']} sentences)")
            
            except Exception as e:
                df.at[idx, "scrape_status"] = f"ERROR: {e}"
                print(f"  Error: {e}")
            
            # Save after every row so progress is never lost
            df.to_csv(output_csv, index=False)
            
            if idx < len(df) - 1:
                delay = random.uniform(DELAY_MIN, DELAY_MAX)
                print(f"  Waiting {delay:.1f}s...")
                time.sleep(delay)
    
    finally:
        driver.quit()
        print("\nBrowser closed.")
    
    df.to_csv(output_csv, index=False)
    
    ok = (df["scrape_status"] == "OK").sum()
    blocked = (df["scrape_status"] == "BLOCKED").sum()
    errors = df["scrape_status"].str.startswith("ERROR", na=False).sum()
    print(f"\n── Sentiment Scraping Summary ──────────────")
    print(f"  Total  : {len(df)}")
    print(f"  Scored : {ok}")
    print(f"  Blocked: {blocked}")
    print(f"  Errors : {errors}")
    print(f"  Output : {output_csv}\n")
    
    return df

# ============================================================================
# STEP 2: DATA PREPROCESSING (Preprocessing)
# ============================================================================

class TrailPreprocessor:
    """Data preprocessing and cleaning for trail popularity dataset."""
    
    def __init__(self, filepath: str):
        """Initialize the preprocessor."""
        self.df = pd.read_csv(filepath)
        self.original_shape = self.df.shape
        print(f"Dataset loaded: {self.original_shape[0]:,} rows, {self.original_shape[1]} columns")
    
    def filter_trail_ids(self) -> 'TrailPreprocessor':
        """Step 1: Keep only rows with 8-digit trail IDs."""
        print("\n[Step 1] Filtering trail_ids to 8-digit numbers...")
        
        def is_valid_trail_id(trail_id):
            try:
                return len(str(int(trail_id))) == 8
            except (ValueError, TypeError):
                return False
        
        initial_rows = len(self.df)
        self.df = self.df[self.df['trail_id'].apply(is_valid_trail_id)]
        removed = initial_rows - len(self.df)
        
        print(f"  Removed {removed} invalid trail_ids")
        print(f"  Remaining rows: {len(self.df)}")
        
        return self
    
    def merge_nested_columns(self) -> 'TrailPreprocessor':
        """Step 2: Merge nested columns (activities/X, collections/X, etc.)."""
        print("\n[Step 2] Merging nested columns into lists...")
        
        nested_prefixes = set()
        for col in self.df.columns:
            if '/' in col:
                prefix = col.split('/')[0]
                nested_prefixes.add(prefix)
        
        print(f"  Found {len(nested_prefixes)} nested column groups: {sorted(nested_prefixes)}")
        
        for prefix in sorted(nested_prefixes):
            nested_cols = [col for col in self.df.columns if col.startswith(prefix + '/')]
            
            if nested_cols:
                def merge_nested_values(row):
                    values = []
                    for col in nested_cols:
                        val = row[col]
                        if pd.notna(val) and val != '' and val is not None:
                            if isinstance(val, str):
                                val = val.strip()
                                if val:
                                    values.append(val)
                            else:
                                values.append(val)
                    return values if values else []
                
                self.df[prefix] = self.df[nested_cols].apply(merge_nested_values, axis=1)
                self.df = self.df.drop(columns=nested_cols)
                print(f"  ✓ Merged {len(nested_cols)} columns into '{prefix}'")
        
        print(f"  Final column count: {len(self.df.columns)}")
        
        return self
    
    def filter_scrape_status(self) -> 'TrailPreprocessor':
        """Step 3: Drop rows with scrape errors, keep only 'OK' status."""
        print("\n[Step 3] Filtering by scrape_status...")
        
        if 'scrape_status' not in self.df.columns:
            print("  ⚠ Warning: 'scrape_status' column not found. Skipping this step.")
            return self
        
        initial_rows = len(self.df)
        
        status_counts = self.df['scrape_status'].value_counts()
        print(f"  Scrape status distribution:\n{status_counts.to_string()}")
        
        errored_df = self.df[self.df['scrape_status'] != 'OK']
        self.errored_trail_ids = errored_df[['trail_id', 'scrape_status']].copy()
        
        self.df = self.df[self.df['scrape_status'] == 'OK']
        removed = initial_rows - len(self.df)
        
        print(f"  Saved {len(self.errored_trail_ids)} errored trail_ids for later retry")
        print(f"  Removed {removed} rows with error status")
        print(f"  Remaining rows: {len(self.df)}")
        
        return self
    
    def fill_sentiment_columns(self) -> 'TrailPreprocessor':
        """Step 4: Fill empty sentiment columns with 0."""
        print("\n[Step 4] Filling empty cells in sentiment columns with 0...")
        
        sentiment_cols = [
            'sentiment_all', 'sentence_count_all', 'sentiment_views',
            'sentiment_difficulty', 'sentiment_crowding', 'sentiment_conditions',
            'sentence_count_views', 'sentence_count_difficulty',
            'sentence_count_crowding', 'sentence_count_conditions'
        ]
        
        existing_sentiment_cols = [col for col in sentiment_cols if col in self.df.columns]
        
        if existing_sentiment_cols:
            nan_counts = self.df[existing_sentiment_cols].isna().sum()
            total_nans = nan_counts.sum()
            
            self.df[existing_sentiment_cols] = self.df[existing_sentiment_cols].fillna(0)
            
            print(f"  Filled {total_nans} NaN values across {len(existing_sentiment_cols)} columns")
            print(f"  Columns filled: {', '.join(existing_sentiment_cols)}")
        
        return self
    
    def create_percentage_columns(self) -> 'TrailPreprocessor':
        """Step 5: Create percentage columns for model generalization."""
        print("\n[Step 5] Creating percentage columns...")
        
        pct_created = 0
        
        # Sentence count percentages
        sentence_cols = [
            'sentence_count_views', 'sentence_count_difficulty',
            'sentence_count_crowding', 'sentence_count_conditions'
        ]
        
        if 'sentence_count_all' in self.df.columns:
            for col in sentence_cols:
                if col in self.df.columns:
                    self.df[f'{col}_pct'] = (
                        self.df[col] / self.df['sentence_count_all'].replace(0, np.nan)
                    ).fillna(0)
                    pct_created += 1
        
        # Photo/Recording percentages
        photo_cols = ['numPhotos', 'numRecordings', 'numFeaturedPhotos']
        
        if 'numReviews' in self.df.columns:
            for col in photo_cols:
                if col in self.df.columns:
                    self.df[f'{col}_pct'] = (
                        self.df[col] / self.df['numReviews'].replace(0, np.nan)
                    ).fillna(0)
                    pct_created += 1
        
        # Collections with photos percentage
        if 'collectionsWithPhotos' in self.df.columns and 'collections' in self.df.columns:
            self.df['collections_with_photos_count'] = self.df['collectionsWithPhotos'].apply(len)
            self.df['collections_count'] = self.df['collections'].apply(len)
            
            self.df['collections_with_photos_pct'] = (
                self.df['collections_with_photos_count'] / 
                self.df['collections_count'].replace(0, np.nan)
            ).fillna(0)
            pct_created += 1
        
        print(f"  Created {pct_created} percentage columns")
        print(f"  Columns after percentage creation: {len(self.df.columns)}")
        
        return self
    
    def handle_list_columns(self) -> 'TrailPreprocessor':
        """Step 6: Process list columns and normalize other columns."""
        print("\n[Step 6] Processing list columns and normalizing other columns...")
        
        list_cols = ['features', 'activities', 'collections']
        
        for col in list_cols:
            if col in self.df.columns:
                print(f"  Processing {col}...")
                
                self.df[f'{col}_list'] = self.df[col].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
                
                exploded = self.df[f'{col}_list'].explode()
                mlb_encoded = pd.get_dummies(exploded).groupby(level=0).max()
                mlb_encoded.columns = [f"{col}_{c}" for c in mlb_encoded.columns]
                
                mlb_encoded = mlb_encoded.fillna(0).astype(int)
                mlb_encoded = mlb_encoded.reindex(self.df.index, fill_value=0)
                
                self.df = self.df.drop(columns=[col, f'{col}_list'])
                self.df = pd.concat([self.df, mlb_encoded], axis=1)
                
                print(f"    ✓ Created {len(mlb_encoded.columns)} binary columns for {col}")
        
        # Handle poiLocations
        if 'poiLocations' in self.df.columns:
            print(f"  Processing poiLocations...")
            self.df['numPOIs'] = self.df['poiLocations'].apply(
                lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else len(x) if pd.notna(x) else 0
            )
            self.df = self.df.drop(columns=['poiLocations'])
            print(f"    ✓ Converted poiLocations to numPOIs")
        
        # Normalize difficulty
        if 'difficulty' in self.df.columns:
            print(f"  Normalizing difficulty...")
            difficulty_map = {1: 1, 3: 2, 5: 3, 7: 4}
            self.df['difficulty'] = self.df['difficulty'].map(difficulty_map)
            print(f"    ✓ Mapped difficulty to 1-4 scale")
        
        # Normalize estimatedTime
        if 'estimatedTime' in self.df.columns:
            print(f"  Normalizing estimatedTime...")
            
            def extract_time_value(time_str):
                if pd.isna(time_str) or time_str == '':
                    return np.nan
                
                time_str = str(time_str).strip()
                time_str = time_str.replace(' hr', '').replace(' hour', '').replace(' hrs', '').strip()
                
                if '-' in time_str or '–' in time_str:
                    time_str = time_str.replace('–', '-').split('-')[0].strip()
                
                numeric_str = ''
                for c in time_str:
                    if c.isdigit() or c == '.':
                        numeric_str += c
                
                try:
                    return float(numeric_str) if numeric_str else np.nan
                except ValueError:
                    return np.nan
            
            self.df['estimatedTime'] = self.df['estimatedTime'].apply(extract_time_value)
            print(f"    ✓ Extracted numeric values from estimatedTime")
        
        print(f"  Columns after list processing: {len(self.df.columns)}")
        
        return self
    
    def handle_missing_values(self) -> 'TrailPreprocessor':
        """Step 8: Handle missing values with imputation."""
        print("\n[Step 8] Handling missing values with imputation...")
        
        # Handle estimatedTime
        if 'estimatedTime' in self.df.columns and 'lengthMiles' in self.df.columns:
            missing_count = self.df['estimatedTime'].isna().sum()
            if missing_count > 0:
                def calculate_time(length_miles):
                    if pd.isna(length_miles) or length_miles == 0:
                        return np.nan
                    time_hours = (length_miles * 25) / 60
                    time_rounded = np.ceil(time_hours * 2) / 2
                    return time_rounded
                
                mask = self.df['estimatedTime'].isna()
                self.df.loc[mask, 'estimatedTime'] = self.df.loc[mask, 'lengthMiles'].apply(calculate_time)
                filled_count = mask.sum()
                print(f"  ✓ Calculated {filled_count} missing estimatedTime values")
        
        # Handle elevationGainMeters
        if 'elevationGainMeters' in self.df.columns and 'highestPoint' in self.df.columns and 'elevationMeters' in self.df.columns:
            missing_count = self.df['elevationGainMeters'].isna().sum()
            if missing_count > 0:
                mask = self.df['elevationGainMeters'].isna()
                self.df.loc[mask, 'elevationGainMeters'] = (
                    self.df.loc[mask, 'highestPoint'] - self.df.loc[mask, 'elevationMeters']
                )
                print(f"  ✓ Calculated {missing_count} missing elevationGainMeters")
        
        print(f"  Remaining missing values after imputation:")
        remaining_missing = self.df.isnull().sum()
        cols_with_missing = remaining_missing[remaining_missing > 0]
        if len(cols_with_missing) > 0:
            for col, count in cols_with_missing.items():
                pct = (count / len(self.df)) * 100
                print(f"    - {col}: {count} ({pct:.2f}%)")
        else:
            print(f"    None! All columns handled.")
        
        return self
    
    def drop_unnecessary_columns(self) -> 'TrailPreprocessor':
        """Step 7: Drop unnecessary columns."""
        print("\n[Step 7] Dropping unnecessary columns...")
        
        columns_to_drop = []
        
        # Duration columns
        duration_cols = [col for col in self.df.columns if col.startswith('duration')]
        columns_to_drop.extend(duration_cols)
        
        # Elevation in feet
        if 'elevationGainFt' in self.df.columns:
            columns_to_drop.append('elevationGainFt')
        
        # URL columns
        url_cols = [col for col in self.df.columns if col.endswith('Url')]
        columns_to_drop.extend(url_cols)
        
        # Slug columns
        slug_cols = [col for col in self.df.columns if 'slug' in col.lower()]
        columns_to_drop.extend(slug_cols)
        
        # ID columns (except trail_id)
        id_cols = [col for col in self.df.columns if col.endswith('Id') and col != 'trail_id']
        columns_to_drop.extend(id_cols)
        
        # Target leakage columns
        leakage_cols = ['difficulty', 'wheelchairFriendly', 'numReviews', 'visitorUsage', 'seasonalPopularity', 'numTextReviews', 'trailId']
        columns_to_drop.extend([col for col in leakage_cols if col in self.df.columns])
        
        # Original count columns (replaced by percentages)
        count_cols = [
            'numPhotos', 'numRecordings', 'numFeaturedPhotos',
            'sentence_count_views', 'sentence_count_difficulty',
            'sentence_count_crowding', 'sentence_count_conditions',
            'collectionsWithPhotos', 'collections_with_photos_count'
        ]
        columns_to_drop.extend([col for col in count_cols if col in self.df.columns])
        
        # Other unnecessary columns
        other_cols = [
            'alertIds', 'alertTypes', 'areaName', 'associatedAreaIds', 'countryName', 'isPrivateProperty', 'popularityByMonth',
            'scrapedAt', 'locationLabel', 'name', 'description', 'profilePhotoUrl',
            'mapUrl', 'photoPageUrl', 'createdAt', 'areaSlug', 'cityUrl', 'trailUrl',
            'scrape_status', 'featuredPhotoIds', 'trailTypeLabel', 'trailType',
            'activities', 'collections', 'features', 'poiLocations'
        ]
        columns_to_drop.extend([col for col in other_cols if col in self.df.columns])
        
        columns_to_drop = list(set(columns_to_drop))
        columns_to_drop = [col for col in columns_to_drop if col in self.df.columns]
        
        initial_columns = len(self.df.columns)
        self.df = self.df.drop(columns=columns_to_drop)
        removed = initial_columns - len(self.df.columns)
        
        print(f"  Dropped {removed} unnecessary columns")
        print(f"  Remaining columns: {len(self.df.columns)}")
        if columns_to_drop:
            print(f"\n  Dropped columns:")
            for col in sorted(columns_to_drop)[:20]:
                print(f"    - {col}")
            if len(columns_to_drop) > 20:
                print(f"    ... and {len(columns_to_drop) - 20} more")
        
        return self
    
    def get_dataframe(self) -> pd.DataFrame:
        """Return the processed dataframe."""
        return self.df
    
    def get_info(self) -> Dict[str, Any]:
        """Return preprocessing summary."""
        return {
            'original_shape': self.original_shape,
            'final_shape': self.df.shape,
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_original': self.original_shape[1],
            'columns_final': self.df.shape[1],
            'columns_changed': self.original_shape[1] - self.df.shape[1]
        }
    
    def save(self, filepath: str) -> None:
        """Save processed dataframe to CSV."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.df.to_csv(filepath, index=False)
        print(f"\n✓ Preprocessed dataset saved to: {filepath}")
        print(f"  Shape: {self.df.shape}")
    
    def save_errored_trails(self, filepath: str) -> None:
        """Save errored trail IDs."""
        if hasattr(self, 'errored_trail_ids') and len(self.errored_trail_ids) > 0:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.errored_trail_ids.to_csv(filepath, index=False)
            print(f"✓ Errored trail_ids saved to: {filepath}")
            print(f"  Count: {len(self.errored_trail_ids)}")
    
    def preview(self, n_rows: int = 3) -> None:
        """Display data preview."""
        print(f"\n[Preview] First {n_rows} rows of processed data:")
        print(f"Shape: {self.df.shape}\n")
        print(self.df.head(n_rows).to_string())


def preprocess_trails(input_csv=INPUT_CSV_SENTIMENT):
    """Main preprocessing pipeline."""
    print("\n" + "="*80)
    print("STEP 2: DATA PREPROCESSING")
    print("="*80)
    
    preprocessor = TrailPreprocessor(input_csv)
    
    (preprocessor
     .filter_trail_ids()
     .merge_nested_columns()
     .filter_scrape_status()
     .fill_sentiment_columns()
     .create_percentage_columns()
     .handle_list_columns()
     .handle_missing_values()
     .drop_unnecessary_columns())
    
    info = preprocessor.get_info()
    print("\n" + "="*80)
    print("PREPROCESSING SUMMARY")
    print("="*80)
    print(f"Original dataset:      {info['original_shape'][0]:,} rows × {info['original_shape'][1]} columns")
    print(f"Final dataset:         {info['final_shape'][0]:,} rows × {info['final_shape'][1]} columns")
    print(f"Rows removed:          {info['rows_removed']:,}")
    print(f"Columns reduced from:  {info['columns_original']} → {info['columns_final']} (change: {-info['columns_changed']})")
    print("="*80)
    
    preprocessor.preview(n_rows=3)
    
    output_path = f'{OUTPUT_DIR}/preprocessed_trails.csv'
    preprocessor.save(output_path)
    
    errored_path = f'{OUTPUT_DIR}/errored_trails.csv'
    preprocessor.save_errored_trails(errored_path)
    
    print(f"\n[Columns] Final dataset has {len(preprocessor.df.columns)} columns:")
    for i, col in enumerate(preprocessor.df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    return preprocessor.df

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main pipeline: Sentiment scraping → Data preprocessing."""
    print("\n" + "="*80)
    print("TRAIL POPULARITY PREDICTION - COMPLETE DATA PREPROCESSING PIPELINE")
    print("="*80)
    
    # Check if config files exist, create if not
    if not Path("config/categories.json").exists():
        print("\nConfig files not found. Creating defaults...")
        setup_config_files()
    
    # Step 1: Sentiment Scraping
    try:
        df_sentiment = scrape_trails(INPUT_CSV_RAW, OUTPUT_CSV_SENTIMENT)
    except Exception as e:
        print(f"\nError during sentiment scraping: {e}")
        print("Trying to continue with existing trails_sentiment.csv...")
        if not Path(OUTPUT_CSV_SENTIMENT).exists():
            print(f"ERROR: {OUTPUT_CSV_SENTIMENT} not found. Cannot continue.")
            return
        df_sentiment = pd.read_csv(OUTPUT_CSV_SENTIMENT)
    
    # Step 2: Data Preprocessing
    try:
        df_final = preprocess_trails(OUTPUT_CSV_SENTIMENT)
    except Exception as e:
        print(f"\nError during preprocessing: {e}")
        raise
    
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE!")
    print("="*80)
    print(f"Final dataset ready at: {OUTPUT_DIR}/preprocessed_trails.csv")
    print(f"Ready for model training!")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
