
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import itertools
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class AggregateVotingModel(nn.Module):
    """
    Aggregate Likelihood Model for Electoral Prediction
    Based on the comprehensive guide document - CORRECTED VERSION
    """
    
    def __init__(self, n_features, n_parties, n_booths, device='cpu'):
        super(AggregateVotingModel, self).__init__()
        
        self.device = device
        
        # Model parameters as per document notation
        # Turnout model: π_ic = σ(α0 + b_i^(T) + x_ic^T β_T)
        self.alpha0 = nn.Parameter(torch.randn(1, device=device))  # Global turnout intercept
        self.beta_T = nn.Parameter(torch.randn(n_features, device=device))  # Turnout coefficients
        self.booth_effects_T = nn.Parameter(torch.randn(n_booths, device=device))  # Booth effects for turnout
        
        # Party choice model: θ_ic = softmax(γ0 + b_i^(P) + x_ic^T β_P)
        self.gamma0 = nn.Parameter(torch.randn(n_parties, device=device))  # Global party intercepts
        self.beta_P = nn.Parameter(torch.randn(n_features, n_parties, device=device))  # Party coefficients
        self.booth_effects_P = nn.Parameter(torch.randn(n_booths, n_parties, device=device))  # Booth effects for parties
        
        self.n_parties = n_parties
        self.n_booths = n_booths
        
    def forward(self, features, booth_indices):
        """
        Forward pass implementing the mathematical framework from document
        """
        batch_size = features.shape[0]
        
        # Ensure inputs are on correct device
        features = features.to(self.device)
        booth_indices = booth_indices.to(self.device)
        
        # Turnout model: π_ic = σ(α0 + b_i^(T) + x_ic^T β_T)
        turnout_logits = (self.alpha0 + 
                         self.booth_effects_T[booth_indices] + 
                         torch.matmul(features, self.beta_T))
        turnout_probs = torch.sigmoid(turnout_logits)
        
        # Party choice model: θ_ic = softmax(γ0 + b_i^(P) + x_ic^T β_P)
        party_logits = (self.gamma0.unsqueeze(0).expand(batch_size, -1) + 
                       self.booth_effects_P[booth_indices] + 
                       torch.matmul(features, self.beta_P))
        
        # Use log_softmax for numerical stability as per guide
        party_log_probs = torch.log_softmax(party_logits, dim=1)
        party_probs = torch.exp(party_log_probs)
        
        return turnout_probs, party_probs, party_log_probs


def classify_alignment(p_hat_value):
    """
    Classify alignment category based on maximum party probability (p_max).
    Heuristics:
      - p_max >= 0.7 -> 'core'
      - 0.4 <= p_max < 0.7 -> 'leaning'
      - p_max < 0.4 -> 'swing'
    Returns a short category string.
    """
    try:
        p = float(p_hat_value)
    except Exception:
        return 'unknown'

    if p >= 0.7:
        return 'core'
    elif p >= 0.4:
        return 'leaning'
    else:
        return 'swing'

class ElectoralDataProcessor:
    """
    FIXED: Data processor with correct column names and Unknown category handling
    """
    
    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=False)
        self.scaler = StandardScaler()
        self.booth_id_to_idx = {}  # Explicit booth ID mapping
        self.party_names = ['BJP', 'Congress', 'AAP', 'Others', 'NOTA']
        self.correction_log = []  # Log corrections as per spec
        self.is_fitted = False  # Track if transformers are fitted
        
        # FIXED: Define demographic axes with correct column names (matching dataset)
        # EXCLUDING Unknown categories as per your suggestions
        self.demographic_axes = {
            'age': ['Age_18-25_Ratio', 'Age_26-35_Ratio', 'Age_36-45_Ratio', 'Age_46-60_Ratio', 'Age_60+_Ratio'],
            'religion': ['Religion_Buddhist_Ratio', 'Religion_Christian_Ratio', 'Religion_Hindu_Ratio',
                        'Religion_Jain_Ratio', 'Religion_Muslim_Ratio', 'Religion_Sikh_Ratio'],
            'caste': ['Caste_Brahmin_Ratio', 'Caste_Kshatriya_Ratio', 'Caste_Obc_Ratio',
                     'Caste_Sc_Ratio', 'Caste_St_Ratio', 'Caste_Vaishya_Ratio', 'Caste_No_caste_system_Ratio'],
            'income': ['income_low_ratio', 'income_middle_ratio', 'income_high_ratio']
        }
        
        # Booth-level caches for efficient training (as per guide section 5)
        self.booth_to_cell_indices = {}
        self.booth_to_weights = {}
        self.booth_metadata = {}  # N_i, T_i, p_i cache

    def validate_data_columns(self, df):
        """
        CRITICAL: Add this function to validate all required columns exist
        """
        required_columns = {
            'basic': ['PartNo', 'AssemblyNo', 'AssemblyName', 'TotalPop', 'Total_Polled', 'Locality'],
            'economic': ['economic_category', 'land_rate_per_sqm', 'construction_cost_per_sqm'],
            'gender': ['Male_Ratio', 'Female_Ratio', 'MaleToFemaleRatio'],
            'age': ['Age_18-25_Ratio', 'Age_26-35_Ratio', 'Age_36-45_Ratio', 'Age_46-60_Ratio', 'Age_60+_Ratio'],
            'religion': ['Religion_Buddhist_Ratio', 'Religion_Christian_Ratio', 'Religion_Hindu_Ratio', 
                        'Religion_Jain_Ratio', 'Religion_Muslim_Ratio', 'Religion_Sikh_Ratio', 'Religion_Unknown_Ratio'],
            'caste': ['Caste_Brahmin_Ratio', 'Caste_Kshatriya_Ratio', 'Caste_No_caste_system_Ratio', 
                     'Caste_Obc_Ratio', 'Caste_Sc_Ratio', 'Caste_St_Ratio', 'Caste_Vaishya_Ratio', 'Caste_Unknown_Ratio'],
            'parties': ['BJP_Ratio', 'Congress_Ratio', 'AAP_Ratio', 'Others_Ratio', 'NOTA_Ratio']
        }
        
        missing_columns = []
        existing_columns = set(df.columns)
        
        for category, columns in required_columns.items():
            missing_in_category = [col for col in columns if col not in existing_columns]
            if missing_in_category:
                missing_columns.extend([(category, col) for col in missing_in_category])
        
        if missing_columns:
            print("❌ MISSING REQUIRED COLUMNS:")
            for category, col in missing_columns:
                print(f"  {category}: {col}")
            
            print("\n📋 AVAILABLE COLUMNS IN DATASET:")
            for i, col in enumerate(sorted(existing_columns), 1):
                print(f"  {i:2d}. {col}")
            
            return False
        else:
            print("✅ All required columns found in dataset")
            return True
        
    def load_and_combine_data(self, file_2020, file_2025):
        """Load and combine datasets. file_2020 can be None for single-year training"""
        df_2025 = pd.read_excel(file_2025)
        df_2025['Year'] = 2025
        
        if file_2020 is not None:
            df_2020 = pd.read_excel(file_2020)
            df_2020['Year'] = 2020
            combined_df = pd.concat([df_2020, df_2025], ignore_index=True)
            print(f"Loaded 2020 data: {len(df_2020)} booths, 2025 data: {len(df_2025)} booths")
        else:
            combined_df = df_2025.copy()
            print(f"Using only 2025 data: {len(df_2025)} booths")
        
        # CRITICAL: Validate columns before proceeding
        print("Validating dataset columns...")
        if not self.validate_data_columns(combined_df):
            raise ValueError("❌ CRITICAL ERROR: Missing required columns. Please check your dataset.")
        
        # Clean and validate data as per document requirements
        combined_df = self._clean_and_validate_data(combined_df)
        
        return combined_df
    
    def _validate_economic_categories(self, df):
        """
        Validate and analyze the economic category distribution
        """
        print("\nAnalyzing economic_category distribution:")
        
        if 'economic_category' in df.columns:
            econ_counts = df['economic_category'].value_counts()
            print("Economic categories found:")
            for category, count in econ_counts.items():
                print(f"  {category}: {count} booths ({count/len(df)*100:.1f}%)")
            
            missing_count = df['economic_category'].isna().sum()
            if missing_count > 0:
                print(f"  Missing/NaN: {missing_count} booths ({missing_count/len(df)*100:.1f}%)")
        else:
            print("ERROR: No 'economic_category' column found in dataset!")
            self.correction_log.append("WARNING: No economic_category column found")
        
        return df

    def _suppress_unknown_and_renormalize(self, df):
        """
        IMPLEMENTATION OF YOUR SUGGESTION: Exclude "Unknown" categories + renormalize
        """
        print("Suppressing Unknown categories and renormalizing...")
        
        # --- Religion: zero-out Unknown, renormalize known to sum 1 per booth ---
        rel_known = [c for c in df.columns if c.startswith('Religion_') and c.endswith('_Ratio') and 'Unknown' not in c]
        rel_unknown = 'Religion_Unknown_Ratio'
        
        if rel_unknown in df.columns:
            print(f"Found {rel_unknown}, suppressing and renormalizing...")
            df[rel_unknown] = 0.0  # drop from model
            denom = df[rel_known].sum(axis=1)
            
            # if denom is 0 (all were unknown or missing), fallback to uniform over known
            uniform = 1.0 / max(len(rel_known), 1)
            for c in rel_known:
                df.loc[denom > 0, c] = df.loc[denom > 0, c] / denom[denom > 0]
                df.loc[denom == 0, c] = uniform
                
            self.correction_log.append("Suppressed Religion_Unknown and renormalized known religion ratios")
        
        # --- Caste: zero-out Unknown globally; Hindu-only caste handled later ---
        caste_known = [c for c in df.columns if c.startswith('Caste_') and c.endswith('_Ratio') and 'Unknown' not in c]
        caste_unknown = 'Caste_Unknown_Ratio'
        
        if caste_unknown in df.columns:
            print(f"Found {caste_unknown}, suppressing...")
            df[caste_unknown] = 0.0
            # We won't renormalize caste here across the whole booth; we do it within Hindus in construct_cells
            self.correction_log.append("Suppressed Caste_Unknown (handled Hindu-only caste later)")
        
        return df

    def _create_income_categories(self, df):
        """
        FIXED: Use actual economic categories from dataset instead of artificial terciles
        Maps 'economic_category' to standardized income ratios
        """
        print("Creating income categories from actual economic_category data...")
        
        # Map actual economic categories to income levels
        economic_to_income_mapping = {
            'LOW INCOME AREAS': 'income_low',
            'LOWER MIDDLE CLASS': 'income_middle', 
            'MIDDLE CLASS': 'income_middle',
            'UPPER MIDDLE CLASS': 'income_high',
            'PREMIUM AREAS': 'income_high'
        }
        
        # Initialize all categories to 0
        df['income_low_ratio'] = 0.0
        df['income_middle_ratio'] = 0.0  
        df['income_high_ratio'] = 0.0
        
        # Map based on actual economic categories
        for economic_cat, income_cat in economic_to_income_mapping.items():
            mask = df['economic_category'] == economic_cat
            df.loc[mask, f'{income_cat}_ratio'] = 1.0
            
            if mask.any():
                print(f"Mapped {mask.sum()} booths from '{economic_cat}' to '{income_cat}'")
        
        # Handle missing/unknown economic categories
        income_cols = ['income_low_ratio', 'income_middle_ratio', 'income_high_ratio']
        no_category_mask = df[income_cols].sum(axis=1) == 0
        
        if no_category_mask.any():
            # For missing categories, use land rate as fallback
            print(f"Using land rate fallback for {no_category_mask.sum()} booths with missing economic_category")
            
            land_rates = df.loc[no_category_mask, 'land_rate_per_sqm'].fillna(
                df['land_rate_per_sqm'].median()
            )
            
            # Create tertiles only for missing data
            low_thresh = land_rates.quantile(0.33)
            high_thresh = land_rates.quantile(0.67)
            
            low_mask = land_rates <= low_thresh
            middle_mask = (land_rates > low_thresh) & (land_rates <= high_thresh)  
            high_mask = land_rates > high_thresh
            
            # Get indices for missing data
            missing_indices = df.index[no_category_mask]
            
            df.loc[missing_indices[low_mask], 'income_low_ratio'] = 1.0
            df.loc[missing_indices[middle_mask], 'income_middle_ratio'] = 1.0
            df.loc[missing_indices[high_mask], 'income_high_ratio'] = 1.0
            
            self.correction_log.append(f"Used land rate fallback for {no_category_mask.sum()} booths")
        
        # Verify they sum to exactly 1
        income_sums = df[income_cols].sum(axis=1)
        assert np.allclose(income_sums, 1.0), f"Income categories don't sum to 1: {income_sums.describe()}"
        
        # Log the distribution
        distribution = {
            'Low Income': (df['income_low_ratio'] == 1.0).sum(),
            'Middle Income': (df['income_middle_ratio'] == 1.0).sum(), 
            'High Income': (df['income_high_ratio'] == 1.0).sum()
        }
        
        print("Final income distribution:")
        for level, count in distribution.items():
            print(f"  {level}: {count} booths ({count/len(df)*100:.1f}%)")
        
        self.correction_log.append("Used actual economic_category data for income classification")
        return df

    def _clean_and_validate_data(self, df):
        """
        FINAL FIXED: Enhanced data cleaning with Unknown category handling
        """
        print("Cleaning and validating data...")
        
        # First validate economic categories
        df = self._validate_economic_categories(df)
        
        # Handle categorical columns with sentinel values (including economic_category)
        categorical_cols = ['economic_category', 'Locality', 'AssemblyName']
        for col in categorical_cols:
            if col in df.columns:
                missing_mask = df[col].isna()
                if missing_mask.any():
                    df.loc[missing_mask, col] = "__MISSING__"
                    self.correction_log.append(f"Imputed {missing_mask.sum()} missing values in {col}")
        
        # Handle numeric columns with appropriate values
        numeric_cols = [col for col in df.columns if col.endswith('_Ratio') or 
                       col in ['TotalPop', 'Total_Polled', 'land_rate_per_sqm', 'construction_cost_per_sqm', 'MaleToFemaleRatio']]
        for col in numeric_cols:
            if col in df.columns:
                missing_mask = df[col].isna()
                if missing_mask.any():
                    if col.endswith('_Ratio'):
                        df.loc[missing_mask, col] = 1e-6
                    else:
                        df.loc[missing_mask, col] = df[col].median()
                    self.correction_log.append(f"Imputed {missing_mask.sum()} missing values in {col}")
        
        # Check for negative ratios and correct
        ratio_cols = [col for col in df.columns if col.endswith('_Ratio')]
        for col in ratio_cols:
            if col in df.columns:
                negative_mask = df[col] < 0
                if negative_mask.any():
                    df.loc[negative_mask, col] = 1e-6
                    self.correction_log.append(f"Corrected {negative_mask.sum()} negative values in {col}")
        
        # IMPLEMENTATION: Suppress Unknown categories and renormalize
        df = self._suppress_unknown_and_renormalize(df)
        
        # FIXED: Validate and normalize demographic axis sums with correct column names
        age_cols = self.demographic_axes['age']
        religion_cols = self.demographic_axes['religion']
        caste_cols = self.demographic_axes['caste']
        
        for axis_name, cols in [('Age', age_cols), ('Religion', religion_cols), ('Caste', caste_cols)]:
            existing_cols = [col for col in cols if col in df.columns]
            if existing_cols:
                axis_sums = df[existing_cols].sum(axis=1)
                mask = (axis_sums < 0.98) | (axis_sums > 1.02)
                if mask.any():
                    print(f"Normalizing {axis_name} ratios for {mask.sum()} booths")
                    df.loc[mask, existing_cols] = df.loc[mask, existing_cols].div(axis_sums[mask], axis=0)
                    self.correction_log.append(f"Normalized {axis_name} ratios for {mask.sum()} booths")
        
        # Normalize party shares
        party_ratio_cols = [f'{party}_Ratio' for party in self.party_names if f'{party}_Ratio' in df.columns]
        if party_ratio_cols:
            party_sums = df[party_ratio_cols].sum(axis=1)
            mask = (party_sums < 0.98) | (party_sums > 1.02)
            if mask.any():
                df.loc[mask, party_ratio_cols] = df.loc[mask, party_ratio_cols].div(party_sums[mask], axis=0)
                self.correction_log.append(f"Normalized party shares for {mask.sum()} booths")
        
        # Create booth_id and explicit booth indexing
        df['booth_id'] = df['PartNo'].astype(str) + '_' + df['Year'].astype(str)
        unique_booth_ids = df['booth_id'].unique()
        self.booth_id_to_idx = {booth_id: idx for idx, booth_id in enumerate(unique_booth_ids)}
        df['booth_idx'] = df['booth_id'].map(self.booth_id_to_idx)
        
        # Create income categories from actual economic categories
        df = self._create_income_categories(df)
        
        return df
    
    def construct_cells(self, df):
        """
        FIXED: Construct demographic cells with proper caste × religion logic
        IMPLEMENTATION OF YOUR SUGGESTIONS for Hindu-only caste handling
        """
        print("Constructing demographic cells with improved caste × religion logic...")
        
        cells_data = []
        booth_cell_indices = defaultdict(list)
        booth_weights = defaultdict(list)
        
        for idx, row in df.iterrows():
            booth_id = row['booth_id']
            booth_idx = row['booth_idx']
            N_i = row['TotalPop']  # Total registered voters
            T_i = row['Total_Polled']  # Total votes polled
            
            # Skip booths with invalid data
            if N_i <= 0 or pd.isna(N_i):
                self.correction_log.append(f"Booth {booth_id}: Invalid registered voters ({N_i}), skipping")
                continue
                
            # Calculate turnout rate and party shares
            t_i = T_i / N_i if N_i > 0 else 0
            
            # Calculate party vote counts and ensure consistent ordering
            party_shares_raw = {}
            for party in self.party_names:
                party_col = f'{party}_Ratio'
                if party_col in row:
                    party_shares_raw[party] = row[party_col]
                else:
                    party_shares_raw[party] = 0
            
            # Renormalize party shares to sum to 1
            total_share = sum(party_shares_raw.values())
            if total_share > 0:
                party_shares = {party: share/total_share for party, share in party_shares_raw.items()}
            else:
                party_shares = {party: 1.0/len(self.party_names) for party in self.party_names}
            
            # FIXED: Get demographic proportions using correct column names
            age_props = {}
            for age_col in self.demographic_axes['age']:
                if age_col in row:
                    age_props[age_col] = row[age_col]
                else:
                    age_props[age_col] = 0
                    
            religion_props = {}
            for religion_col in self.demographic_axes['religion']:
                if religion_col in row:
                    religion_props[religion_col] = row[religion_col]
                else:
                    religion_props[religion_col] = 0
                    
            caste_props = {}
            for caste_col in self.demographic_axes['caste']:
                if caste_col in row:
                    caste_props[caste_col] = row[caste_col]
                else:
                    caste_props[caste_col] = 0
                    
            # Income proportions
            income_props = {
                'income_low_ratio': row['income_low_ratio'],
                'income_middle_ratio': row['income_middle_ratio'], 
                'income_high_ratio': row['income_high_ratio']
            }
            
            # Store booth metadata for caching
            self.booth_metadata[booth_idx] = {
                'booth_id': booth_id,
                'N_i': N_i,
                'T_i': T_i,
                't_i': t_i,
                'p_i': party_shares
            }
            
            # IMPLEMENTATION OF YOUR SUGGESTION: Precompute normalized Hindu-only caste distribution
            hindu_caste_keys = [k for k in caste_props.keys() if k.startswith('Caste_') and k not in ('Caste_No_caste_system_Ratio',)]
            den_hindu = sum(caste_props.get(k, 0.0) for k in hindu_caste_keys)
            caste_given_hindu = {k: (caste_props.get(k, 0.0) / den_hindu if den_hindu > 0 else 1.0/max(len(hindu_caste_keys), 1)) for k in hindu_caste_keys}
            
            # IMPLEMENTATION OF YOUR SUGGESTION: Generate cells with proper caste × religion logic
            cells_for_booth = []
            for age_col in self.demographic_axes['age']:
                for religion_col in self.demographic_axes['religion']:
                    if religion_col == 'Religion_Hindu_Ratio':
                        # Hindus: iterate Hindu caste categories (exclude No_caste_system)
                        for caste_col in hindu_caste_keys:
                            for income_col in self.demographic_axes['income']:
                                caste_factor = caste_given_hindu.get(caste_col, 0.0)
                                n_ic = (N_i * age_props[age_col] * religion_props[religion_col] *
                                       caste_factor * income_props[income_col])
                                
                                cells_for_booth.append({
                                    'n_ic': n_ic,
                                    'age_cat': age_col.replace('_Ratio', ''),
                                    'religion_cat': religion_col.replace('_Ratio', ''),
                                    'caste_cat': caste_col.replace('_Ratio', ''),
                                    'income_cat': income_col.replace('_ratio', '')
                                })
                    else:
                        # Non-Hindus: only "No caste system" with factor 1.0
                        caste_col = 'Caste_No_caste_system_Ratio'
                        for income_col in self.demographic_axes['income']:
                            n_ic = (N_i * age_props[age_col] * religion_props[religion_col] *
                                   1.0 * income_props[income_col])
                            
                            cells_for_booth.append({
                                'n_ic': n_ic,
                                'age_cat': age_col.replace('_Ratio', ''),
                                'religion_cat': religion_col.replace('_Ratio', ''),
                                'caste_cat': caste_col.replace('_Ratio', ''),
                                'income_cat': income_col.replace('_ratio', '')
                            })
            
            # Tiny cell handling with renormalization
            valid_cells = [cell for cell in cells_for_booth if cell['n_ic'] >= 0.5]
            dropped_count = len(cells_for_booth) - len(valid_cells)
            
            if valid_cells:
                # Renormalize remaining cells to preserve ∑n_ic ≈ N_i
                total_weight = sum(cell['n_ic'] for cell in valid_cells)
                if total_weight > 0:
                    renorm_factor = N_i / total_weight
                    for cell in valid_cells:
                        cell['n_ic'] *= renorm_factor
                
                # IMPLEMENTATION OF YOUR SUGGESTION: Safety checks
                total_weight_after = sum(c['n_ic'] for c in valid_cells)
                if abs(total_weight_after - N_i) > 1e-6:
                    self.correction_log.append(f"Booth {booth_id}: cell mass {total_weight_after:.3f} != N_i {N_i:.3f} after renorm")
                
                if dropped_count > 0:
                    self.correction_log.append(f"Booth {booth_id}: dropped {dropped_count} tiny cells, renormalized {len(valid_cells)} cells")
                
                # Create final cell data
                for cell in valid_cells:
                    # Create feature dictionary for this cell
                    cell_features = {
                        'age_category': cell['age_cat'],
                        'religion_category': cell['religion_cat'],
                        'caste_category': cell['caste_cat'],
                        'income_category': cell['income_cat'],
                        'economic_category': row['economic_category'],
                        'locality': row['Locality'],
                        # Continuous covariates (will be standardized separately)
                        'land_rate_per_sqm': row['land_rate_per_sqm'],
                        'construction_cost_per_sqm': row['construction_cost_per_sqm'],
                        'total_population': row['TotalPop'],
                        'male_female_ratio': row.get('MaleToFemaleRatio', 1.0)
                    }
                    
                    # Store cell data
                    cell_data = {
                        'booth_id': booth_id,
                        'booth_idx': booth_idx,
                        'cell_weight': cell['n_ic'],
                        'features': cell_features,
                        'turnout_rate': t_i,
                        'party_shares': party_shares
                    }
                    
                    cell_idx = len(cells_data)
                    cells_data.append(cell_data)
                    booth_cell_indices[booth_idx].append(cell_idx)
                    booth_weights[booth_idx].append(cell['n_ic'])
        
        # Cache booth→indices and booth→weights for efficient training
        self.booth_to_cell_indices = dict(booth_cell_indices)
        self.booth_to_weights = dict(booth_weights)
        
        print(f"Created {len(cells_data)} cells from {len(df)} booths")
        if self.correction_log:
            print(f"Applied {len(self.correction_log)} corrections during processing")
        
        return cells_data
    
    def prepare_features(self, cells_data, fit_transform=True):
        """
        FIXED: Prepare features with separate handling of categorical and continuous
        Only fit on training data to prevent data leakage
        """
        print(f"Preparing features (fit_transform={fit_transform})...")
        
        # Extract feature dictionaries (categorical)
        categorical_features = []
        continuous_features = []
        cell_weights = []
        booth_indices = []
        
        for cell in cells_data:
            # Categorical features (to be one-hot encoded, NOT standardized)
            cat_features = {
                'age': cell['features']['age_category'],
                'religion': cell['features']['religion_category'], 
                'caste': cell['features']['caste_category'],
                'income': cell['features']['income_category'],
                'economic': cell['features']['economic_category'],
                'locality': cell['features']['locality']
            }
            categorical_features.append(cat_features)
            
            # Continuous features (to be standardized separately)
            cont_features = [
                cell['features']['land_rate_per_sqm'],
                cell['features']['construction_cost_per_sqm'],
                cell['features']['total_population'],
                cell['features']['male_female_ratio']
            ]
            continuous_features.append(cont_features)
            
            # Targets and weights
            cell_weights.append(cell['cell_weight'])
            booth_indices.append(cell['booth_idx'])
        
        # One-hot encode categorical features (no standardization)
        if fit_transform:
            categorical_encoded = self.vectorizer.fit_transform(categorical_features)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Must fit transformers before calling transform")
            categorical_encoded = self.vectorizer.transform(categorical_features)
        
        # Standardize ONLY continuous features
        continuous_features = np.array(continuous_features)
        if fit_transform:
            continuous_standardized = self.scaler.fit_transform(continuous_features)
        else:
            continuous_standardized = self.scaler.transform(continuous_features)
        
        # Combine features (categorical + standardized continuous)
        features = np.hstack([categorical_encoded, continuous_standardized])
        
        # Get feature names with fallback for older sklearn versions
        try:
            categorical_feature_names = list(self.vectorizer.get_feature_names_out())
        except AttributeError:
            categorical_feature_names = list(self.vectorizer.feature_names_)
        
        return {
            'features': torch.FloatTensor(features),
            'cell_weights': torch.FloatTensor(cell_weights),
            'booth_indices': torch.LongTensor(booth_indices),
            'feature_names': categorical_feature_names + ['land_rate', 'construction_cost', 'population', 'male_female_ratio']
        }

def save_booth_features_vs_turnout(combined_df, all_data, feature_names, assembly_name="Electoral"):
    """
    NEW: Create Excel file tracking booth-level turnout rates vs MODEL INPUT FEATURES
    Uses the actual transformed features (one-hot encoded + scaled) that the model sees
    """
    print(f"\n📊 Creating booth features vs turnout analysis (using model input features)...")
    
    # Convert tensors to numpy for easier manipulation
    features_array = all_data['features'].cpu().numpy()
    booth_indices_array = all_data['booth_indices'].cpu().numpy()
    cell_weights_array = all_data['cell_weights'].cpu().numpy()
    
    # Create booth ID to index mapping from combined_df
    booth_id_to_idx = {}
    for booth_id in combined_df['booth_id'].unique():
        booth_row = combined_df[combined_df['booth_id'] == booth_id].iloc[0]
        booth_id_to_idx[booth_row['booth_idx']] = booth_id
    
    # Aggregate cell-level features to booth level (weighted by cell population)
    unique_booth_indices = np.unique(booth_indices_array)
    booth_feature_data = []
    
    for booth_idx in unique_booth_indices:
        # Get all cells belonging to this booth
        cell_mask = booth_indices_array == booth_idx
        booth_cells_features = features_array[cell_mask]
        booth_cells_weights = cell_weights_array[cell_mask]
        
        # Compute weighted average of features for this booth
        total_weight = booth_cells_weights.sum()
        if total_weight > 0:
            weighted_features = np.average(booth_cells_features, axis=0, weights=booth_cells_weights)
        else:
            weighted_features = booth_cells_features.mean(axis=0)
        
        # Get booth information from combined_df
        booth_id = booth_id_to_idx.get(booth_idx, f"Unknown_{booth_idx}")
        booth_row = combined_df[combined_df['booth_id'] == booth_id].iloc[0]
        
        # Calculate actual turnout rate
        actual_turnout_rate = booth_row['Total_Polled'] / booth_row['TotalPop'] if booth_row['TotalPop'] > 0 else 0
        
        # Base booth information
        row_data = {
            'booth_id': booth_id,
            'booth_idx': booth_idx,
            'part_no': booth_row['PartNo'],
            'assembly_name': booth_row['AssemblyName'],
            'year': booth_row['Year'],
            'total_registered': round(booth_row['TotalPop'], 3),
            'total_polled': round(booth_row['Total_Polled'], 3),
            'turnout_rate': round(actual_turnout_rate, 3)
        }
        
        # Add all model features (weighted average across cells in booth)
        for i, feature_name in enumerate(feature_names):
            row_data[f'feature_{feature_name}'] = round(float(weighted_features[i]), 6)
        
        booth_feature_data.append(row_data)
    
    # Create DataFrame
    booth_features_df = pd.DataFrame(booth_feature_data)
    
    # Calculate correlation with turnout rate for all model features
    correlation_data = []
    
    # Get all feature columns (those starting with 'feature_')
    feature_cols = [col for col in booth_features_df.columns if col.startswith('feature_')]
    
    for col in feature_cols:
        if booth_features_df[col].dtype in ['float64', 'int64']:
            correlation = booth_features_df['turnout_rate'].corr(booth_features_df[col])
            
            # Clean feature name for display
            display_name = col.replace('feature_', '')
            
            correlation_data.append({
                'feature': display_name,
                'correlation_with_turnout': round(correlation, 4),
                'abs_correlation': round(abs(correlation), 4),
                'mean_value': round(booth_features_df[col].mean(), 6),
                'std_value': round(booth_features_df[col].std(), 6),
                'min_value': round(booth_features_df[col].min(), 6),
                'max_value': round(booth_features_df[col].max(), 6),
                'feature_type': 'categorical_dummy' if display_name not in ['land_rate', 'construction_cost', 'population', 'male_female_ratio'] else 'continuous'
            })
    
    # Sort by absolute correlation
    correlation_df = pd.DataFrame(correlation_data).sort_values('abs_correlation', ascending=False)
    
    # Separate categorical dummy features from continuous features for summary
    categorical_features_df = correlation_df[correlation_df['feature_type'] == 'categorical_dummy']
    continuous_features_df = correlation_df[correlation_df['feature_type'] == 'continuous']
    
    # Create summary statistics by year
    year_summary = []
    for year in booth_features_df['year'].unique():
        subset = booth_features_df[booth_features_df['year'] == year]
        year_summary.append({
            'year': str(year),
            'booth_count': len(subset),
            'avg_turnout': round(subset['turnout_rate'].mean(), 3),
            'std_turnout': round(subset['turnout_rate'].std(), 3),
            'min_turnout': round(subset['turnout_rate'].min(), 3),
            'max_turnout': round(subset['turnout_rate'].max(), 3)
        })
    
    year_summary_df = pd.DataFrame(year_summary)
    
    # Create top/bottom booths by turnout
    top_bottom_data = []
    
    # Top 10 turnout booths
    top_10 = booth_features_df.nlargest(10, 'turnout_rate')
    for idx, row in top_10.iterrows():
        top_bottom_data.append({
            'rank_type': 'Top 10',
            'booth_id': row['booth_id'],
            'part_no': row['part_no'],
            'turnout_rate': round(row['turnout_rate'], 3),
            'total_registered': round(row['total_registered'], 0)
        })
    
    # Bottom 10 turnout booths
    bottom_10 = booth_features_df.nsmallest(10, 'turnout_rate')
    for idx, row in bottom_10.iterrows():
        top_bottom_data.append({
            'rank_type': 'Bottom 10',
            'booth_id': row['booth_id'],
            'part_no': row['part_no'],
            'turnout_rate': round(row['turnout_rate'], 3),
            'total_registered': round(row['total_registered'], 0)
        })
    
    top_bottom_df = pd.DataFrame(top_bottom_data)
    
    # Save to Excel with multiple sheets
    filename = f'{assembly_name}_booth_features_vs_turnout.xlsx'
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        booth_features_df.to_excel(writer, sheet_name='Booth_Features_Turnout', index=False)
        correlation_df.to_excel(writer, sheet_name='All_Feature_Correlations', index=False)
        categorical_features_df.to_excel(writer, sheet_name='Categorical_Feature_Corr', index=False)
        continuous_features_df.to_excel(writer, sheet_name='Continuous_Feature_Corr', index=False)
        year_summary_df.to_excel(writer, sheet_name='Year_Summary', index=False)
        top_bottom_df.to_excel(writer, sheet_name='Top_Bottom_Booths', index=False)
    
    print(f"✅ Booth features vs turnout saved to '{filename}' with {len(booth_features_df)} booths")
    print(f"   - {len(feature_cols)} model features tracked ({len(categorical_features_df)} categorical dummies + {len(continuous_features_df)} continuous)")
    print(f"   - Top 5 correlated features with turnout:")
    for i, row in correlation_df.head(5).iterrows():
        print(f"     • {row['feature']}: {row['correlation_with_turnout']:.4f} ({row['feature_type']})")
    
    return booth_features_df, correlation_df

def save_cells_to_excel(cells_data, filename='demographic_cells_data.xlsx'):
    """
    NEW: Save all constructed cells to Excel for analysis and verification
    """
    print(f"\n📊 Saving {len(cells_data)} cells to {filename}...")
    
    # Prepare cells data for export
    cells_export_data = []
    
    for i, cell in enumerate(cells_data):
        row = {
            'cell_id': i,
            'booth_id': cell['booth_id'],
            'booth_idx': cell['booth_idx'],
            'cell_weight': round(cell['cell_weight'], 3),
            'age_category': cell['features']['age_category'],
            'religion_category': cell['features']['religion_category'],
            'caste_category': cell['features']['caste_category'],
            'income_category': cell['features']['income_category'],
            'economic_category': cell['features']['economic_category'],
            'locality': cell['features']['locality'],
            'land_rate_per_sqm': round(cell['features']['land_rate_per_sqm'], 3),
            'construction_cost_per_sqm': round(cell['features']['construction_cost_per_sqm'], 3),
            'total_population': round(cell['features']['total_population'], 3),
            'male_female_ratio': round(cell['features']['male_female_ratio'], 3),
            'observed_turnout_rate': round(cell['turnout_rate'], 3)
        }
        
        # Add party shares
        for party in ['BJP', 'Congress', 'AAP', 'Others', 'NOTA']:
            if party in cell['party_shares']:
                row[f'observed_{party}_share'] = round(cell['party_shares'][party], 3)
            else:
                row[f'observed_{party}_share'] = 0.0
        
        cells_export_data.append(row)
    
    # Create DataFrame and save
    cells_df = pd.DataFrame(cells_export_data)
    
    # Create summary statistics
    summary_data = []
    
    # Cell count by booth
    booth_cell_counts = cells_df.groupby('booth_id').size()
    summary_data.append({
        'Metric': 'Total_Cells',
        'Value': len(cells_data),
        'Description': 'Total number of demographic cells created'
    })
    summary_data.append({
        'Metric': 'Average_Cells_Per_Booth',
        'Value': round(booth_cell_counts.mean(), 2),
        'Description': 'Average number of cells per booth'
    })
    summary_data.append({
        'Metric': 'Min_Cells_Per_Booth',
        'Value': booth_cell_counts.min(),
        'Description': 'Minimum cells in any booth'
    })
    summary_data.append({
        'Metric': 'Max_Cells_Per_Booth',
        'Value': booth_cell_counts.max(),
        'Description': 'Maximum cells in any booth'
    })
    
    # Weight statistics
    weights = cells_df['cell_weight']
    summary_data.append({
        'Metric': 'Average_Cell_Weight',
        'Value': round(weights.mean(), 3),
        'Description': 'Average cell weight (population)'
    })
    summary_data.append({
        'Metric': 'Total_Population_Cells',
        'Value': round(weights.sum(), 0),
        'Description': 'Total population across all cells'
    })
    
    # Category distributions
    for category in ['age_category', 'religion_category', 'caste_category', 'income_category']:
        unique_count = cells_df[category].nunique()
        summary_data.append({
            'Metric': f'Unique_{category}',
            'Value': unique_count,
            'Description': f'Number of unique {category} values'
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Demographic distributions
    demo_distributions = []
    
    for category in ['age_category', 'religion_category', 'caste_category', 'income_category']:
        category_counts = cells_df.groupby(category)['cell_weight'].sum().sort_values(ascending=False)
        total_weight = category_counts.sum()
        
        for cat_value, weight in category_counts.items():
            demo_distributions.append({
                'category_type': category,
                'category_value': cat_value,
                'total_population': round(weight, 0),
                'percentage': round((weight / total_weight) * 100, 2)
            })
    
    demo_dist_df = pd.DataFrame(demo_distributions)
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        cells_df.to_excel(writer, sheet_name='All_Cells', index=False)
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        demo_dist_df.to_excel(writer, sheet_name='Demographic_Distributions', index=False)
        
        # Booth-level aggregation
        booth_summary = cells_df.groupby('booth_id').agg({
            'cell_weight': ['count', 'sum'],
            'observed_turnout_rate': 'first',
            'age_category': 'nunique',
            'religion_category': 'nunique',
            'caste_category': 'nunique',
            'income_category': 'nunique'
        }).round(3)
        
        booth_summary.columns = ['cell_count', 'total_population', 'turnout_rate', 
                               'unique_age_cats', 'unique_religion_cats', 'unique_caste_cats', 'unique_income_cats']
        booth_summary.to_excel(writer, sheet_name='Booth_Level_Summary')
    
    print(f"✅ Cells data saved to {filename} with {len(cells_df)} rows across 4 sheets")
    return cells_df

def split_booths_for_training(combined_df, test_size=0.2, random_state=42):
    """
    Split by booth instead of cells to prevent data leakage
    Handles both single-year and multi-year data
    """
    unique_booth_ids = combined_df['booth_id'].unique()
    
    # Check if we have multiple years for stratification
    unique_years = combined_df['Year'].nunique()
    
    if unique_years > 1:
        # Multi-year data: stratify by year
        booth_years = combined_df.groupby('booth_id')['Year'].first()
        aligned_years = booth_years.loc[unique_booth_ids]
        
        train_booths, val_booths = train_test_split(
            unique_booth_ids, 
            test_size=test_size, 
            random_state=random_state,
            stratify=aligned_years
        )
        print(f"Multi-year split: stratified by year")
    else:
        # Single-year data: random split without stratification
        train_booths, val_booths = train_test_split(
            unique_booth_ids, 
            test_size=test_size, 
            random_state=random_state
        )
        print(f"Single-year split: random split")
    
    return train_booths, val_booths

def create_booth_batches(booth_indices, batch_size=32):
    """Create mini-batches of booths as per guide section 8"""
    unique_booths = torch.unique(booth_indices).tolist()
    np.random.shuffle(unique_booths)
    
    batches = []
    for i in range(0, len(unique_booths), batch_size):
        batch_booths = unique_booths[i:i+batch_size]
        batches.append(batch_booths)
    
    return batches

def compute_loss_booth_batch(model, data_dict, booth_batch, booth_metadata, 
                           lambda_kl=1.0, lambda_T=0.01, lambda_P=0.01, 
                           lambda_bT=0.1, lambda_bP=0.1):
    """
    CORRECTED: Compute loss for a batch of booths with proper weighting
    - Turnout loss weighted by N_i (registered voters)
    - Party loss weighted by T_i (actual votes cast)
    """
    # Convert booth_batch to tensor for proper comparison
    booth_batch_tensor = torch.tensor(booth_batch, device=model.device)
    
    # Filter data to only include cells from booths in this batch
    mask = torch.isin(data_dict['booth_indices'], booth_batch_tensor)
    
    if not mask.any().item():
        return torch.tensor(0.0, device=model.device)
    
    batch_features = data_dict['features'][mask]
    batch_weights = data_dict['cell_weights'][mask]
    batch_booth_indices = data_dict['booth_indices'][mask]
    
    # Forward pass
    turnout_probs, party_probs, party_log_probs = model(batch_features, batch_booth_indices)
    
    # Aggregate by booth
    booth_losses = []
    
    for booth_idx in booth_batch:
        if booth_idx not in booth_metadata:
            continue
            
        booth_mask = batch_booth_indices == booth_idx
        if not booth_mask.any().item():
            continue
        
        # Get booth metadata
        booth_info = booth_metadata[booth_idx]
        N_i = booth_info['N_i']  # Registered voters
        T_i = booth_info['T_i']  # Actual votes cast
        t_i = booth_info['t_i']  # Observed turnout rate
        p_i = torch.FloatTensor([booth_info['p_i'][party] for party in model.party_names]).to(model.device)
        
        # Skip if invalid data
        if N_i <= 0:
            continue
        
        # Aggregate predictions for this booth
        booth_cell_weights = batch_weights[booth_mask]
        booth_turnout_probs = turnout_probs[booth_mask]
        booth_party_probs = party_probs[booth_mask]
        
        # T̂_i = Σ_c n_ic π_ic
        T_hat = torch.sum(booth_cell_weights * booth_turnout_probs)
        
        # V̂_ik = Σ_c n_ic π_ic θ_ic,k
        V_hat = torch.sum(booth_cell_weights.unsqueeze(1) * booth_turnout_probs.unsqueeze(1) * booth_party_probs, dim=0)
        
        # Predicted rates
        t_hat = T_hat / N_i
        p_hat = V_hat / (T_hat + 1e-12)
        
        t_i_tensor = torch.tensor(t_i, device=model.device, dtype=torch.float32)
        
        # CORRECTED: Separate weighting for turnout and party losses
        
        # 1. Turnout loss (weighted by registered voters N_i)
        bce_loss = -(t_i_tensor * torch.log(t_hat + 1e-12) + (1 - t_i_tensor) * torch.log(1 - t_hat + 1e-12))
        weighted_turnout_loss = N_i * bce_loss
        
        # 2. Party loss (weighted by actual votes T_i)
        if T_i > 0:
            kl_loss = torch.sum(p_i * torch.log((p_i + 1e-12) / (p_hat + 1e-12)))
            weighted_party_loss = T_i * lambda_kl * kl_loss
        else:
            weighted_party_loss = torch.tensor(0.0, device=model.device)
        
        # Combined booth loss
        booth_losses.append(weighted_turnout_loss + weighted_party_loss)
    
    # Average loss across booths in batch
    if booth_losses:
        avg_loss = sum(booth_losses) / len(booth_losses)
    else:
        avg_loss = torch.tensor(0.0, device=model.device)
    
    # Regularization terms (unchanged)
    reg_loss = (lambda_T * torch.sum(model.beta_T ** 2) + 
                lambda_P * torch.sum(model.beta_P ** 2) +
                lambda_bT * torch.sum(model.booth_effects_T ** 2) +
                lambda_bP * torch.sum(model.booth_effects_P ** 2))
    
    return avg_loss + reg_loss

def train_model_with_early_stopping(model, train_data, val_data, train_booth_indices, 
                                   val_booth_indices, booth_metadata, epochs=1500, 
                                   lr=0.001, batch_size=32, patience=75, min_delta=1e-4):
    """
    Enhanced training loop with early stopping and learning rate scheduling
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                    patience=25)
    model.party_names = ['BJP', 'Congress', 'AAP', 'Others', 'NOTA']
    
    # Initialize coefficient tracking
    coefficient_history = []
    
    # Move data to device
    for key in ['features', 'cell_weights', 'booth_indices']:
        train_data[key] = train_data[key].to(model.device)
        val_data[key] = val_data[key].to(model.device)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    
    print(f"Starting training with early stopping (max epochs: {epochs}, patience: {patience})...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        
        # Create booth batches for this epoch
        train_batches = create_booth_batches(train_booth_indices, batch_size)
        
        for batch_booths in train_batches:
            optimizer.zero_grad()
            
            batch_loss = compute_loss_booth_batch(model, train_data, batch_booths, booth_metadata)
            if batch_loss.requires_grad:
                batch_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                optimizer.step()
            
            epoch_train_loss += batch_loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_batches) if train_batches else 0.0
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_batches = create_booth_batches(val_booth_indices, batch_size)
            epoch_val_loss = 0.0
            
            for batch_booths in val_batches:
                batch_loss = compute_loss_booth_batch(model, val_data, batch_booths, booth_metadata)
                epoch_val_loss += batch_loss.item()
            
            avg_val_loss = epoch_val_loss / len(val_batches) if val_batches else 0.0
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping logic with minimum delta
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            improvement_msg = f" (New best: {best_val_loss:.4f})"
        else:
            patience_counter += 1
            improvement_msg = f" (No improvement for {patience_counter} epochs)"
        
        # Track coefficients every 10 epochs
        if epoch % 10 == 0:
            # Extract current coefficients
            current_coeffs = {
                'epoch': epoch,
                'train_loss': round(avg_train_loss, 6),
                'val_loss': round(avg_val_loss, 6),
                'learning_rate': round(optimizer.param_groups[0]['lr'], 8),
                'patience_counter': patience_counter
            }
            
            # Add turnout coefficients
            current_coeffs['turnout_intercept'] = round(model.alpha0.item(), 6)
            
            # Add turnout feature coefficients (β_T)
            feature_names = train_data.get('feature_names', [])
            for i, feature_name in enumerate(feature_names):
                if i < len(model.beta_T):
                    current_coeffs[f'turnout_{feature_name}'] = round(model.beta_T[i].item(), 6)
            
            # Add party intercepts
            for j, party in enumerate(model.party_names):
                current_coeffs[f'party_{party}_intercept'] = round(model.gamma0[j].item(), 6)
                
                # Add party feature coefficients (β_P)
                for i, feature_name in enumerate(feature_names):
                    if i < len(model.beta_P) and j < model.beta_P.shape[1]:
                        current_coeffs[f'party_{party}_{feature_name}'] = round(model.beta_P[i, j].item(), 6)
            
            # Add booth effects statistics
            booth_effects_T = model.booth_effects_T.detach().cpu().numpy()
            booth_effects_P = model.booth_effects_P.detach().cpu().numpy()
            
            current_coeffs['booth_effects_T_mean'] = round(float(np.mean(booth_effects_T)), 6)
            current_coeffs['booth_effects_T_std'] = round(float(np.std(booth_effects_T)), 6)
            
            for j, party in enumerate(model.party_names):
                party_effects = booth_effects_P[:, j]
                current_coeffs[f'booth_effects_{party}_mean'] = round(float(np.mean(party_effects)), 6)
                current_coeffs[f'booth_effects_{party}_std'] = round(float(np.std(party_effects)), 6)
            
            coefficient_history.append(current_coeffs)
        
        # Progress reporting
        if epoch % 10 == 0 or patience_counter >= patience - 5:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, "
                  f"LR = {current_lr:.2e}{improvement_msg}")
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
            print(f"Restoring best model state from epoch {best_epoch}")
            model.load_state_dict(best_model_state)
            break
    
    # If training completed without early stopping
    if patience_counter < patience:
        print(f"\nTraining completed all {epochs} epochs")
        if best_model_state is not None:
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
            print(f"Loading best model state from epoch {best_epoch}")
            model.load_state_dict(best_model_state)
    
    return train_losses, val_losses, best_epoch, coefficient_history

def validate_and_calibrate(model, data_dict, booth_indices, booth_metadata, party_names):
    """
    FIXED: Validation and calibration with proper tensor handling
    """
    model.eval()
    results = {
        'turnout_mae': 0.0,
        'turnout_rmse': 0.0,
        'weighted_kl': 0.0,
        'calibration_reliability': [],
        'booth_predictions': {}
    }
    
    with torch.no_grad():
        # Get predictions for all booths
        turnout_probs, party_probs, _ = model(data_dict['features'], data_dict['booth_indices'])
        
        # FIXED: Proper GPU handling with .detach().cpu()
        turnout_probs = turnout_probs.detach().cpu()
        party_probs = party_probs.detach().cpu()
        cell_weights = data_dict['cell_weights'].detach().cpu()
        booth_indices_cpu = data_dict['booth_indices'].detach().cpu()
        
        # Aggregate by booth
        observed_turnout = []
        predicted_turnout = []
        observed_party_shares = []
        predicted_party_shares = []
        booth_weights = []
        
        unique_booths = torch.unique(booth_indices).tolist()
        
        for booth_idx in unique_booths:
            if booth_idx not in booth_metadata:
                continue
                
            booth_mask = booth_indices_cpu == booth_idx
            # FIXED: Use .item() for tensor truthiness
            if not booth_mask.any().item():
                continue
            
            # Get booth info
            booth_info = booth_metadata[booth_idx]
            N_i = booth_info['N_i']
            T_i = booth_info['T_i']
            t_i = booth_info['t_i']
            p_i = np.array([booth_info['p_i'][party] for party in party_names])
            
            if N_i <= 0:
                continue
            
            # Aggregate predictions
            booth_weights_tensor = cell_weights[booth_mask]
            booth_turnout = turnout_probs[booth_mask]
            booth_parties = party_probs[booth_mask]
            
            T_hat = torch.sum(booth_weights_tensor * booth_turnout)
            V_hat = torch.sum(booth_weights_tensor.unsqueeze(1) * booth_turnout.unsqueeze(1) * booth_parties, dim=0)
            
            t_hat = (T_hat / N_i).item() if N_i > 0 else 0.0
            p_hat = (V_hat / (T_hat + 1e-12)).numpy()
            
            observed_turnout.append(t_i)
            predicted_turnout.append(t_hat)
            observed_party_shares.append(p_i)
            predicted_party_shares.append(p_hat)
            booth_weights.append(N_i)
            
            results['booth_predictions'][booth_idx] = {
                'observed_turnout': t_i,
                'predicted_turnout': t_hat,
                'observed_party_shares': p_i,
                'predicted_party_shares': p_hat
            }
        
        # Calculate metrics
        if observed_turnout:
            results['turnout_mae'] = mean_absolute_error(observed_turnout, predicted_turnout, 
                                                       sample_weight=booth_weights)
            results['turnout_rmse'] = np.sqrt(mean_squared_error(observed_turnout, predicted_turnout, 
                                                               sample_weight=booth_weights))
            
            # Weighted KL divergence
            total_weight = sum(booth_weights)
            weighted_kl = 0.0
            for i in range(len(observed_party_shares)):
                p_obs = observed_party_shares[i]
                p_pred = predicted_party_shares[i]
                weight = booth_weights[i]
                
                # KL divergence: sum(p_obs * log(p_obs / p_pred))
                kl = np.sum(p_obs * np.log((p_obs + 1e-12) / (p_pred + 1e-12)))
                weighted_kl += weight * kl
            
            results['weighted_kl'] = weighted_kl / total_weight if total_weight > 0 else 0.0
            
            # Calibration analysis (reliability plot)
            pred_turnout_array = np.array(predicted_turnout)
            obs_turnout_array = np.array(observed_turnout)
            weights_array = np.array(booth_weights)
            
            # Create deciles
            for decile in range(10):
                lower = decile / 10.0
                upper = (decile + 1) / 10.0
                
                # FIXED: Include 1.0 in last bin to prevent dropping predictions equal to exactly 1.0
                if decile < 9:
                    mask = (pred_turnout_array >= lower) & (pred_turnout_array < upper)
                else:
                    mask = (pred_turnout_array >= lower) & (pred_turnout_array <= upper)
                if mask.any():
                    avg_predicted = np.average(pred_turnout_array[mask], weights=weights_array[mask])
                    avg_observed = np.average(obs_turnout_array[mask], weights=weights_array[mask])
                    count = mask.sum()
                    
                    results['calibration_reliability'].append({
                        'decile': decile,
                        'avg_predicted': avg_predicted,
                        'avg_observed': avg_observed,
                        'count': count
                    })
    
    return results

def aggregate_predictions(model, data_dict, booth_metadata, party_names):
    """
    FIXED: Aggregate cell-level predictions to booth level with proper GPU handling
    """
    model.eval()
    booth_predictions = {}
    
    with torch.no_grad():
        turnout_probs, party_probs, _ = model(data_dict['features'], data_dict['booth_indices'])
        
        # FIXED: Proper GPU handling with .detach().cpu()
        turnout_probs = turnout_probs.detach().cpu()
        party_probs = party_probs.detach().cpu()
        cell_weights = data_dict['cell_weights'].detach().cpu()
        booth_indices = data_dict['booth_indices'].detach().cpu()
        
        # Group by booth for aggregation
        unique_booths = torch.unique(booth_indices).tolist()
        
        for booth_idx in unique_booths:
            if booth_idx not in booth_metadata:
                continue
                
            booth_mask = booth_indices == booth_idx
            # FIXED: Use .item() for tensor truthiness
            if not booth_mask.any().item():
                continue
            
            booth_info = booth_metadata[booth_idx]
            N_i = booth_info['N_i']
            
            if N_i <= 0:
                continue
            
            # Aggregate predictions
            booth_weights = cell_weights[booth_mask]
            booth_turnout = turnout_probs[booth_mask]
            booth_parties = party_probs[booth_mask]
            
            # Aggregation formulas from document
            T_hat = torch.sum(booth_weights * booth_turnout).item()
            V_hat = torch.sum(booth_weights.unsqueeze(1) * booth_turnout.unsqueeze(1) * booth_parties, dim=0).numpy()
            
            # Final metrics
            t_hat = T_hat / N_i if N_i > 0 else 0.0
            p_hat = V_hat / T_hat if T_hat > 0 else np.zeros_like(V_hat)

            # Determine top party and alignment classification (core/leaning/swing)
            try:
                p_hat_arr = np.array(p_hat)
                top_idx = int(np.argmax(p_hat_arr)) if p_hat_arr.size > 0 else None
                top_party = party_names[top_idx] if top_idx is not None else None
                p_max = float(np.max(p_hat_arr)) if p_hat_arr.size > 0 else 0.0
            except Exception:
                top_party = None
                p_max = 0.0

            alignment = classify_alignment(p_max)

            booth_predictions[booth_info['booth_id']] = {
                'predicted_turnout_rate': t_hat,
                'predicted_total_votes': T_hat,
                'predicted_party_votes': V_hat,
                'predicted_party_shares': p_hat,
                'total_registered': N_i,
                'top_party': top_party,
                'p_max': p_max,
                'alignment_category': alignment
            }
    
    return booth_predictions

def extract_coefficients(model, feature_names, party_names):
    """
    Extract model coefficients with booth bias analysis per document section 10
    """
    coefficients = {}
    
    # Turnout coefficients (β_T)
    coefficients['turnout'] = {}
    coefficients['turnout']['intercept'] = model.alpha0.item()
    for i, feature_name in enumerate(feature_names):
        coefficients['turnout'][feature_name] = model.beta_T[i].item()
    
    # Party choice coefficients (β_P)
    for j, party in enumerate(party_names):
        coefficients[f'party_{party}'] = {}
        coefficients[f'party_{party}']['intercept'] = model.gamma0[j].item()
        for i, feature_name in enumerate(feature_names):
            coefficients[f'party_{party}'][feature_name] = model.beta_P[i, j].item()
    
    # Booth effects analysis as per guide
    booth_effects_T = model.booth_effects_T.detach().cpu().numpy()
    booth_effects_P = model.booth_effects_P.detach().cpu().numpy()
    
    coefficients['booth_effects_summary'] = {
        'turnout_effects': {
            'mean': float(np.mean(booth_effects_T)),
            'std': float(np.std(booth_effects_T)),
            'min': float(np.min(booth_effects_T)),
            'max': float(np.max(booth_effects_T)),
            'top_5_positive': booth_effects_T.argsort()[-5:][::-1].tolist(),
            'top_5_negative': booth_effects_T.argsort()[:5].tolist()
        },
        'party_effects': {}
    }
    
    for j, party in enumerate(party_names):
        party_effects = booth_effects_P[:, j]
        coefficients['booth_effects_summary']['party_effects'][party] = {
            'mean': float(np.mean(party_effects)),
            'std': float(np.std(party_effects)),
            'min': float(np.min(party_effects)),
            'max': float(np.max(party_effects)),
            'top_5_positive': party_effects.argsort()[-5:][::-1].tolist(),
            'top_5_negative': party_effects.argsort()[:5].tolist()
        }
    
    coefficients['booth_effects_turnout'] = booth_effects_T
    coefficients['booth_effects_party'] = booth_effects_P
    
    return coefficients

def save_coefficient_history_to_excel(coefficient_history, filename='model_coefficients_evolution.xlsx'):
    """
    NEW: Save coefficient evolution during training to Excel file
    """
    if not coefficient_history:
        print("No coefficient history to save.")
        return
    
    print(f"\n📊 Saving coefficient evolution to {filename}...")
    
    # Convert to DataFrame
    coeff_df = pd.DataFrame(coefficient_history)
    
    # Create summary statistics for coefficient changes
    summary_stats = []
    
    if len(coefficient_history) > 1:
        # Calculate coefficient stability metrics
        for col in coeff_df.columns:
            if col not in ['epoch', 'train_loss', 'val_loss', 'learning_rate', 'patience_counter']:
                values = coeff_df[col].values
                
                # Calculate stability metrics
                initial_val = values[0]
                final_val = values[-1]
                total_change = final_val - initial_val
                max_val = np.max(values)
                min_val = np.min(values)
                volatility = np.std(values)
                
                # Categorize coefficient type for better organization
                coeff_type = 'Other'
                if 'turnout_intercept' in col:
                    coeff_type = 'Turnout_Intercept'
                elif 'turnout_' in col:
                    coeff_type = 'Turnout_Feature'
                elif 'party_' in col and '_intercept' in col:
                    coeff_type = 'Party_Intercept'
                elif 'party_' in col and '_feature_' in col:
                    coeff_type = 'Party_Feature'
                elif 'party_' in col and any(feat in col for feat in ['age_category', 'religion_category', 'caste_category', 'income_category', 'economic_category', 'locality']):
                    coeff_type = 'Party_Demographic'
                elif 'booth_effects_' in col:
                    coeff_type = 'Booth_Effects'
                
                summary_stats.append({
                    'coefficient': col,
                    'coefficient_type': coeff_type,
                    'initial_value': round(initial_val, 6),
                    'final_value': round(final_val, 6),
                    'total_change': round(total_change, 6),
                    'min_value': round(min_val, 6),
                    'max_value': round(max_val, 6),
                    'volatility_std': round(volatility, 6),
                    'relative_change_pct': round((total_change / (abs(initial_val) + 1e-8)) * 100, 2),
                    'is_significant_change': abs(total_change) > 0.01  # Flag significant changes
                })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Create convergence analysis
    convergence_data = []
    if len(coefficient_history) > 1:
        epochs = coeff_df['epoch'].values
        train_losses = coeff_df['train_loss'].values
        val_losses = coeff_df['val_loss'].values
        
        # Find best epoch
        best_val_epoch = epochs[np.argmin(val_losses)]
        best_val_loss = np.min(val_losses)
        
        # Calculate loss improvement rates
        loss_improvements = []
        for i in range(1, len(train_losses)):
            train_improvement = train_losses[i-1] - train_losses[i]
            val_improvement = val_losses[i-1] - val_losses[i]
            loss_improvements.append({
                'epoch': epochs[i],
                'train_loss_improvement': round(train_improvement, 6),
                'val_loss_improvement': round(val_improvement, 6),
                'cumulative_train_improvement': round(train_losses[0] - train_losses[i], 6),
                'cumulative_val_improvement': round(val_losses[0] - val_losses[i], 6)
            })
        
        convergence_data.extend(loss_improvements)
        
        # Add convergence summary
        convergence_summary = [{
            'metric': 'Best_Validation_Epoch',
            'value': int(best_val_epoch),
            'description': 'Epoch with lowest validation loss'
        }, {
            'metric': 'Best_Validation_Loss',
            'value': round(best_val_loss, 6),
            'description': 'Lowest validation loss achieved'
        }, {
            'metric': 'Total_Training_Epochs_Recorded',
            'value': len(coefficient_history),
            'description': 'Number of epochs with coefficient records'
        }, {
            'metric': 'Final_Training_Loss',
            'value': round(train_losses[-1], 6),
            'description': 'Training loss at final recorded epoch'
        }, {
            'metric': 'Final_Validation_Loss',
            'value': round(val_losses[-1], 6),
            'description': 'Validation loss at final recorded epoch'
        }]
    
    convergence_df = pd.DataFrame(convergence_data)
    convergence_summary_df = pd.DataFrame(convergence_summary) if len(coefficient_history) > 1 else pd.DataFrame()
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        coeff_df.to_excel(writer, sheet_name='Coefficient_Evolution', index=False)
        if not summary_df.empty:
            summary_df.to_excel(writer, sheet_name='Coefficient_Summary', index=False)
        if not convergence_df.empty:
            convergence_df.to_excel(writer, sheet_name='Loss_Improvements', index=False)
        if not convergence_summary_df.empty:
            convergence_summary_df.to_excel(writer, sheet_name='Convergence_Summary', index=False)
    
    print(f"✅ Coefficient evolution saved to '{filename}' with {len(coeff_df)} epoch records")
    
    # Count different types of coefficients
    coeff_cols = [col for col in coeff_df.columns if col not in ['epoch', 'train_loss', 'val_loss', 'learning_rate', 'patience_counter']]
    intercept_count = len([col for col in coeff_cols if 'intercept' in col])
    feature_count = len([col for col in coeff_cols if any(x in col for x in ['turnout_', 'party_']) and 'intercept' not in col and 'booth_effects' not in col])
    booth_effects_count = len([col for col in coeff_cols if 'booth_effects' in col])
    
    print(f"   - Tracked {len(coeff_cols)} total coefficient parameters:")
    print(f"     • {intercept_count} intercepts (turnout + party)")
    print(f"     • {feature_count} feature coefficients (demographics, locality, etc.)")
    print(f"     • {booth_effects_count} booth effect statistics")
    print(f"   - Recorded every 10 epochs during training")
    
    return coeff_df

def save_results(model, coefficients, booth_predictions, validation_results, 
                feature_names, party_names, processor, training_info=None, coefficient_history=None, assembly_name="Electoral"):
    """
    Save model and results with complete persistence
    """
    # Save complete model data including transformers
    model_data = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_features': len(feature_names),
            'n_parties': len(party_names),
            'n_booths': model.n_booths
        },
        'feature_names': feature_names,
        'party_names': party_names,
        'vectorizer': processor.vectorizer,  # Save fitted vectorizer
        'scaler': processor.scaler,  # Save fitted scaler
        'booth_id_to_idx': processor.booth_id_to_idx,  # Save booth mapping
        'correction_log': processor.correction_log,  # Save correction log
        'training_info': training_info  # Save training details
    }
    
    with open(f'{assembly_name}_trained_electoral_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to '{assembly_name}_trained_electoral_model.pkl'")
    
    # Save coefficients to Excel with enhanced interpretability
    coeff_data = []
    for model_type, coeffs in coefficients.items():
        if model_type.startswith('booth_effects'):
            continue  # Handle booth effects separately
        if model_type == 'booth_effects_summary':
            continue  # Handle summary separately
            
        for feature, coeff_value in coeffs.items():
            coeff_data.append({
                'Model_Type': model_type,
                'Feature': feature,
                'Coefficient': coeff_value,
                'Odds_Ratio': np.exp(coeff_value) if model_type == 'turnout' else None,
                'Interpretation': 'Increases log-odds' if coeff_value > 0 else 'Decreases log-odds'
            })
    
    coeff_df = pd.DataFrame(coeff_data)
    
    # Booth effects summary sheet
    booth_summary_data = []
    summary = coefficients['booth_effects_summary']
    
    # Turnout effects
    booth_summary_data.append({
        'Effect_Type': 'Turnout',
        'Party': 'All',
        'Mean': summary['turnout_effects']['mean'],
        'Std': summary['turnout_effects']['std'],
        'Min': summary['turnout_effects']['min'],
        'Max': summary['turnout_effects']['max']
    })
    
    # Party effects
    for party, effects in summary['party_effects'].items():
        booth_summary_data.append({
            'Effect_Type': 'Party',
            'Party': party,
            'Mean': effects['mean'],
            'Std': effects['std'],
            'Min': effects['min'],
            'Max': effects['max']
        })
    
    booth_summary_df = pd.DataFrame(booth_summary_data)
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter(f'{assembly_name}_model_coefficients.xlsx') as writer:
        coeff_df.to_excel(writer, sheet_name='Coefficients', index=False)
        booth_summary_df.to_excel(writer, sheet_name='Booth_Effects_Summary', index=False)
    
    print(f"Coefficients saved to '{assembly_name}_model_coefficients.xlsx'")
    
    # Save booth-level predictions with validation metrics
    predictions_data = []
    for booth_id, pred in booth_predictions.items():
        row = {
            'booth_id': booth_id,
            'predicted_turnout_rate': round(pred['predicted_turnout_rate'], 3),
            'predicted_total_votes': round(pred['predicted_total_votes'], 3),
            'total_registered': round(pred['total_registered'], 3)
        }

        # Add alignment / swing capture fields if present
        row['top_party'] = pred.get('top_party', None)
        row['p_max'] = round(pred.get('p_max', 0.0), 3)
        row['alignment_category'] = pred.get('alignment_category', 'unknown')
        
        # Add predicted party votes and shares
        for i, party in enumerate(party_names):
            row[f'predicted_{party}_votes'] = round(pred['predicted_party_votes'][i], 3)
            row[f'predicted_{party}_share'] = round(pred['predicted_party_shares'][i], 3)
        
        predictions_data.append(row)
    
    predictions_df = pd.DataFrame(predictions_data)
    
    # Enhanced validation results with training info
    validation_summary_data = [{
        'Metric': 'Turnout_MAE',
        'Value': validation_results['turnout_mae']
    }, {
        'Metric': 'Turnout_RMSE', 
        'Value': validation_results['turnout_rmse']
    }, {
        'Metric': 'Weighted_KL_Divergence',
        'Value': validation_results['weighted_kl']
    }]
    
    # Add training info if available
    if training_info:
        validation_summary_data.extend([
            {'Metric': 'Best_Epoch', 'Value': training_info.get('best_epoch', 'N/A')},
            {'Metric': 'Total_Epochs_Trained', 'Value': training_info.get('total_epochs', 'N/A')},
            {'Metric': 'Early_Stopping_Used', 'Value': training_info.get('early_stopping', False)},
            {'Metric': 'Final_Learning_Rate', 'Value': training_info.get('final_lr', 'N/A')}
        ])
    
    validation_summary = pd.DataFrame(validation_summary_data)
    
    # Calibration results
    calibration_df = pd.DataFrame(validation_results['calibration_reliability'])
    
    # Save predictions with validation
    with pd.ExcelWriter(f'{assembly_name}_booth_level_predictions.xlsx') as writer:
        predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
        validation_summary.to_excel(writer, sheet_name='Validation_Summary', index=False)
        if not calibration_df.empty:
            calibration_df.to_excel(writer, sheet_name='Calibration', index=False)
    
    print(f"Booth-level predictions saved to '{assembly_name}_booth_level_predictions.xlsx'")
    
    # Save correction log
    if processor.correction_log:
        log_df = pd.DataFrame({'Correction': processor.correction_log})
        log_df.to_excel(f'{assembly_name}_data_corrections_log.xlsx', index=False)
        print(f"Data corrections log saved to '{assembly_name}_data_corrections_log.xlsx'")
    
    # NEW: Save coefficient evolution history
    if coefficient_history:
        save_coefficient_history_to_excel(coefficient_history, f'{assembly_name}_model_coefficients_evolution.xlsx')

def create_enhanced_all_booths_predictions(model, all_data, booth_metadata, party_names, 
                                         combined_df, train_booth_ids, val_booth_ids, assembly_name="Electoral"):
    """
    NEW: Create enhanced all booths predictions with actual data and 3 decimal formatting
    """
    print("\n📊 Creating enhanced all booths predictions with actual vs predicted comparison...")
    
    # Get predictions for all booths
    all_booth_predictions = aggregate_predictions(model, all_data, booth_metadata, party_names)
    
    # Create enhanced predictions data with actual values
    enhanced_predictions_data = []
    
    for booth_id, pred in all_booth_predictions.items():
        # Get actual data from combined_df
        booth_actual = combined_df[combined_df['booth_id'] == booth_id].iloc[0]
        
        # Base row with booth info
        row = {
            'booth_id': booth_id,
            'part_no': booth_actual['PartNo'],
            'assembly_name': booth_actual['AssemblyName'],
            'locality': booth_actual['Locality'],
            'year': booth_actual['Year'],
            'data_split': 'train' if booth_id in train_booth_ids else 'validation',
            'total_registered': round(pred['total_registered'], 3),
            
            # Turnout comparison
            'actual_total_polled': round(booth_actual['Total_Polled'], 3),
            'predicted_total_votes': round(pred['predicted_total_votes'], 3),
            'actual_turnout_rate': round(booth_actual['Total_Polled'] / booth_actual['TotalPop'], 3),
            'predicted_turnout_rate': round(pred['predicted_turnout_rate'], 3),
            'turnout_error': round(pred['predicted_turnout_rate'] - (booth_actual['Total_Polled'] / booth_actual['TotalPop']), 3),
            'turnout_abs_error': round(abs(pred['predicted_turnout_rate'] - (booth_actual['Total_Polled'] / booth_actual['TotalPop'])), 3),
        }

        # Include swing/core/leaning classification from predictions if available
        row['top_party'] = pred.get('top_party', None)
        row['p_max'] = round(pred.get('p_max', 0.0), 3)
        row['alignment_category'] = pred.get('alignment_category', 'unknown')
        
        # Party-wise comparison with actual vs predicted
        for i, party in enumerate(party_names):
            actual_share = booth_actual[f'{party}_Ratio']
            predicted_share = pred['predicted_party_shares'][i]
            actual_votes = booth_actual['Total_Polled'] * actual_share
            predicted_votes = pred['predicted_party_votes'][i]
            
            row[f'actual_{party}_share'] = round(actual_share, 3)
            row[f'predicted_{party}_share'] = round(predicted_share, 3)
            row[f'actual_{party}_votes'] = round(actual_votes, 3)
            row[f'predicted_{party}_votes'] = round(predicted_votes, 3)
            row[f'{party}_share_error'] = round(predicted_share - actual_share, 3)
            row[f'{party}_share_abs_error'] = round(abs(predicted_share - actual_share), 3)
        
        # Economic and demographic info
        row.update({
            'economic_category': booth_actual['economic_category'],
            'land_rate_per_sqm': round(booth_actual['land_rate_per_sqm'], 3),
            'construction_cost_per_sqm': round(booth_actual['construction_cost_per_sqm'], 3),
            'male_female_ratio': round(booth_actual.get('MaleToFemaleRatio', 1.0), 3),
        })
        
        enhanced_predictions_data.append(row)
    
    # Create DataFrame
    enhanced_df = pd.DataFrame(enhanced_predictions_data)
    
    # Calculate summary statistics
    summary_stats = []
    
    # Overall turnout metrics
    train_mask = enhanced_df['data_split'] == 'train'
    val_mask = enhanced_df['data_split'] == 'validation'
    
    for split_name, mask in [('Overall', slice(None)), ('Training', train_mask), ('Validation', val_mask)]:
        if isinstance(mask, slice) or mask.any():
            subset = enhanced_df[mask] if not isinstance(mask, slice) else enhanced_df
            
            summary_stats.append({
                'data_split': split_name,
                'metric': 'Turnout_MAE',
                'value': round(subset['turnout_abs_error'].mean(), 4),
                'description': 'Mean Absolute Error for turnout rate'
            })
            
            summary_stats.append({
                'data_split': split_name,
                'metric': 'Turnout_RMSE',
                'value': round(np.sqrt(subset['turnout_error'].pow(2).mean()), 4),
                'description': 'Root Mean Square Error for turnout rate'
            })
            
            # Party-wise MAE
            for party in party_names:
                party_mae = subset[f'{party}_share_abs_error'].mean()
                summary_stats.append({
                    'data_split': split_name,
                    'metric': f'{party}_Share_MAE',
                    'value': round(party_mae, 4),
                    'description': f'Mean Absolute Error for {party} vote share'
                })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Error analysis by categories
    error_analysis = []
    
    # By economic category
    for econ_cat in enhanced_df['economic_category'].unique():
        if pd.notna(econ_cat):
            subset = enhanced_df[enhanced_df['economic_category'] == econ_cat]
            error_analysis.append({
                'category_type': 'Economic',
                'category_value': econ_cat,
                'booth_count': len(subset),
                'avg_turnout_error': round(subset['turnout_abs_error'].mean(), 4),
                'avg_total_party_error': round(sum(subset[f'{party}_share_abs_error'].mean() for party in party_names), 4)
            })
    
    # By data split
    for split in ['train', 'validation']:
        subset = enhanced_df[enhanced_df['data_split'] == split]
        if len(subset) > 0:
            error_analysis.append({
                'category_type': 'Data_Split',
                'category_value': split,
                'booth_count': len(subset),
                'avg_turnout_error': round(subset['turnout_abs_error'].mean(), 4),
                'avg_total_party_error': round(sum(subset[f'{party}_share_abs_error'].mean() for party in party_names), 4)
            })
    
    # By year
    for year in enhanced_df['year'].unique():
        subset = enhanced_df[enhanced_df['year'] == year]
        error_analysis.append({
            'category_type': 'Year',
            'category_value': str(year),
            'booth_count': len(subset),
            'avg_turnout_error': round(subset['turnout_abs_error'].mean(), 4),
            'avg_total_party_error': round(sum(subset[f'{party}_share_abs_error'].mean() for party in party_names), 4)
        })
    
    error_analysis_df = pd.DataFrame(error_analysis)
    
    # Save enhanced file
    filename = f'{assembly_name}_enhanced_all_booth_predictions.xlsx'
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        enhanced_df.to_excel(writer, sheet_name='All_Predictions_Enhanced', index=False)
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        error_analysis_df.to_excel(writer, sheet_name='Error_Analysis', index=False)
        
        # Create separate sheets for actual vs predicted comparison
        comparison_data = []
        for _, row in enhanced_df.iterrows():
            # Turnout comparison
            comparison_data.append({
                'booth_id': row['booth_id'],
                'metric_type': 'Turnout_Rate',
                'actual': row['actual_turnout_rate'],
                'predicted': row['predicted_turnout_rate'],
                'error': row['turnout_error'],
                'abs_error': row['turnout_abs_error'],
                'data_split': row['data_split']
            })
            
            # Party comparisons
            for party in party_names:
                comparison_data.append({
                    'booth_id': row['booth_id'],
                    'metric_type': f'{party}_Share',
                    'actual': row[f'actual_{party}_share'],
                    'predicted': row[f'predicted_{party}_share'],
                    'error': row[f'{party}_share_error'],
                    'abs_error': row[f'{party}_share_abs_error'],
                    'data_split': row['data_split']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_excel(writer, sheet_name='Actual_vs_Predicted', index=False)
    
    print(f"✅ Enhanced predictions saved to '{filename}' with {len(enhanced_df)} booths across 4 sheets")
    print(f"   - Included actual vs predicted comparison with 3 decimal precision")
    print(f"   - Added error analysis by economic category, data split, and year")
    print(f"   - Training booths: {train_mask.sum()}, Validation booths: {val_mask.sum()}")
    
    return enhanced_df, summary_df, error_analysis_df

# ENHANCED: Main execution function with early stopping option
def main(file_2020=None, file_2025='Assembly_Madipur_2025_Election.xlsx', 
         use_early_stopping=True, max_epochs=1500, patience=75):
    """
    ENHANCED: Main execution function with optional 2020 data
    Args:
        file_2020: Path to 2020 data file (optional, can be None)
        file_2025: Path to 2025 data file (required)
    """
    print("=== ENHANCED ELECTORAL PREDICTION MODEL WITH CELLS EXPORT ===")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize processor
    processor = ElectoralDataProcessor()
    
    # Load and combine datasets (validation happens inside)
    print("\n1. Loading datasets...")
    combined_df = processor.load_and_combine_data(file_2020, file_2025)
    
    # Extract assembly name for file prefixing
    assembly_name = combined_df['AssemblyName'].iloc[0].replace(' ', '_').replace('-', '_')
    print(f"Assembly Name: {assembly_name}")
    
    # Split by booth
    print("\n2. Splitting booths for training...")
    train_booth_ids, val_booth_ids = split_booths_for_training(combined_df, test_size=0.2)
    print(f"Train booths: {len(train_booth_ids)}, Validation booths: {len(val_booth_ids)}")
    
    # Construct cells
    print("\n3. Constructing demographic cells...")
    cells_data = processor.construct_cells(combined_df)
    
    # NEW: Save cells data to Excel
    print("\n4. Saving cells data to Excel...")
    cells_df = save_cells_to_excel(cells_data, f'{assembly_name}_demographic_cells_data.xlsx')
    
    # Split cells
    print("\n5. Splitting cells based on booth membership...")
    train_cells = [cell for cell in cells_data if cell['booth_id'] in train_booth_ids]
    val_cells = [cell for cell in cells_data if cell['booth_id'] in val_booth_ids]
    
    print(f"Train cells: {len(train_cells)}, Validation cells: {len(val_cells)}")
    
    # Prepare features
    print("\n6. Preparing features (fitting on training data only)...")
    train_data = processor.prepare_features(train_cells, fit_transform=True)
    val_data = processor.prepare_features(val_cells, fit_transform=False)
    
    train_booth_indices = torch.unique(train_data['booth_indices'])
    val_booth_indices = torch.unique(val_data['booth_indices'])
    
    # Initialize model
    print("\n7. Initializing model...")
    n_features = train_data['features'].shape[1]
    n_parties = len(processor.party_names)
    n_booths = len(processor.booth_id_to_idx)
    
    print(f"Model architecture: {n_features} features, {n_parties} parties, {n_booths} booths")
    
    model = AggregateVotingModel(n_features, n_parties, n_booths, device=device)
    model.to(device)
    
    # Train model with early stopping option
    print(f"\n8. Training model (Early Stopping: {use_early_stopping})...")
    
    if use_early_stopping:
        train_losses, val_losses, best_epoch, coefficient_history = train_model_with_early_stopping(
            model, train_data, val_data, 
            train_booth_indices, val_booth_indices,
            processor.booth_metadata, 
            epochs=max_epochs, 
            patience=patience
        )
        training_info = {
            'early_stopping': True,
            'best_epoch': best_epoch,
            'total_epochs': len(train_losses),
            'max_epochs': max_epochs,
            'patience': patience
        }
    else:
        # Use original training function for compatibility
        train_losses, val_losses, coefficient_history = train_model_simple(
            model, train_data, val_data, 
            train_booth_indices, val_booth_indices,
            processor.booth_metadata, 
            epochs=max_epochs
        )
        training_info = {
            'early_stopping': False,
            'total_epochs': len(train_losses),
            'max_epochs': max_epochs
        }
    
    # Validation
    print("\n9. Validation and calibration...")
    validation_results = validate_and_calibrate(
        model, val_data, val_booth_indices, 
        processor.booth_metadata, processor.party_names
    )
    
    # Generate predictions
    print("\n10. Generating predictions...")
    val_booth_predictions = aggregate_predictions(
        model, val_data, processor.booth_metadata, processor.party_names
    )
    
    # Extract coefficients
    print("\n11. Extracting coefficients...")
    coefficients = extract_coefficients(
        model, train_data['feature_names'], processor.party_names
    )
    
    # Save results
    print("\n12. Saving results...")
    save_results(
        model, coefficients, val_booth_predictions, validation_results,
        train_data['feature_names'], processor.party_names, processor, training_info, coefficient_history, assembly_name
    )
    
    # Generate enhanced predictions for all booths
    print("\n13. Generating enhanced predictions for all booths...")
    all_data = processor.prepare_features(cells_data, fit_transform=False)
    
    enhanced_df, summary_df, error_analysis_df = create_enhanced_all_booths_predictions(
        model, all_data, processor.booth_metadata, processor.party_names,
        combined_df, train_booth_ids, val_booth_ids, assembly_name
    )
    
    # Generate booth features vs turnout analysis
    print("\n14. Generating booth features vs turnout analysis...")
    booth_features_df, correlation_df = save_booth_features_vs_turnout(
        combined_df, all_data, train_data['feature_names'], assembly_name
    )
    
    print("\n=== MODEL TRAINING COMPLETE ===")
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"Turnout MAE: {validation_results['turnout_mae']:.4f}")
    print(f"Turnout RMSE: {validation_results['turnout_rmse']:.4f}")
    print(f"Weighted KL Divergence: {validation_results['weighted_kl']:.4f}")
    print(f"Total Cells Created: {len(cells_data)}")
    print(f"Features: {len(train_data['feature_names'])}")
    print(f"Train Booths: {len(train_booth_indices)}, Val Booths: {len(val_booth_indices)}")
    
    if use_early_stopping and 'best_epoch' in training_info:
        print(f"Early stopping: Best model from epoch {training_info['best_epoch']}")
    
    if processor.correction_log:
        print(f"Applied {len(processor.correction_log)} data corrections")
        print(f"Check '{assembly_name}_data_corrections_log.xlsx' for details")
    
    print("\n📁 FILES CREATED:")
    print(f"  - {assembly_name}_demographic_cells_data.xlsx (All constructed cells with analysis)")
    print(f"  - {assembly_name}_enhanced_all_booth_predictions.xlsx (Predictions vs actual with 3 decimals)")
    print(f"  - {assembly_name}_booth_features_vs_turnout.xlsx (Booth-level features vs turnout rates)")
    print(f"  - {assembly_name}_trained_electoral_model.pkl (Complete trained model)")
    print(f"  - {assembly_name}_model_coefficients.xlsx (Model coefficients and effects)")
    print(f"  - {assembly_name}_model_coefficients_evolution.xlsx (Coefficient evolution every 10 epochs)")
    print(f"  - {assembly_name}_booth_level_predictions.xlsx (Validation predictions)")
    print(f"  - {assembly_name}_data_corrections_log.xlsx (Data processing corrections)")
    
    return model, coefficients, val_booth_predictions, validation_results, training_info

def train_model_simple(model, train_data, val_data, train_booth_indices, val_booth_indices, 
                      booth_metadata, epochs=800, lr=0.001, batch_size=32):
    """
    Simple training loop (for backward compatibility when not using early stopping)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.party_names = ['BJP', 'Congress', 'AAP', 'Others', 'NOTA']
    
    # Initialize coefficient tracking
    coefficient_history = []
    
    # Move data to device
    for key in ['features', 'cell_weights', 'booth_indices']:
        train_data[key] = train_data[key].to(model.device)
        val_data[key] = val_data[key].to(model.device)
    
    train_losses = []
    val_losses = []
    
    print("Starting simple training...")
    for epoch in range(epochs):
        # Training with booth-based mini-batching
        model.train()
        epoch_train_loss = 0.0
        
        # Create booth batches
        train_batches = create_booth_batches(train_booth_indices, batch_size)
        
        for batch_booths in train_batches:
            optimizer.zero_grad()
            
            batch_loss = compute_loss_booth_batch(model, train_data, batch_booths, booth_metadata)
            if batch_loss.requires_grad:
                batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                optimizer.step()
            
            epoch_train_loss += batch_loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_batches) if train_batches else 0.0
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_batches = create_booth_batches(val_booth_indices, batch_size)
            epoch_val_loss = 0.0
            
            for batch_booths in val_batches:
                batch_loss = compute_loss_booth_batch(model, val_data, batch_booths, booth_metadata)
                epoch_val_loss += batch_loss.item()
            
            avg_val_loss = epoch_val_loss / len(val_batches) if val_batches else 0.0
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Track coefficients every 10 epochs
        if epoch % 10 == 0:
            # Extract current coefficients
            current_coeffs = {
                'epoch': epoch,
                'train_loss': round(avg_train_loss, 6),
                'val_loss': round(avg_val_loss, 6),
                'learning_rate': round(optimizer.param_groups[0]['lr'], 8),
                'patience_counter': 0  # No early stopping in simple mode
            }
            
            # Add turnout coefficients
            current_coeffs['turnout_intercept'] = round(model.alpha0.item(), 6)
            
            # Add turnout feature coefficients (β_T) - need to get feature names from training data
            # We'll use a simple index-based approach since we don't have access to feature_names here
            for i in range(len(model.beta_T)):
                current_coeffs[f'turnout_feature_{i}'] = round(model.beta_T[i].item(), 6)
            
            # Add party intercepts
            for j, party in enumerate(model.party_names):
                current_coeffs[f'party_{party}_intercept'] = round(model.gamma0[j].item(), 6)
                
                # Add party feature coefficients (β_P)
                for i in range(model.beta_P.shape[0]):
                    if j < model.beta_P.shape[1]:
                        current_coeffs[f'party_{party}_feature_{i}'] = round(model.beta_P[i, j].item(), 6)
            
            # Add booth effects statistics
            booth_effects_T = model.booth_effects_T.detach().cpu().numpy()
            booth_effects_P = model.booth_effects_P.detach().cpu().numpy()
            
            current_coeffs['booth_effects_T_mean'] = round(float(np.mean(booth_effects_T)), 6)
            current_coeffs['booth_effects_T_std'] = round(float(np.std(booth_effects_T)), 6)
            
            for j, party in enumerate(model.party_names):
                party_effects = booth_effects_P[:, j]
                current_coeffs[f'booth_effects_{party}_mean'] = round(float(np.mean(party_effects)), 6)
                current_coeffs[f'booth_effects_{party}_std'] = round(float(np.std(party_effects)), 6)
            
            coefficient_history.append(current_coeffs)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    return train_losses, val_losses, coefficient_history

# Debug function to test column access
def debug_column_access(df, processor):
    """
    Debug function to test column access patterns
    """
    print("=== DEBUGGING COLUMN ACCESS ===")
    
    # Test a sample row
    sample_row = df.iloc[0]
    
    print("\n🔍 Testing Age columns:")
    for age_col in processor.demographic_axes['age']:
        if age_col in sample_row:
            print(f"  ✅ {age_col}: {sample_row[age_col]}")
        else:
            print(f"  ❌ {age_col}: NOT FOUND")
    
    print("\n🔍 Testing Religion columns:")
    for religion_col in processor.demographic_axes['religion']:
        if religion_col in sample_row:
            print(f"  ✅ {religion_col}: {sample_row[religion_col]}")
        else:
            print(f"  ❌ {religion_col}: NOT FOUND")
    
    print("\n🔍 Testing Caste columns:")
    for caste_col in processor.demographic_axes['caste']:
        if caste_col in sample_row:
            print(f"  ✅ {caste_col}: {sample_row[caste_col]}")
        else:
            print(f"  ❌ {caste_col}: NOT FOUND")
    
    print("\n🔍 Testing Party columns:")
    for party in processor.party_names:
        party_col = f'{party}_Ratio'
        if party_col in sample_row:
            print(f"  ✅ {party_col}: {sample_row[party_col]}")
        else:
            print(f"  ❌ {party_col}: NOT FOUND")

# Quick validation test before running full model
def quick_validation_test():
    """
    Quick test to validate column names before running the full model
    """
    print("=== QUICK COLUMN VALIDATION TEST ===")
    
    try:
        # Load 2025 data
        df_2025 = pd.read_excel('Aggregate Data/Assembly40_NewDelhi_2025.xlsx')
        
        # Initialize processor
        processor = ElectoralDataProcessor()
        
        print("\n📋 Expected vs Actual Column Names:")
        
        print("\n🔍 AGE COLUMNS:")
        for expected_col in processor.demographic_axes['age']:
            if expected_col in df_2025.columns:
                print(f"  ✅ {expected_col} - FOUND")
            else:
                print(f"  ❌ {expected_col} - NOT FOUND")
        
        print("\n🔍 RELIGION COLUMNS:")
        for expected_col in processor.demographic_axes['religion']:
            if expected_col in df_2025.columns:
                print(f"  ✅ {expected_col} - FOUND")
            else:
                print(f"  ❌ {expected_col} - NOT FOUND")
                
        print("\n🔍 CASTE COLUMNS:")
        for expected_col in processor.demographic_axes['caste']:
            if expected_col in df_2025.columns:
                print(f"  ✅ {expected_col} - FOUND")
            else:
                print(f"  ❌ {expected_col} - NOT FOUND")
        
        print("\n🔍 PARTY COLUMNS:")
        for party in processor.party_names:
            party_col = f'{party}_Ratio'
            if party_col in df_2025.columns:
                print(f"  ✅ {party_col} - FOUND")
            else:
                print(f"  ❌ {party_col} - NOT FOUND")
        
        print(f"\n📊 Total columns in dataset: {len(df_2025.columns)}")
        print("✅ Column validation test completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return False

if __name__ == "__main__":
    # Optional: Run quick validation test first
    print("Running quick validation test...")
    if quick_validation_test():
        print("\n" + "="*60)
        
        # OPTION 1: With both 2020 and 2025 data
        model, coefficients, predictions, validation, training_info = main(
            file_2020=None,
            file_2025='Aggregate Data/Assembly40_NewDelhi_2025.xlsx',
            use_early_stopping=True, 
            max_epochs=2500, 
            patience=100
        )
        
        # OPTION 2: With only 2025 data (for assemblies with single year)
        # model, coefficients, predictions, validation, training_info = main(
        #     file_2020=None,
        #     file_2025='Assembly_YourAssembly_2025_Election.xlsx',
        #     use_early_stopping=True, 
        #     max_epochs=1500, 
        #     patience=75
        # )
    else:
        print("❌ Validation test failed. Please check your dataset columns.")

# SUMMARY OF NEW FEATURES:
"""
✅ NEW: save_cells_to_excel() function
  - Exports all constructed demographic cells to Excel
  - Includes 4 sheets: All_Cells, Summary_Statistics, Demographic_Distributions, Booth_Level_Summary
  - Shows cell weights, demographic categories, and booth-level aggregations
  - Provides detailed analysis of cell construction process

✅ NEW: create_enhanced_all_booths_predictions() function
  - Creates comprehensive actual vs predicted comparison
  - All values formatted to 3 decimal places as requested
  - Includes error analysis by economic category, data split, and year
  - 4 sheets: All_Predictions_Enhanced, Summary_Statistics, Error_Analysis, Actual_vs_Predicted
  - Shows booth-level metrics, turnout comparison, and party-wise analysis

✅ NEW: save_booth_features_vs_turnout() function
  - Creates comprehensive booth-level features vs turnout rate analysis
  - Tracks all continuous features (land rate, construction cost, etc.) and demographic ratios
  - Includes 4 sheets: Booth_Features_Turnout, Feature_Correlations, Categorical_Summary, Top_Bottom_Booths
  - Calculates correlation of each feature with turnout rate
  - Shows top/bottom 10 booths by turnout
  - Provides summary statistics by economic category and year
  - Enables detailed analysis of which features most influence voter turnout

✅ ENHANCED: All prediction values formatted to 3 decimals
  - Applied consistent rounding to 3 decimal places throughout
  - Enhanced readability and professional presentation
  - Maintains precision while improving usability

✅ ENHANCED: Comprehensive file output with Assembly Name Prefix
  - {assembly_name}_demographic_cells_data.xlsx: Complete cells analysis
  - {assembly_name}_enhanced_all_booth_predictions.xlsx: Actual vs predicted with errors
  - {assembly_name}_booth_features_vs_turnout.xlsx: Booth features vs turnout analysis (NEW)
  - {assembly_name}_trained_electoral_model.pkl: Complete trained model
  - {assembly_name}_model_coefficients.xlsx: Model coefficients and effects
  - {assembly_name}_model_coefficients_evolution.xlsx: Coefficient evolution tracking
  - {assembly_name}_booth_level_predictions.xlsx: Validation predictions
  - {assembly_name}_data_corrections_log.xlsx: Data processing corrections
  - All files now include assembly name prefix for easy identification

✅ NEW: Coefficient Evolution Tracking
  - model_coefficients_evolution.xlsx: Tracks coefficients every 10 epochs
  - Includes 4 sheets: Coefficient_Evolution, Coefficient_Summary, Loss_Improvements, Convergence_Summary
  - Shows coefficient stability, volatility, and convergence patterns
  - Monitors turnout intercepts, party intercepts, and booth effects statistics
  - Available in both early stopping and simple training modes

The model now provides complete transparency into the cell construction process,
detailed actual vs predicted comparisons, and comprehensive coefficient evolution
tracking for better model evaluation, interpretability, and training insights.
""" 