##Enhanced Electoral Prediction Model with Candidate-Aware Aggregated Likelihood Model (CALM)

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

class CandidateAwareVotingModel(nn.Module):
    """
    Candidate-Aware Aggregated Likelihood Model (CALM) for Electoral Prediction
    Extends the base model with candidate affinity features.

    IMPORTANT: Candidate affinity is 3D (age, religion, caste). Income only appears on voter side.
    """
    
    def __init__(self, n_features, n_parties, n_booths, device='cpu', use_candidate_affinity=True, 
                 use_mobilization=False, rho_age=0.8):
        super(CandidateAwareVotingModel, self).__init__()
        
        self.device = device
        self.use_candidate_affinity = use_candidate_affinity
        self.use_mobilization = use_mobilization
        self.rho_age = rho_age
        
        # Model parameters as per document notation
        # Turnout model: π_ic = σ(α0 + b_i^(T) + x_ic^T β_T + ξ * m_ic)
        self.alpha0 = nn.Parameter(torch.randn(1, device=device))  # Global turnout intercept
        self.beta_T = nn.Parameter(torch.randn(n_features, device=device))  # Turnout coefficients
        self.booth_effects_T = nn.Parameter(torch.randn(n_booths, device=device))  # Booth effects for turnout
        
        # Candidate mobilization parameter (optional)
        if self.use_mobilization:
            # Scalar parameters to ensure clean shapes in mobilization path
            self.xi = nn.Parameter(torch.randn((), device=device))  # Mobilization effect (scalar)
            self.w_A = nn.Parameter(torch.randn((), device=device))  # Age mobilization weight (scalar)
            self.w_R = nn.Parameter(torch.randn((), device=device))  # Religion mobilization weight (scalar)
            self.w_C = nn.Parameter(torch.randn((), device=device))  # Caste mobilization weight (scalar)
            # NOTE: No candidate income axis; mobilization uses only age, religion, caste
        
        # Party choice model: θ_ic = softmax(γ0 + b_i^(P) + x_ic^T β_P + η^T g_ic,k)
        self.gamma0 = nn.Parameter(torch.randn(n_parties, device=device))  # Global party intercepts
        self.beta_P = nn.Parameter(torch.randn(n_features, n_parties, device=device))  # Party coefficients
        self.booth_effects_P = nn.Parameter(torch.randn(n_booths, n_parties, device=device))  # Booth effects for parties
        
        # Candidate affinity parameters
        if self.use_candidate_affinity:
            # 3D affinity weights: [age, religion, caste]
            self.eta = nn.Parameter(torch.randn(3, device=device))
        
        self.n_parties = n_parties
        self.n_booths = n_booths
        
        # Note: age kernel for candidate affinity is managed by CandidateProcessor
        
    def forward(self, features, booth_indices, candidate_affinities=None):
        """
        Forward pass implementing CALM framework
        """
        batch_size = features.shape[0]
        
        # Ensure inputs are on correct device
        features = features.to(self.device)
        booth_indices = booth_indices.to(self.device)
        
        # Turnout model: π_ic = σ(α0 + b_i^(T) + x_ic^T β_T + ξ * m_ic)
        turnout_logits = (self.alpha0 + 
                         self.booth_effects_T[booth_indices] + 
                         torch.matmul(features, self.beta_T))
        
        # Add mobilization effect if enabled
        if self.use_mobilization and candidate_affinities is not None:
            mobilization = self._compute_mobilization(candidate_affinities, booth_indices)
            turnout_logits = turnout_logits + self.xi * mobilization
            
        turnout_probs = torch.sigmoid(turnout_logits)
        
        # Party choice model: θ_ic = softmax(γ0 + b_i^(P) + x_ic^T β_P + η^T g_ic,k)
        party_logits = (self.gamma0.unsqueeze(0).expand(batch_size, -1) + 
                       self.booth_effects_P[booth_indices] + 
                       torch.matmul(features, self.beta_P))
        
        # Add candidate affinity effects if enabled
        if self.use_candidate_affinity and candidate_affinities is not None:
            # candidate_affinities shape: [batch_size, n_parties, 3]
            # eta shape: [3]
            # We need to compute eta^T * g_ic,k for each (cell, party) pair
            
            affinity_effects = torch.zeros(batch_size, self.n_parties, device=self.device)
            for party_idx in range(self.n_parties):
                # g_ic,k shape: [batch_size, 3], eta shape: [3]
                affinity_effects[:, party_idx] = torch.matmul(candidate_affinities[:, party_idx, :], self.eta)
            
            party_logits = party_logits + affinity_effects
        
        # Use log_softmax for numerical stability
        party_log_probs = torch.log_softmax(party_logits, dim=1)
        party_probs = torch.exp(party_log_probs)
        
        return turnout_probs, party_probs, party_log_probs
    
    def _compute_mobilization(self, candidate_affinities, booth_indices):
        """Compute mobilization effect m_ic = Σ_k w_ik * s_ic,k using 3D affinity (age, religion, caste)"""
        # candidate_affinities shape: [batch_size, n_parties, 3]
        # Compute s_ic,k = w_A * g_age + w_R * g_rel + w_C * g_caste
        
        weights = torch.stack([self.w_A, self.w_R, self.w_C])  # [3]
        s_ick = torch.matmul(candidate_affinities, weights)  # [batch_size, n_parties]
        
        # For simplicity, use uniform party weights (can be enhanced with booth-specific weights)
        w_ik = torch.ones_like(s_ick) / self.n_parties  # Uniform weights
        
        m_ic = torch.sum(w_ik * s_ick, dim=1)  # [batch_size]
        return m_ic

class CandidateProcessor:
    """
    Process candidate information and compute affinity features.

    Candidate affinity is 3D (age, religion, caste). Income only appears on voter side.
    """
    
    def __init__(self, rho_age=0.8):
        self.rho_age = rho_age
        self.candidates = {}  # Will store candidate info by party
        
        # Candidate age tertile bin edges (min, q1, q2, max). Will be computed from data.
        self.age_bin_edges = None  # e.g., [min_age, q1, q2, max_age]
        
        # Define mappings from candidate traits to model categories
        self.religion_mapping = {
            'Hindu': 0, 'Muslim': 1, 'Christian': 2, 'Sikh': 3, 'Buddhist': 4, 'Jain': 5
        }
        
        self.caste_mapping = {
            'Brahmin': 0, 'Kshatriya': 1, 'Vaishya': 2, 'OBC': 3, 'SC': 4, 'ST': 5, 'No_caste_system': 6
        }
        
    def fit_candidate_age_bins(self, ages, k=3):
        """Set fixed age bins: Young (18-35), Middle-aged (36-60), Senior (61+)."""
        self.age_bin_edges = [18.0, 35.0, 60.0, 100.0]
        print(f"Using fixed age bin edges (Young/Middle/Senior): {self.age_bin_edges}")
        return self.age_bin_edges

    def _age_to_meta_bin(self, age):
        """Map a numeric age to a meta-bin index 0,1,2 using learned edges."""
        if self.age_bin_edges is None:
            # Default tertile-ish edges
            self.age_bin_edges = [18.0, 42.0, 56.0, 100.0]
        a0, a1, a2, a3 = self.age_bin_edges
        if age <= a1:
            return 0
        elif age <= a2:
            return 1
        else:
            return 2

    def _cell_age_category_to_rep_age(self, age_cat):
        """Map cell age category string to a representative age for meta-binning."""
        reps = {
            'Age_18-25': 21.5,
            'Age_26-35': 30.5,
            'Age_36-45': 40.5,
            'Age_46-60': 53.0,
            'Age_60+': 65.0
        }
        return reps.get(age_cat, 40.0)

    def add_candidate(self, party, age, religion, caste=None, income_class='Middle'):
        """Add candidate information for a party (income ignored in CALM affinity)."""
        # Map to meta age bin lazily once edges are known; store raw age here
        candidate_info = {
            'age': float(age) if age is not None else None,
            'religion': self.religion_mapping.get(religion, -1),
            'caste': self.caste_mapping.get(caste, -1) if caste else -1
        }
        self.candidates[party] = candidate_info
        print(f"Added candidate for {party}: Age {age}, Religion {religion}, Caste {caste} (Income ignored)")
        
    def _build_age_kernel(self):
        """3x3 kernel: same=1, adjacent=rho_age, non-adj=0."""
        K = 3
        A = torch.eye(K)
        for i in range(K-1):
            A[i, i+1] = self.rho_age
            A[i+1, i] = self.rho_age
        return A
        
    def compute_candidate_affinities(self, cells_data, party_names):
        """
        Compute candidate affinity features g_ic,k for all cells and parties.
        Returns tensor of shape [num_cells, num_parties, 3] for [age, religion, caste].
        """
        num_cells = len(cells_data)
        num_parties = len(party_names)
        affinities = torch.zeros(num_cells, num_parties, 3)

        # Ensure age bins are fitted from candidate ages
        if self.age_bin_edges is None:
            ages = [info.get('age') for info in self.candidates.values()]
            self.fit_candidate_age_bins(ages, k=3)
        age_kernel = self._build_age_kernel()
        
        print(f"\n🔍 Computing candidate affinities for {num_cells} cells (3D: age, religion, caste)...")
        
        for cell_idx, cell in enumerate(cells_data):
            # Cell age meta-bin
            cell_age_cat = cell['features']['age_category']
            rep_age = self._cell_age_category_to_rep_age(cell_age_cat)
            cell_age_bin = self._age_to_meta_bin(rep_age)
            
            # Cell religion and caste
            cell_religion = self._get_cell_religion(cell)
            cell_caste = self._get_cell_caste(cell)
            
            for party_idx, party in enumerate(party_names):
                if party not in self.candidates:
                    affinities[cell_idx, party_idx] = torch.tensor([0.0, 0.0, 0.0])
                    continue
                cand = self.candidates[party]
                cand_age_bin = self._age_to_meta_bin(cand['age']) if cand.get('age') is not None else -1
                
                # 1) Age affinity via kernel
                if cand_age_bin >= 0:
                    age_affinity = age_kernel[cell_age_bin, cand_age_bin].item()
                else:
                    age_affinity = 0.0
                
                # 2) Religion affinity (hard match)
                religion_affinity = 1.0 if (cell_religion >= 0 and cand['religion'] >= 0 and cell_religion == cand['religion']) else 0.0
                
                # 3) Caste affinity (Hindu-only hard match)
                if (cell_religion == 0 and cand['religion'] == 0 and cell_caste >= 0 and cand['caste'] >= 0):
                    caste_affinity = 1.0 if cell_caste == cand['caste'] else 0.0
                else:
                    caste_affinity = 0.0
                
                affinities[cell_idx, party_idx] = torch.tensor([age_affinity, religion_affinity, caste_affinity])
        
        print("✅ Candidate affinities computed (3D). Income is not included on the candidate side.")
        
        # Debug sample removed: Using actuals summary after training instead
        
        return affinities
    
    def _get_cell_religion(self, cell):
        religion_cat = cell['features']['religion_category']
        religion_mapping = {
            'Religion_Hindu': 0, 'Religion_Muslim': 1, 'Religion_Christian': 2,
            'Religion_Sikh': 3, 'Religion_Buddhist': 4, 'Religion_Jain': 5
        }
        return religion_mapping.get(religion_cat, -1)
    
    def _get_cell_caste(self, cell):
        caste_cat = cell['features']['caste_category']
        caste_mapping = {
            'Caste_Brahmin': 0, 'Caste_Kshatriya': 1, 'Caste_Vaishya': 2,
            'Caste_Obc': 3, 'Caste_Sc': 4, 'Caste_St': 5, 'Caste_No_caste_system': 6
        }
        return caste_mapping.get(caste_cat, -1)

class ElectoralDataProcessor:
    """
    Enhanced data processor with candidate-aware features
    """
    
    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=False)
        self.scaler = StandardScaler()
        self.booth_id_to_idx = {}
        self.party_names = ['BJP', 'Congress', 'AAP', 'Others', 'NOTA']
        self.correction_log = []
        self.is_fitted = False
        
        # Define demographic axes with correct column names
        self.demographic_axes = {
            'age': ['Age_18-25_Ratio', 'Age_26-35_Ratio', 'Age_36-45_Ratio', 'Age_46-60_Ratio', 'Age_60+_Ratio'],
            'religion': ['Religion_Buddhist_Ratio', 'Religion_Christian_Ratio', 'Religion_Hindu_Ratio',
                        'Religion_Jain_Ratio', 'Religion_Muslim_Ratio', 'Religion_Sikh_Ratio'],
            'caste': ['Caste_Brahmin_Ratio', 'Caste_Kshatriya_Ratio', 'Caste_Obc_Ratio',
                     'Caste_Sc_Ratio', 'Caste_St_Ratio', 'Caste_Vaishya_Ratio', 'Caste_No_caste_system_Ratio'],
            'income': ['income_low_ratio', 'income_middle_ratio', 'income_high_ratio']
        }
        
        # Initialize candidate processor
        self.candidate_processor = CandidateProcessor()
    
    def add_candidate_info(self, party, age, religion, caste=None, income_class='Middle'):
        """Add candidate information (income_class is accepted but ignored for affinity)."""
        self.candidate_processor.add_candidate(party, age, religion, caste, income_class)
    
    def validate_data_columns(self, df):
        """Validate required columns exist in dataset"""
        required_base_columns = [
            'PartNo', 'AssemblyName', 'TotalPop', 'Total_Polled',
            'BJP_Ratio', 'Congress_Ratio', 'AAP_Ratio', 'Others_Ratio', 'NOTA_Ratio'
        ]
        
        missing_base = [col for col in required_base_columns if col not in df.columns]
        if missing_base:
            raise ValueError(f"Missing required base columns: {missing_base}")
        
        # Check demographic columns
        all_demo_cols = []
        for axis_cols in self.demographic_axes.values():
            all_demo_cols.extend(axis_cols)
        
        missing_demo = [col for col in all_demo_cols if col not in df.columns]
        if missing_demo:
            print(f"Warning: Missing demographic columns: {missing_demo}")
        
        # Check for unknown categories
        unknown_cols = [col for col in df.columns if 'Unknown' in col and col.endswith('_Ratio')]
        if unknown_cols:
            print(f"Found Unknown categories that will be suppressed: {unknown_cols}")
        
        return True
    
    def load_and_combine_data(self, file_2020, file_2025):
        """Load and validate datasets"""
        dfs = []
        
        if file_2020:
            print(f"Loading 2020 data: {file_2020}")
            df_2020 = pd.read_excel(file_2020)
            df_2020['Year'] = 2020
            dfs.append(df_2020)
        
        print(f"Loading 2025 data: {file_2025}")
        df_2025 = pd.read_excel(file_2025)
        df_2025['Year'] = 2025
        dfs.append(df_2025)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Standardize column names for compatibility
        combined_df['booth_id'] = combined_df['PartNo'].astype(str)
        
        # Calculate turnout percentage from Total_Polled and TotalPop
        combined_df['Turnout%'] = (combined_df['Total_Polled'] / combined_df['TotalPop']) * 100
        
        # Convert ratio columns to vote counts if needed
        # Assuming the ratios are proportions that sum to 1 for each party
        combined_df['BJP'] = combined_df['BJP_Ratio'] * combined_df['Total_Polled']
        combined_df['Congress'] = combined_df['Congress_Ratio'] * combined_df['Total_Polled']
        combined_df['AAP'] = combined_df['AAP_Ratio'] * combined_df['Total_Polled']
        combined_df['Others'] = combined_df['Others_Ratio'] * combined_df['Total_Polled']
        combined_df['NOTA'] = combined_df['NOTA_Ratio'] * combined_df['Total_Polled']
        
        # Validate data
        self.validate_data_columns(combined_df)
        
        # Clean and process data
        combined_df = self._clean_and_validate_data(combined_df)
        combined_df = self._create_income_categories(combined_df)
        combined_df = self._suppress_unknown_and_renormalize(combined_df)
        
        return combined_df
    
    def _suppress_unknown_and_renormalize(self, df):
        """Suppress Unknown categories and renormalize"""
        # Religion: suppress Unknown and renormalize
        rel_known = [c for c in df.columns if c.startswith('Religion_') and c.endswith('_Ratio') and 'Unknown' not in c]
        rel_unknown = 'Religion_Unknown_Ratio'
        
        if rel_unknown in df.columns:
            print(f"Suppressing {rel_unknown} and renormalizing...")
            df[rel_unknown] = 0.0
            denom = df[rel_known].sum(axis=1)
            
            uniform = 1.0 / max(len(rel_known), 1)
            for c in rel_known:
                df.loc[denom > 0, c] = df.loc[denom > 0, c] / denom[denom > 0]
                df.loc[denom == 0, c] = uniform
                
            self.correction_log.append("Suppressed Religion_Unknown and renormalized")
        
        # Caste: suppress Unknown
        caste_unknown = 'Caste_Unknown_Ratio'
        if caste_unknown in df.columns:
            print(f"Suppressing {caste_unknown}...")
            df[caste_unknown] = 0.0
            self.correction_log.append("Suppressed Caste_Unknown")
        
        return df
    
    def _create_income_categories(self, df):
        """Create income categories from economic data"""
        economic_to_income_mapping = {
            'LOW INCOME AREAS': 'income_low',
            'LOWER MIDDLE CLASS': 'income_middle', 
            'MIDDLE CLASS': 'income_middle',
            'UPPER MIDDLE CLASS': 'income_high',
            'PREMIUM AREAS': 'income_high'
        }
        
        df['income_low_ratio'] = 0.0
        df['income_middle_ratio'] = 0.0  
        df['income_high_ratio'] = 0.0
        
        for economic_cat, income_cat in economic_to_income_mapping.items():
            mask = df['economic_category'] == economic_cat
            df.loc[mask, f'{income_cat}_ratio'] = 1.0
        
        # Handle missing categories with fallback
        income_cols = ['income_low_ratio', 'income_middle_ratio', 'income_high_ratio']
        no_category_mask = df[income_cols].sum(axis=1) == 0
        
        if no_category_mask.any():
            # Use middle class as default
            df.loc[no_category_mask, 'income_middle_ratio'] = 1.0
            self.correction_log.append(f"Set {no_category_mask.sum()} booths to middle income as default")
        
        return df
    
    def _clean_and_validate_data(self, df):
        """Clean and validate the dataset"""
        # Basic cleaning
        df['booth_id'] = df['booth_id'].astype(str)
        df['Turnout%'] = pd.to_numeric(df['Turnout%'], errors='coerce')
        
        # Ensure booth IDs are unique per year
        duplicates = df.groupby(['booth_id', 'Year']).size()
        if duplicates.max() > 1:
            print(f"Warning: Found duplicate booth_id-Year combinations")
        
        # Fill missing values for continuous variables
        continuous_cols = ['land_rate_per_sqm', 'construction_cost_per_sqm', 'MaleToFemaleRatio']
        for col in continuous_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Handle missing economic categories
        if 'economic_category' not in df.columns and 'economic_category_code' in df.columns:
            # Map economic codes to categories if needed
            df['economic_category'] = df['economic_category_code']
        
        return df
    
    def construct_cells(self, df):
        """
        Construct demographic cells for each booth
        """
        print("Constructing demographic cells...")
        
        cells_data = []
        booth_cell_indices = defaultdict(list)
        booth_weights = defaultdict(list)
        
        # Create booth index mapping
        unique_booth_ids = df['booth_id'].unique()
        self.booth_id_to_idx = {booth_id: idx for idx, booth_id in enumerate(unique_booth_ids)}
        
        # Cache booth metadata
        self.booth_metadata = {}
        
        for _, row in df.iterrows():
            booth_id = str(row['booth_id'])
            booth_idx = self.booth_id_to_idx[booth_id]
            
            # Cache booth metadata
            self.booth_metadata[booth_idx] = {
                'booth_id': booth_id,
                'registered_voters': row['TotalPop'],
                'actual_turnout': row['Turnout%'] / 100.0,
                'actual_votes': {
                    'BJP': row['BJP'], 'Congress': row['Congress'], 'AAP': row['AAP'],
                    'Others': row['Others'], 'NOTA': row['NOTA']
                }
            }
            
            # Calculate basic values
            N_i = row['TotalPop']
            t_i = row['Turnout%'] / 100.0
            T_i = N_i * t_i
            
            party_votes = [row['BJP'], row['Congress'], row['AAP'], row['Others'], row['NOTA']]
            total_votes = sum(party_votes)
            party_shares = [v / max(total_votes, 1) for v in party_votes]
            
            # Get demographic ratios
            age_ratios = [row.get(col, 0) for col in self.demographic_axes['age']]
            religion_ratios = [row.get(col, 0) for col in self.demographic_axes['religion']]
            caste_ratios = [row.get(col, 0) for col in self.demographic_axes['caste']]
            income_ratios = [row.get(col, 0) for col in self.demographic_axes['income']]
            
            # Create cells for each demographic combination
            age_cats = ['Age_18-25', 'Age_26-35', 'Age_36-45', 'Age_46-60', 'Age_60+']
            religion_cats = ['Religion_Buddhist', 'Religion_Christian', 'Religion_Hindu', 
                           'Religion_Jain', 'Religion_Muslim', 'Religion_Sikh']
            caste_cats = ['Caste_Brahmin', 'Caste_Kshatriya', 'Caste_Obc', 'Caste_Sc', 
                         'Caste_St', 'Caste_Vaishya', 'Caste_No_caste_system']
            income_cats = ['income_low', 'income_middle', 'income_high']
            
            valid_cells = []
            
            for age_idx, age_cat in enumerate(age_cats):
                for rel_idx, rel_cat in enumerate(religion_cats):
                    for caste_idx, caste_cat in enumerate(caste_cats):
                        for inc_idx, inc_cat in enumerate(income_cats):
                            # Calculate cell weight
                            p_age = age_ratios[age_idx]
                            p_rel = religion_ratios[rel_idx]
                            p_inc = income_ratios[inc_idx]
                            
                            # Caste only applies to Hindus
                            if rel_cat == 'Religion_Hindu':
                                p_caste = caste_ratios[caste_idx]
                            else:
                                if caste_cat != 'Caste_No_caste_system':
                                    continue  # Skip non-applicable caste combinations
                                p_caste = 1.0
                            
                            n_ic = N_i * p_age * p_rel * p_caste * p_inc
                            
                            if n_ic < 0.1:  # Skip tiny cells
                                continue
                            
                            cell = {
                                'age_cat': age_cat,
                                'religion_cat': rel_cat,
                                'caste_cat': caste_cat,
                                'income_cat': inc_cat,
                                'n_ic': n_ic
                            }
                            valid_cells.append(cell)
            
            # Renormalize cells
            total_weight = sum(cell['n_ic'] for cell in valid_cells)
            if total_weight > 0:
                renorm_factor = N_i / total_weight
                for cell in valid_cells:
                    cell['n_ic'] *= renorm_factor
            
            # Create final cell data
            for cell in valid_cells:
                cell_features = {
                    'age_category': cell['age_cat'],
                    'religion_category': cell['religion_cat'],
                    'caste_category': cell['caste_cat'],
                    'income_category': cell['income_cat'],
                    'economic_category': row['economic_category'],
                    'locality': row['Locality'],
                    'land_rate_per_sqm': row['land_rate_per_sqm'],
                    'construction_cost_per_sqm': row['construction_cost_per_sqm'],
                    'total_population': row['TotalPop'],
                    'male_female_ratio': row.get('MaleToFemaleRatio', 1.0)
                }
                
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
        
        self.booth_to_cell_indices = dict(booth_cell_indices)
        self.booth_to_weights = dict(booth_weights)
        
        print(f"Created {len(cells_data)} cells from {len(df)} booths")
        return cells_data
    
    def prepare_features(self, cells_data, fit_transform=True):
        """Prepare features with candidate affinity computation"""
        print(f"Preparing features with CALM (fit_transform={fit_transform})...")
        
        # Extract standard features
        categorical_features = []
        continuous_features = []
        cell_weights = []
        booth_indices = []
        
        for cell in cells_data:
            cat_features = {
                'age': cell['features']['age_category'],
                'religion': cell['features']['religion_category'], 
                'caste': cell['features']['caste_category'],
                'income': cell['features']['income_category'],
                'economic': cell['features']['economic_category'],
                'locality': cell['features']['locality']
            }
            categorical_features.append(cat_features)
            
            cont_features = [
                cell['features']['land_rate_per_sqm'],
                cell['features']['construction_cost_per_sqm'],
                cell['features']['total_population'],
                cell['features']['male_female_ratio']
            ]
            continuous_features.append(cont_features)
            
            cell_weights.append(cell['cell_weight'])
            booth_indices.append(cell['booth_idx'])
        
        # One-hot encode categorical features
        if fit_transform:
            categorical_encoded = self.vectorizer.fit_transform(categorical_features)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Must fit on training data first")
            categorical_encoded = self.vectorizer.transform(categorical_features)
        
        # Standardize continuous features
        continuous_array = np.array(continuous_features)
        if fit_transform:
            continuous_standardized = self.scaler.fit_transform(continuous_array)
        else:
            continuous_standardized = self.scaler.transform(continuous_array)
        
        # Combine features
        all_features = np.hstack([categorical_encoded, continuous_standardized])
        feature_names = (list(self.vectorizer.get_feature_names_out()) + 
                        ['land_rate', 'construction_cost', 'population', 'male_female_ratio'])
        
    # Compute candidate affinities
        candidate_affinities = self.candidate_processor.compute_candidate_affinities(cells_data, self.party_names)
        
        return {
            'features': torch.FloatTensor(all_features),
            'cell_weights': torch.FloatTensor(cell_weights),
            'booth_indices': torch.LongTensor(booth_indices),
            'candidate_affinities': candidate_affinities,
            'feature_names': feature_names
        }

def compute_loss_booth_batch_calm(model, data_dict, booth_batch, booth_metadata,
                                 lambda_kl=1.0, lambda_T=0.01, lambda_P=0.01,
                                 lambda_bT=0.1, lambda_bP=0.1, lambda_eta=0.01):
    """
    Compute loss for CALM model with candidate affinity regularization
    """
    booth_batch_tensor = torch.tensor(booth_batch, device=model.device)
    
    mask = torch.isin(data_dict['booth_indices'], booth_batch_tensor)
    
    if not mask.any().item():
        return torch.tensor(0.0, device=model.device, requires_grad=True)
    
    batch_features = data_dict['features'][mask]
    batch_weights = data_dict['cell_weights'][mask]
    batch_booth_indices = data_dict['booth_indices'][mask]
    batch_candidate_affinities = data_dict['candidate_affinities'][mask]
    
    # Forward pass
    turnout_probs, party_probs, party_log_probs = model(
        batch_features, batch_booth_indices, batch_candidate_affinities
    )
    
    # Aggregate by booth
    booth_losses = []
    
    for booth_idx in booth_batch:
        booth_mask = batch_booth_indices == booth_idx
        
        if not booth_mask.any():
            continue
            
        booth_weights = batch_weights[booth_mask]
        booth_turnout = turnout_probs[booth_mask]
        booth_party_probs = party_probs[booth_mask]
        
        # Aggregate predictions
        pred_total_turnout = torch.sum(booth_weights * booth_turnout)
        pred_party_votes = torch.sum(booth_weights.unsqueeze(1) * booth_turnout.unsqueeze(1) * booth_party_probs, dim=0)
        
        # Get actual values
        booth_data = booth_metadata[booth_idx]
        actual_turnout = booth_data['actual_turnout']
        actual_votes = torch.tensor([
            booth_data['actual_votes']['BJP'],
            booth_data['actual_votes']['Congress'], 
            booth_data['actual_votes']['AAP'],
            booth_data['actual_votes']['Others'],
            booth_data['actual_votes']['NOTA']
        ], device=model.device, dtype=torch.float32)
        
        # Turnout loss
        N_i = booth_data['registered_voters']
        pred_turnout_rate = pred_total_turnout / N_i
        turnout_loss = N_i * torch.nn.functional.binary_cross_entropy(
            pred_turnout_rate.clamp(1e-7, 1-1e-7),
            torch.tensor(actual_turnout, device=model.device)
        )
        
        # Party choice loss  
        T_i = actual_votes.sum()
        if T_i > 1e-6:
            actual_party_shares = actual_votes / T_i
            pred_party_shares = pred_party_votes / (pred_party_votes.sum() + 1e-8)
            
            kl_loss = torch.nn.functional.kl_div(
                torch.log(pred_party_shares + 1e-8),
                actual_party_shares,
                reduction='sum'
            )
            party_loss = T_i * kl_loss
        else:
            party_loss = torch.tensor(0.0, device=model.device)
        
        booth_losses.append(turnout_loss + party_loss)
    
    avg_loss = torch.stack(booth_losses).mean() if booth_losses else torch.tensor(0.0, device=model.device, requires_grad=True)
    
    # Regularization
    reg_loss = (lambda_T * torch.sum(model.beta_T ** 2) + 
                lambda_P * torch.sum(model.beta_P ** 2) +
                lambda_bT * torch.sum(model.booth_effects_T ** 2) +
                lambda_bP * torch.sum(model.booth_effects_P ** 2))
    
    # Add candidate affinity regularization
    if model.use_candidate_affinity:
        reg_loss += lambda_eta * torch.sum(model.eta ** 2)
    
    if model.use_mobilization:
        reg_loss += lambda_eta * (torch.sum(model.xi ** 2) + 
                                 torch.sum(model.w_A ** 2) + torch.sum(model.w_R ** 2) +
                                 torch.sum(model.w_C ** 2))
    
    return avg_loss + reg_loss

def train_calm_model(model, train_data, val_data, train_booth_indices, val_booth_indices, 
                    booth_metadata, epochs=1500, lr=0.001, batch_size=32, patience=75,
                    feature_names=None, party_names=None, track_coefficients=True):
    """
    Train CALM model with early stopping and optional coefficient tracking
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25)
    model.party_names = ['BJP', 'Congress', 'AAP', 'Others', 'NOTA']
    
    # Initialize coefficient tracking
    coefficient_history = [] if track_coefficients and feature_names and party_names else None
    
    # Move data to device
    for key in ['features', 'cell_weights', 'booth_indices', 'candidate_affinities']:
        if key in train_data:
            train_data[key] = train_data[key].to(model.device)
            val_data[key] = val_data[key].to(model.device)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    
    print(f"Starting CALM training (epochs: {epochs}, patience: {patience})...")
    
    for epoch in range(epochs):
        model.train()
        
        # Create booth batches
        train_booth_list = train_booth_indices.cpu().numpy().tolist()
        np.random.shuffle(train_booth_list)
        
        epoch_train_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(train_booth_list), batch_size):
            booth_batch = train_booth_list[i:i + batch_size]
            
            optimizer.zero_grad()
            loss = compute_loss_booth_batch_calm(
                model, train_data, booth_batch, booth_metadata
            )
            
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_train_loss += loss.item()
                num_batches += 1
        
        avg_train_loss = epoch_train_loss / max(num_batches, 1)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_booth_list = val_booth_indices.cpu().numpy().tolist()
            val_loss = compute_loss_booth_batch_calm(
                model, val_data, val_booth_list, booth_metadata
            )
            val_loss_item = val_loss.item()
        
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss_item)
        
        # Learning rate scheduling
        scheduler.step(val_loss_item)
        
        # Early stopping
        if val_loss_item < best_val_loss - 1e-4:
            best_val_loss = val_loss_item
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss_item:.4f}, "
                  f"Best = {best_val_loss:.4f} (epoch {best_epoch}), Patience = {patience_counter}")
        
        # Track coefficients every 10 epochs
        if coefficient_history is not None and epoch % 10 == 0:
            epoch_coeffs = extract_calm_coefficients(model, feature_names, party_names)
            epoch_coeffs['epoch'] = epoch
            epoch_coeffs['train_loss'] = avg_train_loss
            epoch_coeffs['val_loss'] = val_loss_item
            epoch_coeffs['learning_rate'] = optimizer.param_groups[0]['lr']
            coefficient_history.append(epoch_coeffs)
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model from epoch {best_epoch}")
    
    training_info = {
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1], 
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'total_epochs': len(train_losses),
        'early_stopped': patience_counter >= patience,
        'coefficient_history': coefficient_history
    }
    
    return train_losses, val_losses, training_info

def extract_calm_coefficients(model, feature_names, party_names):
    """Extract CALM model coefficients including candidate affinity effects"""
    coefficients = {}
    
    # Standard coefficients
    coefficients['turnout'] = {}
    coefficients['turnout']['intercept'] = model.alpha0.item()
    for i, feature_name in enumerate(feature_names):
        coefficients['turnout'][feature_name] = model.beta_T[i].item()
    
    for j, party in enumerate(party_names):
        coefficients[f'party_{party}'] = {}
        coefficients[f'party_{party}']['intercept'] = model.gamma0[j].item()
        for i, feature_name in enumerate(feature_names):
            coefficients[f'party_{party}'][feature_name] = model.beta_P[i, j].item()
    
    # Candidate affinity coefficients
    if model.use_candidate_affinity:
        coefficients['candidate_affinity'] = {
            'age_effect': model.eta[0].item(),
            'religion_effect': model.eta[1].item(), 
            'caste_effect': model.eta[2].item()
        }
        print("Candidate Affinity Effects (3D):")
        print(f"  Age: {model.eta[0].item():.4f}")
        print(f"  Religion: {model.eta[1].item():.4f}")
        print(f"  Caste: {model.eta[2].item():.4f}")
    
    # Mobilization coefficients  
    if model.use_mobilization:
        coefficients['mobilization'] = {
            'mobilization_effect': model.xi.item(),
            'age_mobilization': model.w_A.item(),
            'religion_mobilization': model.w_R.item(),
            'caste_mobilization': model.w_C.item()
        }
    
    # Booth effects analysis
    booth_effects_T = model.booth_effects_T.detach().cpu().numpy()
    booth_effects_P = model.booth_effects_P.detach().cpu().numpy()
    
    coefficients['booth_effects_summary'] = {
        'turnout_effects': {
            'mean': float(np.mean(booth_effects_T)),
            'std': float(np.std(booth_effects_T)),
            'min': float(np.min(booth_effects_T)),
            'max': float(np.max(booth_effects_T))
        },
        'party_effects': {}
    }
    
    for j, party in enumerate(party_names):
        party_effects = booth_effects_P[:, j]
        coefficients['booth_effects_summary']['party_effects'][party] = {
            'mean': float(np.mean(party_effects)),
            'std': float(np.std(party_effects)),
            'min': float(np.min(party_effects)),
            'max': float(np.max(party_effects))
        }
    
    return coefficients

def split_booths_for_training(combined_df, test_size=0.2, random_state=42):
    """Split booths for training and validation"""
    unique_booth_ids = combined_df['booth_id'].unique()
    
    unique_years = combined_df['Year'].nunique()
    
    if unique_years > 1:
        # Stratify by year if multiple years
        year_booth_map = {}
        for _, row in combined_df.iterrows():
            year = row['Year']
            if year not in year_booth_map:
                year_booth_map[year] = []
            if row['booth_id'] not in year_booth_map[year]:
                year_booth_map[year].append(row['booth_id'])
        
        train_booths = []
        val_booths = []
        
        for year, booths in year_booth_map.items():
            year_train, year_val = train_test_split(booths, test_size=test_size, random_state=random_state)
            train_booths.extend(year_train)
            val_booths.extend(year_val)
    else:
        train_booths, val_booths = train_test_split(unique_booth_ids, test_size=test_size, random_state=random_state)
    
    return train_booths, val_booths

def main_calm(file_2020=None, file_2025='Assembly_Madipur_2025_Election.xlsx',
             use_candidate_affinity=True, use_mobilization=False,
             candidate_info=None, max_epochs=1500, patience=75):
    """
    Main function for CALM model training
    
    Args:
        candidate_info: dict with candidate information, e.g.:
        {
            'BJP': {'age': 54, 'religion': 'Hindu', 'caste': 'OBC', 'income_class': 'Middle'},
            'Congress': {'age': 45, 'religion': 'Hindu', 'caste': 'SC', 'income_class': 'Middle'},
            # ... other parties
        }
        Note: income_class is accepted for completeness but ignored in candidate affinity (income only on voter side).
    """
    print("=== CANDIDATE-AWARE AGGREGATED LIKELIHOOD MODEL (CALM) ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize processor
    processor = ElectoralDataProcessor()
    
    # Add candidate information if provided
    if candidate_info:
        print("\nAdding candidate information:")
        for party, info in candidate_info.items():
            processor.add_candidate_info(
                party, info['age'], info['religion'], 
                info.get('caste'), info.get('income_class', 'Middle')
            )
        # Fit candidate age tertile bins across all provided candidates
        all_ages = [info.get('age') for info in candidate_info.values()]
        processor.candidate_processor.fit_candidate_age_bins(all_ages, k=3)
    
    # Load and combine datasets
    print("\n1. Loading datasets...")
    combined_df = processor.load_and_combine_data(file_2020, file_2025)
    
    assembly_name = combined_df['AssemblyName'].iloc[0].replace(' ', '_').replace('-', '_')
    print(f"Assembly Name: {assembly_name}")
    
    # Split by booth
    print("\n2. Splitting booths for training...")
    train_booth_ids, val_booth_ids = split_booths_for_training(combined_df, test_size=0.2)
    print(f"Train booths: {len(train_booth_ids)}, Validation booths: {len(val_booth_ids)}")
    
    # Construct cells
    print("\n3. Constructing demographic cells...")
    cells_data = processor.construct_cells(combined_df)
    
    # Split cells
    print("\n4. Splitting cells based on booth membership...")
    train_cells = [cell for cell in cells_data if cell['booth_id'] in train_booth_ids]
    val_cells = [cell for cell in cells_data if cell['booth_id'] in val_booth_ids]
    
    print(f"Train cells: {len(train_cells)}, Validation cells: {len(val_cells)}")
    
    # Prepare features with candidate affinity
    print("\n5. Preparing features with candidate affinity...")
    train_data = processor.prepare_features(train_cells, fit_transform=True)
    val_data = processor.prepare_features(val_cells, fit_transform=False)
    
    train_booth_indices = torch.unique(train_data['booth_indices'])
    val_booth_indices = torch.unique(val_data['booth_indices'])
    
    # Initialize CALM model
    print("\n6. Initializing CALM model...")
    n_features = train_data['features'].shape[1]
    n_parties = len(processor.party_names)
    n_booths = len(processor.booth_id_to_idx)
    
    print(f"Model architecture: {n_features} features, {n_parties} parties, {n_booths} booths")
    print(f"Candidate affinity: {use_candidate_affinity}, Mobilization: {use_mobilization}")
    
    model = CandidateAwareVotingModel(
        n_features, n_parties, n_booths, device=device,
        use_candidate_affinity=use_candidate_affinity,
        use_mobilization=use_mobilization
    )
    model.to(device)
    
    # Train model
    print(f"\n7. Training CALM model...")
    train_losses, val_losses, training_info = train_calm_model(
        model, train_data, val_data, train_booth_indices, val_booth_indices,
        processor.booth_metadata, epochs=max_epochs, patience=patience,
        feature_names=train_data['feature_names'], party_names=processor.party_names,
        track_coefficients=True
    )
    
    # Extract coefficients
    print("\n8. Extracting coefficients...")
    coefficients = extract_calm_coefficients(
        model, train_data['feature_names'], processor.party_names
    )
    
    # Save model
    print("\n9. Saving CALM model...")
    model_data = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_features': n_features,
            'n_parties': n_parties,
            'n_booths': n_booths,
            'use_candidate_affinity': use_candidate_affinity,
            'use_mobilization': use_mobilization,
            'rho_age': model.rho_age,
            'candidate_affinity_axes': ['age', 'religion', 'caste'],
            'age_bin_edges': processor.candidate_processor.age_bin_edges
        },
        'feature_names': train_data['feature_names'],
        'party_names': processor.party_names,
        'vectorizer': processor.vectorizer,
        'scaler': processor.scaler,
        'booth_id_to_idx': processor.booth_id_to_idx,
        'candidate_info': candidate_info,
        'training_info': training_info
    }
    
    filename = f'{assembly_name}_CALM_trained_model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"CALM model saved to '{filename}'")
    
    # Save coefficients
    coeff_data = []
    for model_type, coeffs in coefficients.items():
        if isinstance(coeffs, dict):
            for param_name, value in coeffs.items():
                coeff_data.append({
                    'Model_Component': model_type,
                    'Parameter': param_name,
                    'Value': value,
                    'Interpretation': f'{model_type}_{param_name}'
                })
        else:
            coeff_data.append({
                'Model_Component': model_type,
                'Parameter': 'value',
                'Value': coeffs,
                'Interpretation': model_type
            })
    
    coeff_df = pd.DataFrame(coeff_data)
    coeff_filename = f'{assembly_name}_CALM_coefficients.xlsx'
    coeff_df.to_excel(coeff_filename, index=False)
    print(f"CALM coefficients saved to '{coeff_filename}'")
    
    # Save age bin configuration and candidate mapping
    age_bin_data = []
    age_bin_edges = processor.candidate_processor.age_bin_edges
    age_bin_data.append({
        'Configuration': 'Age Bin Edges',
        'Bin_0_Young': f"{age_bin_edges[0]}-{age_bin_edges[1]}",
        'Bin_1_Middle': f"{age_bin_edges[1]}-{age_bin_edges[2]}",
        'Bin_2_Senior': f"{age_bin_edges[2]}-{age_bin_edges[3]}",
        'Raw_Edges': str(age_bin_edges)
    })
    
    # Add candidate age mappings
    if candidate_info:
        for party, info in candidate_info.items():
            if info.get('age'):
                cand_age = info['age']
                cand_bin = processor.candidate_processor._age_to_meta_bin(cand_age)
                bin_names = ['Young (0)', 'Middle-aged (1)', 'Senior (2)']
                age_bin_data.append({
                    'Configuration': f'Candidate_{party}',
                    'Age': cand_age,
                    'Age_Bin': bin_names[cand_bin],
                    'Religion': info.get('religion', 'N/A'),
                    'Caste': info.get('caste', 'N/A')
                })
    
    age_bin_df = pd.DataFrame(age_bin_data)
    age_bin_filename = f'{assembly_name}_CALM_age_bins_config.xlsx'
    age_bin_df.to_excel(age_bin_filename, index=False)
    print(f"Age bin configuration saved to '{age_bin_filename}'")
    
    # Save demographic cells data
    print("\nSaving demographic cells data...")
    cells_export_data = []
    for cell in cells_data:
        cell_row = {
            'booth_id': cell['booth_id'],
            'booth_idx': cell['booth_idx'],
            'cell_weight': cell['cell_weight'],
            'age_category': cell['features']['age_category'],
            'religion_category': cell['features']['religion_category'],
            'caste_category': cell['features']['caste_category'],
            'income_category': cell['features']['income_category'],
            'economic_category': cell['features']['economic_category'],
            'locality': cell['features']['locality'],
            'land_rate_per_sqm': cell['features']['land_rate_per_sqm'],
            'construction_cost_per_sqm': cell['features']['construction_cost_per_sqm'],
            'total_population': cell['features']['total_population'],
            'male_female_ratio': cell['features']['male_female_ratio'],
            'turnout_rate': cell['turnout_rate'],
            'BJP_share': cell['party_shares'][0],
            'Congress_share': cell['party_shares'][1],
            'AAP_share': cell['party_shares'][2],
            'Others_share': cell['party_shares'][3],
            'NOTA_share': cell['party_shares'][4]
        }
        cells_export_data.append(cell_row)
    
    cells_df = pd.DataFrame(cells_export_data)
    cells_filename = f'{assembly_name}_CALM_demographic_cells_data.xlsx'
    cells_df.to_excel(cells_filename, index=False)
    print(f"Demographic cells data saved to '{cells_filename}'")
    
    # Save data corrections log
    if processor.correction_log:
        corrections_df = pd.DataFrame({
            'Correction_Step': range(1, len(processor.correction_log) + 1),
            'Description': processor.correction_log
        })
        corrections_filename = f'{assembly_name}_CALM_data_corrections_log.xlsx'
        corrections_df.to_excel(corrections_filename, index=False)
        print(f"Data corrections log saved to '{corrections_filename}'")
    
    # 10. Validation actuals vs predictions (aggregated)
    print("\n10. Validation actuals vs predictions (aggregated):")
    model.eval()
    with torch.no_grad():
        v_features = val_data['features'].to(device)
        v_booth_idx = val_data['booth_indices'].to(device)
        v_aff = val_data['candidate_affinities'].to(device)
        v_weights = val_data['cell_weights'].to(device)

        v_turnout, v_party_probs, _ = model(v_features, v_booth_idx, v_aff)

        total_pred_turnout = torch.tensor(0.0, device=device)
        total_actual_turnout = torch.tensor(0.0, device=device)
        total_registered = torch.tensor(0.0, device=device)
        pred_party_votes_total = torch.zeros(n_parties, device=device)
        actual_party_votes_total = torch.zeros(n_parties, device=device)

        val_booth_list = val_booth_indices.cpu().numpy().tolist()
        for b in val_booth_list:
            m = (v_booth_idx == b)
            if not m.any():
                continue
            bw = v_weights[m]
            bt = v_turnout[m]
            bp = v_party_probs[m]

            pred_turnout_b = torch.sum(bw * bt)
            pred_party_votes_b = torch.sum(bw.unsqueeze(1) * bt.unsqueeze(1) * bp, dim=0)

            meta = processor.booth_metadata[b]
            N_i = torch.tensor(meta['registered_voters'], dtype=torch.float32, device=device)
            T_i_actual = N_i * torch.tensor(meta['actual_turnout'], dtype=torch.float32, device=device)
            actual_votes_b = torch.tensor([
                meta['actual_votes']['BJP'], meta['actual_votes']['Congress'], meta['actual_votes']['AAP'],
                meta['actual_votes']['Others'], meta['actual_votes']['NOTA']
            ], dtype=torch.float32, device=device)

            total_pred_turnout += pred_turnout_b
            total_actual_turnout += T_i_actual
            total_registered += N_i
            pred_party_votes_total += pred_party_votes_b
            actual_party_votes_total += actual_votes_b

        # Aggregate turnout rates and party shares
        pred_turnout_rate = (total_pred_turnout / total_registered).item() if total_registered > 0 else float('nan')
        actual_turnout_rate = (total_actual_turnout / total_registered).item() if total_registered > 0 else float('nan')

        pred_party_shares = (pred_party_votes_total / (pred_party_votes_total.sum() + 1e-8)).cpu().numpy()
        actual_party_shares = (actual_party_votes_total / (actual_party_votes_total.sum() + 1e-8)).cpu().numpy()

        print(f"  Turnout rate: predicted={pred_turnout_rate:.4f}, actual={actual_turnout_rate:.4f}, diff={(pred_turnout_rate-actual_turnout_rate):+.4f}")
        print("  Party shares (pred vs actual, diff):")
        for idx, party in enumerate(processor.party_names):
            p = float(pred_party_shares[idx]); a = float(actual_party_shares[idx]); d = p - a
            print(f"    {party}: {p:.4f} vs {a:.4f}  (Δ {d:+.4f})")
    
    # Save booth-level predictions for validation set
    print("\nSaving booth-level validation predictions...")
    booth_predictions = []
    
    model.eval()
    with torch.no_grad():
        for booth_idx in val_booth_indices.cpu().numpy().tolist():
            booth_mask = val_data['booth_indices'] == booth_idx
            if not booth_mask.any():
                continue
            
            booth_weights = val_data['cell_weights'][booth_mask]
            booth_features = val_data['features'][booth_mask]
            booth_booth_indices = val_data['booth_indices'][booth_mask]
            booth_affinities = val_data['candidate_affinities'][booth_mask]
            
            turnout_probs, party_probs, _ = model(booth_features, booth_booth_indices, booth_affinities)
            
            # Aggregate predictions
            pred_total_turnout = torch.sum(booth_weights * turnout_probs)
            pred_party_votes = torch.sum(booth_weights.unsqueeze(1) * turnout_probs.unsqueeze(1) * party_probs, dim=0)
            
            # Get actual values
            booth_data = processor.booth_metadata[booth_idx]
            N_i = booth_data['registered_voters']
            actual_turnout_rate = booth_data['actual_turnout']
            actual_turnout_count = N_i * actual_turnout_rate
            
            pred_turnout_rate = (pred_total_turnout / N_i).item()
            pred_party_shares = (pred_party_votes / (pred_party_votes.sum() + 1e-8)).cpu().numpy()
            
            actual_total_votes = sum(booth_data['actual_votes'].values())
            actual_party_shares = {party: votes/max(actual_total_votes, 1) 
                                 for party, votes in booth_data['actual_votes'].items()}
            
            booth_pred = {
                'booth_id': booth_data['booth_id'],
                'booth_idx': booth_idx,
                'registered_voters': N_i,
                'actual_turnout_rate': actual_turnout_rate,
                'predicted_turnout_rate': pred_turnout_rate,
                'turnout_error': pred_turnout_rate - actual_turnout_rate,
                'turnout_error_pct': ((pred_turnout_rate - actual_turnout_rate) / max(actual_turnout_rate, 0.01)) * 100,
                'actual_turnout_count': actual_turnout_count,
                'predicted_turnout_count': pred_total_turnout.item(),
            }
            
            for idx, party in enumerate(processor.party_names):
                booth_pred[f'actual_{party}_share'] = actual_party_shares[party]
                booth_pred[f'predicted_{party}_share'] = float(pred_party_shares[idx])
                booth_pred[f'{party}_error'] = float(pred_party_shares[idx]) - actual_party_shares[party]
                booth_pred[f'actual_{party}_votes'] = booth_data['actual_votes'][party]
                booth_pred[f'predicted_{party}_votes'] = float(pred_party_votes[idx].item())
            
            booth_predictions.append(booth_pred)
    
    booth_pred_df = pd.DataFrame(booth_predictions)
    booth_pred_filename = f'{assembly_name}_CALM_booth_level_predictions.xlsx'
    booth_pred_df.to_excel(booth_pred_filename, index=False)
    print(f"Booth-level predictions saved to '{booth_pred_filename}'")
    
    # Save enhanced all booth predictions (train + val)
    print("\nSaving enhanced all booth predictions...")
    all_booth_predictions = []
    
    model.eval()
    with torch.no_grad():
        # Combine train and val data
        all_features = torch.cat([train_data['features'], val_data['features']], dim=0).to(device)
        all_booth_indices = torch.cat([train_data['booth_indices'], val_data['booth_indices']], dim=0).to(device)
        all_weights = torch.cat([train_data['cell_weights'], val_data['cell_weights']], dim=0).to(device)
        all_affinities = torch.cat([train_data['candidate_affinities'], val_data['candidate_affinities']], dim=0).to(device)
        
        unique_booths = torch.unique(all_booth_indices).cpu().numpy().tolist()
        
        for booth_idx in unique_booths:
            booth_mask = all_booth_indices == booth_idx
            if not booth_mask.any():
                continue
            
            booth_weights = all_weights[booth_mask]
            booth_features = all_features[booth_mask]
            booth_booth_indices = all_booth_indices[booth_mask]
            booth_affinities = all_affinities[booth_mask]
            
            turnout_probs, party_probs, _ = model(booth_features, booth_booth_indices, booth_affinities)
            
            pred_total_turnout = torch.sum(booth_weights * turnout_probs)
            pred_party_votes = torch.sum(booth_weights.unsqueeze(1) * turnout_probs.unsqueeze(1) * party_probs, dim=0)
            
            booth_data = processor.booth_metadata[booth_idx]
            N_i = booth_data['registered_voters']
            actual_turnout_rate = booth_data['actual_turnout']
            
            pred_turnout_rate = (pred_total_turnout / N_i).item()
            pred_party_shares = (pred_party_votes / (pred_party_votes.sum() + 1e-8)).cpu().numpy()
            
            actual_total_votes = sum(booth_data['actual_votes'].values())
            actual_party_shares = {party: votes/max(actual_total_votes, 1) 
                                 for party, votes in booth_data['actual_votes'].items()}
            
            is_train = booth_idx in train_booth_indices.cpu().numpy().tolist()
            
            booth_pred = {
                'booth_id': booth_data['booth_id'],
                'booth_idx': booth_idx,
                'dataset': 'Train' if is_train else 'Validation',
                'registered_voters': N_i,
                'actual_turnout_rate': actual_turnout_rate,
                'predicted_turnout_rate': pred_turnout_rate,
                'turnout_error': pred_turnout_rate - actual_turnout_rate,
                'turnout_abs_error': abs(pred_turnout_rate - actual_turnout_rate),
                'turnout_error_pct': ((pred_turnout_rate - actual_turnout_rate) / max(actual_turnout_rate, 0.01)) * 100,
            }
            
            for idx, party in enumerate(processor.party_names):
                booth_pred[f'actual_{party}_share'] = actual_party_shares[party]
                booth_pred[f'predicted_{party}_share'] = float(pred_party_shares[idx])
                booth_pred[f'{party}_error'] = float(pred_party_shares[idx]) - actual_party_shares[party]
                booth_pred[f'{party}_abs_error'] = abs(float(pred_party_shares[idx]) - actual_party_shares[party])
            
            all_booth_predictions.append(booth_pred)
    
    all_booth_df = pd.DataFrame(all_booth_predictions)
    all_booth_filename = f'{assembly_name}_CALM_enhanced_all_booth_predictions.xlsx'
    all_booth_df.to_excel(all_booth_filename, index=False)
    print(f"Enhanced all booth predictions saved to '{all_booth_filename}'")
    
    # Save detailed coefficient evolution tracking (every 10 epochs)
    if training_info.get('coefficient_history'):
        print("\nSaving detailed coefficient evolution (every 10 epochs)...")
        coeff_evolution = []
        
        for epoch_data in training_info['coefficient_history']:
            epoch = epoch_data['epoch']
            train_loss = epoch_data['train_loss']
            val_loss = epoch_data['val_loss']
            lr = epoch_data['learning_rate']
            
            # Extract turnout coefficients
            for param_name, value in epoch_data['turnout'].items():
                coeff_evolution.append({
                    'Epoch': epoch,
                    'Model_Component': 'Turnout',
                    'Parameter': param_name,
                    'Value': value,
                    'Train_Loss': train_loss,
                    'Val_Loss': val_loss,
                    'Learning_Rate': lr
                })
            
            # Extract party coefficients
            for party in processor.party_names:
                party_key = f'party_{party}'
                if party_key in epoch_data:
                    for param_name, value in epoch_data[party_key].items():
                        coeff_evolution.append({
                            'Epoch': epoch,
                            'Model_Component': party,
                            'Parameter': param_name,
                            'Value': value,
                            'Train_Loss': train_loss,
                            'Val_Loss': val_loss,
                            'Learning_Rate': lr
                        })
            
            # Extract candidate affinity coefficients
            if 'candidate_affinity' in epoch_data:
                for param_name, value in epoch_data['candidate_affinity'].items():
                    coeff_evolution.append({
                        'Epoch': epoch,
                        'Model_Component': 'Candidate_Affinity',
                        'Parameter': param_name,
                        'Value': value,
                        'Train_Loss': train_loss,
                        'Val_Loss': val_loss,
                        'Learning_Rate': lr
                    })
            
            # Extract mobilization coefficients
            if 'mobilization' in epoch_data:
                for param_name, value in epoch_data['mobilization'].items():
                    coeff_evolution.append({
                        'Epoch': epoch,
                        'Model_Component': 'Mobilization',
                        'Parameter': param_name,
                        'Value': value,
                        'Train_Loss': train_loss,
                        'Val_Loss': val_loss,
                        'Learning_Rate': lr
                    })
        
        coeff_evolution_df = pd.DataFrame(coeff_evolution)
        coeff_evolution_filename = f'{assembly_name}_CALM_coefficients_evolution_per_10_epochs.xlsx'
        coeff_evolution_df.to_excel(coeff_evolution_filename, index=False)
        print(f"Detailed coefficient evolution saved to '{coeff_evolution_filename}'")
        print(f"  Total snapshots: {len(training_info['coefficient_history'])} (every 10 epochs)")
        print(f"  Total coefficient records: {len(coeff_evolution_df)}")

    print("\n=== CALM MODEL TRAINING COMPLETE ===")
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"Best Validation Loss: {training_info['best_val_loss']:.4f} (epoch {training_info['best_epoch']})")
    print(f"Total Cells Created: {len(cells_data)}")
    print(f"Features: {len(train_data['feature_names'])}")
    print(f"Train Booths: {len(train_booth_indices)}, Val Booths: {len(val_booth_indices)}")
    
    if use_candidate_affinity:
        print("\nCandidate Affinity Effects (3D: age, religion, caste):")
        for effect, value in coefficients['candidate_affinity'].items():
            print(f"  {effect}: {value:.4f}")
    
    print("\n📁 FILES CREATED:")
    print(f"  1. {assembly_name}_CALM_trained_model.pkl - Complete trained model")
    print(f"  2. {assembly_name}_CALM_coefficients.xlsx - Model coefficients and effects")
    print(f"  3. {assembly_name}_CALM_age_bins_config.xlsx - Age bin configuration and candidate mapping")
    print(f"  4. {assembly_name}_CALM_demographic_cells_data.xlsx - Complete cells analysis")
    print(f"  5. {assembly_name}_CALM_data_corrections_log.xlsx - Data processing corrections")
    print(f"  6. {assembly_name}_CALM_booth_level_predictions.xlsx - Validation predictions")
    print(f"  7. {assembly_name}_CALM_enhanced_all_booth_predictions.xlsx - All booth predictions with errors")
    print(f"  8. {assembly_name}_CALM_coefficients_evolution_per_10_epochs.xlsx - Detailed coefficient tracking every 10 epochs")
    
    return model, coefficients, training_info

# Example usage with candidate information
if __name__ == "__main__":
    # Example candidate information for Madipur constituency
    candidate_info = {
        'BJP': {'age': 54, 'religion': 'Hindu', 'caste': 'OBC'},
        'Congress': {'age': 74, 'religion': 'Hindu', 'caste': 'SC'},
        'AAP': {'age': 34, 'religion': 'Hindu', 'caste': 'Vaishya'},
        # Note: Others and NOTA don't have specific candidates, affinity will be zero
    }
    
    # Train CALM model
    model, coefficients, training_info = main_calm(
        file_2020=None,
        file_2025='Assembly_Madipur_2025_Election.xlsx',
        use_candidate_affinity=True,
        use_mobilization=False,  # Start with just affinity, can enable later
        candidate_info=candidate_info,
        max_epochs=2000,
        patience=100
    )
    
    print("\n📁 FILES CREATED: See saved filenames printed above (trained model and coefficients).")

# SUMMARY OF CALM IMPLEMENTATION:
"""
✅ IMPLEMENTED: Candidate-Aware Aggregated Likelihood Model (CALM)

🔸 Core Features:
    - CandidateAwareVotingModel: Extends base model with candidate affinity terms
    - CandidateProcessor: Handles candidate encoding and affinity computation
    - Candidate affinity is 3D: [age, religion, caste]. Income only appears on voter side.
    - Quantile (tertile) candidate age bins with a 3x3 proximity kernel (same=1, adjacent=ρ_age≈0.8, non-adj=0)
    - Hindu-only caste logic (no caste matching for non-Hindu cells/candidates)
    - Raw (non-standardized) affinity values for interpretability

🔸 Mathematical Implementation:
    - Party utilities: ν_ic,k = γ_k + b_i^(P) + x_ic^T β_P + η^T g_ic,k
    - Optional mobilization: π_ic = σ(α_0 + b_i^(T) + x_ic^T β_T + ξ * m_ic)
    - Candidate affinity vector: g_ic,k = [g_age, g_rel, g_caste] (3D)
    - Proper regularization with λ_eta penalty on candidate parameters

🔸 Key Innovations:
    - Quantile-based candidate age meta-bins (youth-ish / mid / older) learned from provided candidates
    - Voter cells keep Age × Religion × Caste × Income; only the age axis is collapsed to 3 meta-bins for affinity comparison
    - Configurable age proximity (ρ_age default 0.8)
    - Early stopping with booth-aggregated loss

🔸 Usage Example:
    candidate_info = {
            'BJP': {'age': 54, 'religion': 'Hindu', 'caste': 'OBC', 'income_class': 'Middle'},
            'Congress': {'age': 45, 'religion': 'Hindu', 'caste': 'SC', 'income_class': 'Middle'},
            # ... etc
    }
  
    model, coefficients, training_info = main_calm(
            file_2025='Assembly_Madipur_2025_Election.xlsx',
            candidate_info=candidate_info,
            use_candidate_affinity=True
    )

🔸 Interpretability:
    - η_age > 0: Age proximity (by meta-bin kernel) helps party performance
    - η_religion > 0: Same religion gives boost
    - η_caste > 0: Same caste helps (Hindu voters only)

The implementation provides a clean, mathematically grounded way to incorporate
candidate characteristics while preserving the aggregation framework and
interpretability of the original model.
"""