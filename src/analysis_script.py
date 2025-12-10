import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import xml.etree.ElementTree as ET
import os
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_FILE = 'EUTO_Public_CbCR_Database_2021 2.xls'
OUTPUT_FILE = 'Final_Germany_Strict_Clean.xlsx'
TARGET_COUNTRY = 'Germany'

# Columns allowed in the final clean file
METADATA_COLUMNS = [
    'mnc', 'year', 'sector', 'upe_code', 'upe_name', 'jur_code', 'jur_name',
    'total_revenues', 'profit_before_tax', 'tax_paid', 'employees',
    'tangible_assets', 'tax_accrued', 'unrelated_revenues', 'related_revenues',
    'stated_capital', 'accumulated_earnings', 'currency'
]

# ==============================================================================
# CLASS 1: DATA LOADER (Extract)
# ==============================================================================
class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_raw_xml(self):
        print(f"[DataLoader] Reading file: {self.file_path}...")
        # Check if file exists
        if not os.path.exists(self.file_path):
             # Try adding .csv or .xlsx extension if not found
            if os.path.exists(self.file_path + ".csv"):
                self.file_path += ".csv"
            elif os.path.exists(self.file_path.replace(".xls", ".xlsx")):
                self.file_path = self.file_path.replace(".xls", ".xlsx")
            else:
                raise FileNotFoundError(f"Input file {self.file_path} not found.")
        
        # Load based on extension
        if self.file_path.endswith('.csv'):
            return pd.read_csv(self.file_path)
        else:
            return pd.read_excel(self.file_path)

# ==============================================================================
# CLASS 2: AGGRESSIVE DATA CLEANER (Transform)
# ==============================================================================
class StrictDataCleaner:
    def __init__(self, raw_df, country):
        self.raw_df = raw_df
        self.country = country

    def create_master_clean_file(self):
        print("[DataCleaner] Starting Deep Cleaning Pipeline...")
        df = self.raw_df.copy()

        # 1. Clean Column Headers
        df.columns = df.columns.str.strip()

        # 2. Filter Country
        if 'upe_name' not in df.columns:
            raise KeyError("'upe_name' column missing.")
        df = df[df['upe_name'] == self.country].copy()

        # 3. AGGRESSIVE NULL REPLACEMENT (The Fix)
        # Convert empty strings, spaces, and 'None' strings to real NaN
        df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        df.replace('None', np.nan, inplace=True)
        
        # 4. Strict Schema Filtering
        # Keep ONLY columns defined in Metadata
        valid_cols = [c for c in METADATA_COLUMNS if c in df.columns]
        df = df[valid_cols]

        # 5. Type Enforcement
        numeric_cols = [
            'total_revenues', 'profit_before_tax', 'tax_paid', 'employees',
            'tangible_assets', 'tax_accrued', 'unrelated_revenues', 
            'related_revenues', 'stated_capital', 'accumulated_earnings'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 6. Drop Completely Empty Rows (in critical columns)
        critical = ['profit_before_tax', 'tax_accrued']
        available_crit = [c for c in critical if c in df.columns]
        df = df.dropna(subset=available_crit, how='any') # Strict drop if Profit/Tax missing

        # 7. Drop Completely Empty Columns
        # This will now work because we converted "" to NaN in step 3
        df = df.dropna(axis=1, how='all')

        print(f"[DataCleaner] Cleaned Data Shape: {df.shape}")
        return df

# ==============================================================================
# CLASS 3: ANALYZER (Load & Analyze)
# ==============================================================================
class Analyzer:
    def __init__(self, clean_df):
        self.df = clean_df

    def prepare_datasets(self, tax_col):
        if tax_col not in self.df.columns:
            print(f"[Analyzer] Warning: {tax_col} missing.")
            return None, None, None

        # BASE 1
        base1 = self.df.dropna(subset=['profit_before_tax', tax_col]).copy()
        base1['ETR'] = base1[tax_col] / base1['profit_before_tax']
        
        base1 = base1[
            (base1['profit_before_tax'] > 0) & 
            (base1['ETR'] >= 0) & 
            (base1['ETR'] < 0.5)
        ]
        base1['ln_profits'] = np.log(base1['profit_before_tax'])
        base1['ETR_sq'] = base1['ETR'] ** 2

        # BASE 2
        base2 = base1.copy()
        possible_controls = ['employees', 'tangible_assets', 'related_revenues']
        valid_controls = []
        
        for ctrl in possible_controls:
            if ctrl in base2.columns:
                if (base2[ctrl] > 0).sum() > 5: 
                    valid_controls.append(ctrl)
        
        base2 = base2.dropna(subset=valid_controls)
        for ctrl in valid_controls:
            base2 = base2[base2[ctrl] > 0]
            base2[f'ln_{ctrl}'] = np.log(base2[ctrl])

        return base1, base2, valid_controls

    def run_regression(self, df, formula, title):
        output = f"\n{'='*30}\n{title}\n{'='*30}\n"
        if len(df) < 5:
            output += "Insufficient Data."
            return output
        try:
            model = smf.ols(formula, data=df).fit()
            output += model.summary().as_text()
            if 'ETR_sq' in model.params:
                b1 = model.params['ETR']
                b2 = model.params['ETR_sq']
                tp = -b1 / (2*b2) if b2 != 0 else 0
                shape = "Inverted U" if b2 < 0 else "U-Shape"
                output += f"\n\n[U-Test] Shape: {shape}, Turning Point: {tp:.4f}"
        except Exception as e:
            output += f"Error: {e}"
        return output

# ==============================================================================
# CLASS 4: PIPELINE
# ==============================================================================
class Pipeline:
    def __init__(self):
        self.sheets = {}
        self.final_report = []

    def run(self):
        # 1. Extract
        loader = DataLoader(INPUT_FILE)
        raw_df = loader.load_raw_xml()
        if raw_df is None: return

        # 2. Transform
        cleaner = StrictDataCleaner(raw_df, TARGET_COUNTRY)
        master_clean_df = cleaner.create_master_clean_file()
        if master_clean_df is None: return

        # STORE MASTER CLEAN FIRST
        self.sheets['MASTER_CLEANED_DATA'] = master_clean_df

        # 3. Analyze
        analyzer = Analyzer(master_clean_df)
        self._run_cycle(analyzer, 'tax_accrued', 'Accrued')
        self._run_cycle(analyzer, 'tax_paid', 'Paid')

        # 4. Export
        self._save()

    def _run_cycle(self, analyzer, tax_col, label):
        b1, b2, ctrls = analyzer.prepare_datasets(tax_col)
        if b1 is None: return
        self.sheets[f'{label}_BASE1'] = b1
        self.sheets[f'{label}_BASE2'] = b2
        
        self.final_report.append(f"*** {label} Analysis ***")
        self.final_report.append(analyzer.run_regression(b1, "ln_profits ~ ETR", f"{label} M1"))
        self.final_report.append(analyzer.run_regression(b1, "ln_profits ~ ETR + ETR_sq", f"{label} M2"))
        
        if ctrls:
            f_ctrl = " + ".join([f"ln_{c}" for c in ctrls])
            self.final_report.append(analyzer.run_regression(b2, f"ln_profits ~ ETR + {f_ctrl}", f"{label} M3"))
            self.final_report.append(analyzer.run_regression(b2, f"ln_profits ~ ETR + ETR_sq + {f_ctrl}", f"{label} M4"))

    def _save(self):
        print(f"[Pipeline] Saving to {OUTPUT_FILE}...")
        with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
            for name, df in self.sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)
            
            rep_df = pd.DataFrame([x.split('\n') for x in "\n".join(self.final_report).split('\n')])
            rep_df.to_excel(writer, sheet_name='Regression_Results', index=False, header=False)
        print("Done.")

if __name__ == "__main__":
    Pipeline().run()