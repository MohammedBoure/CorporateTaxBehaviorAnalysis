import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os
import warnings
import logging
# Import Imputation Library
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# ==============================================================================
# 0. LOGGING SETUP
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_FILE = 'EUTO_Public_CbCR_Database_2021.xlsx'
OUTPUT_FILE = 'Italy_Imputation_Results_Final.xlsx'
SHEET_NAME = 'Public_CbCRs'
TARGET_COUNTRY = 'Italy' # Targeted Country

# Study Conditions
ETR_MIN = 0.0
ETR_MAX = 0.5

# ==============================================================================
# 1. DATA LOADER & COLUMN MAPPING
# ==============================================================================
def load_and_standardize_data(filepath):
    logger.info("="*60)
    logger.info(f"STEP 1: LOADING DATA FROM [{filepath}]")
    if not os.path.exists(filepath):
        filepath = os.path.join('..', filepath)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File '{filepath}' not found.")
            
    df = pd.read_excel(filepath, sheet_name=SHEET_NAME, engine='openpyxl')
    df.columns = df.columns.str.strip()
    
    column_map = {
        'Profit (Loss) before Income Tax': 'profit_before_tax',
        'Profit Before Tax': 'profit_before_tax',
        'Income Tax Accrued': 'tax_accrued',
        'Income Tax Paid': 'tax_paid',
        'Number of Employees': 'employees',
        'Tangible Assets other than Cash and Cash Equivalents': 'tangible_assets',
        'Tangible Assets': 'tangible_assets',
        'Related Party Revenues': 'related_revenues',
        'Total Revenues': 'total_revenues',
        'Ultimate Parent Entity': 'upe_name',
        'UPE Name': 'upe_name'
    }
    df = df.rename(columns=column_map)
    
    # Convert columns to numeric
    numeric_cols = ['profit_before_tax', 'tax_accrued', 'tax_paid', 
                    'employees', 'tangible_assets', 'related_revenues']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
    return df

# ==============================================================================
# 2. ANALYSIS LOGIC
# ==============================================================================
def run_regression(df, formula, title):
    logger.info(f"Running Regression: {title}")
    out = f"\n\n{'-'*20}\n{title}\n{'-'*20}\n"
    try:
        # Safety check for Infinite values
        if np.isinf(df.select_dtypes(include=np.number)).values.any():
            logger.warning("   -> Found Infinite values! Dropping them before regression...")
            df = df.replace([np.inf, -np.inf], np.nan).dropna()

        model = smf.ols(formula, data=df).fit()
        out += model.summary().as_text()
        
        # U-Test Logic
        if 'ETR_sq' in model.params:
            b1, b2 = model.params['ETR'], model.params['ETR_sq']
            tp = -b1 / (2*b2) if b2 != 0 else 0
            min_etr, max_etr = df['ETR'].min(), df['ETR'].max()
            is_valid = min_etr <= tp <= max_etr
            out += f"\n\n>> U-TEST: Coeff ETR^2={b2:.4f}, TP={tp:.4%}, In Range? {'YES' if is_valid else 'NO'}"
            
        logger.info(f"   -> Success. R-squared: {model.rsquared:.4f}, N: {int(model.nobs)}")
        
    except Exception as e:
        logger.error(f"Regression Failed: {e}")
        out += f"\nRegression Failed: {str(e)}"
    return out

def prepare_data_and_analyze(df, tax_col, method='CC'):
    controls = ['employees', 'tangible_assets', 'related_revenues']
    cols = ['profit_before_tax', tax_col] + controls
    
    # 1. Filter Country
    data = df[df['upe_name'] == TARGET_COUNTRY].copy()
    data = data[cols] 
    
    # *** CRITICAL FIX: Treat 0 as NaN for Imputation/Log purposes ***
    # This ensures we don't get log(0) = -inf later.
    data[controls] = data[controls].replace(0, np.nan)

    # --- METHOD HANDLING ---
    if method == 'CC': # Complete Case (Listwise Deletion)
        data = data.dropna()
        logger.info(f"[{method}] After Listwise Deletion: {len(data)} observations")
        
    elif method == 'Imputation': # MICE
        logger.info(f"[{method}] Before Imputation: {len(data)} observations")
        
        # Using MICE to fill missing values (and zeros converted to NaNs)
        imputer = IterativeImputer(min_value=0.1, max_iter=20, random_state=42)
        
        if len(data) > 5:
            data_imputed = imputer.fit_transform(data)
            data = pd.DataFrame(data_imputed, columns=cols, index=data.index)
            logger.info(f"[{method}] After Imputation: {len(data)} observations")
        else:
            return None, "Insufficient data"

    # 2. Standard Filters (Profit & ETR)
    #
    data = data[data['profit_before_tax'] > 0]
    data['ETR'] = data[tax_col] / data['profit_before_tax']
    data = data[(data['ETR'] >= ETR_MIN) & (data['ETR'] < ETR_MAX)]
    
    # 3. Log Transformations (Safe Mode)
    # Ensure strictly positive before log
    for c in ['profit_before_tax'] + controls:
        data = data[data[c] > 0]
        
    data['ln_profits'] = np.log(data['profit_before_tax'])
    for c in controls:
        data[f'ln_{c}'] = np.log(data[c])
        
    data['ETR_sq'] = data['ETR'] ** 2
    
    logger.info(f"[{method}] Final Sample Size for Regression: {len(data)}")
    
    return data, ""

# ==============================================================================
# 3. MAIN
# ==============================================================================
def main():
    try:
        df_raw = load_and_standardize_data(INPUT_FILE)
    except Exception as e:
        logger.critical(f"Execution stopped: {e}")
        return
    
    all_sheets = {}
    report_lines = []
    
    # Run for both Accrued and Paid
    for tax_type in ['tax_accrued', 'tax_paid']:
        type_label = "Accrued" if tax_type == 'tax_accrued' else "Paid"
        
        # Compare CC (Old Method) vs Imputation (New Method)
        for method in ['CC', 'Imputation']:
            label = f"{type_label}_{method}"
            logger.info(f"\n--- PROCESSING: {label} ---")
            
            df_res, msg = prepare_data_and_analyze(df_raw, tax_type, method)
            
            if df_res is not None and not df_res.empty:
                all_sheets[label] = df_res
                
                controls = ['ln_employees', 'ln_tangible_assets', 'ln_related_revenues']
                formula_lin = f"ln_profits ~ ETR + {' + '.join(controls)}"
                formula_non = f"ln_profits ~ ETR + ETR_sq + {' + '.join(controls)}"
                
                report_lines.append(f"*** {TARGET_COUNTRY} {label} ***")
                report_lines.append(f"N = {len(df_res)}")
                
                if len(df_res) > 10:
                    report_lines.append(run_regression(df_res, formula_lin, f"{label} Linear"))
                    report_lines.append(run_regression(df_res, formula_non, f"{label} Non-Linear"))
                else:
                    report_lines.append("Insufficient data for regression.")
            else:
                report_lines.append(f"*** {TARGET_COUNTRY} {label} ***")
                report_lines.append(f"Failed/Empty. {msg}")

    # Save
    logger.info("="*60)
    logger.info(f"STEP 3: SAVING RESULTS TO [{OUTPUT_FILE}]")
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        for name, data in all_sheets.items():
            data.to_excel(writer, sheet_name=name, index=False)
        pd.DataFrame("\n".join(report_lines).split('\n')).to_excel(writer, sheet_name='Regression_Output', index=False, header=False)
        
    logger.info(f"Done. Execution completed successfully.")

if __name__ == "__main__":
    main()