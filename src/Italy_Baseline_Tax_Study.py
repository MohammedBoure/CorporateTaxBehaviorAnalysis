import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os
import warnings
import logging

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
INPUT_FILE = 'EUTO_Public_CbCR_Database_2021.xlsx'  # Verify your filename
OUTPUT_FILE = 'Italy_Utilities_Analysis.xlsx'       # Output filename
SHEET_NAME = 'Public_CbCRs'                         # Verify sheet name

# Filter Requirements
TARGET_UPE = 'Italy'         # UPE Name
TARGET_JUR = 'All'           # Jur Name / Partner Jurisdiction (Look for "All")
TARGET_SECTOR = 'Utilities'  # Sector

# ETR Thresholds
ETR_MIN = 0.0
ETR_MAX = 1.0  # Slightly increased range to capture more data points

# ==============================================================================
# 1. DATA LOADER & COLUMN MAPPING
# ==============================================================================
def load_and_standardize_data(filepath):
    logger.info("="*60)
    logger.info(f"STEP 1: LOADING DATA FROM [{filepath}]")
    logger.info("="*60)

    if not os.path.exists(filepath):
        # Try looking in parent directory if file not found
        filepath = os.path.join('..', filepath)
        if not os.path.exists(filepath):
            logger.error(f"File '{filepath}' not found.")
            raise FileNotFoundError(f"File '{filepath}' not found.")
            
    df = pd.read_excel(filepath, sheet_name=SHEET_NAME, engine='openpyxl')
    logger.info(f"-> Raw Data Loaded. Total Rows: {len(df)}")
    
    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()
    
    # === COLUMN MAPPING ===
    # Keys (left) must match the Excel Headers exactly.
    # Values (right) are the internal names used in the script.
    column_map = {
        # Financial Data
        'Profit (Loss) before Income Tax': 'profit_before_tax',
        'Profit Before Tax': 'profit_before_tax',
        'Income Tax Accrued': 'tax_accrued',
        'Income Tax Paid': 'tax_paid',
        'Number of Employees': 'employees',
        'Tangible Assets other than Cash and Cash Equivalents': 'tangible_assets',
        'Tangible Assets': 'tangible_assets',
        'Related Party Revenues': 'related_revenues',
        'Total Revenues': 'total_revenues',
        
        # Classification Data
        'Ultimate Parent Entity': 'upe_name',
        'UPE Name': 'upe_name',
        'Jur Name': 'jurisdiction',          # Column for Jurisdiction
        'Partner Jurisdiction': 'jurisdiction', 
        'Sector': 'sector',                  # Column for Sector
        'Main Business Activity': 'sector'
    }
    
    # Rename columns
    df = df.rename(columns=column_map)
    
    # Check for missing critical columns for filtering
    required_cols = ['upe_name', 'jurisdiction', 'sector']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning(f"Warning: The following filter columns are missing: {missing}")
        logger.warning("Please check the 'column_map' dictionary in the code.")
    
    return df

# ==============================================================================
# 2. GLOBAL CLEANING LOGIC
# ==============================================================================
def create_global_clean_bases(df, tax_col):
    logger.info(f"\n--- PREPARING DATA FOR TAX TYPE: {tax_col} ---")
    
    # Convert columns to numeric
    cols_to_num = ['profit_before_tax', tax_col]
    for c in cols_to_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Clean Control Variables
    controls = ['employees', 'tangible_assets', 'related_revenues']
    valid_controls = [c for c in controls if c in df.columns]
    for c in valid_controls:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # -----------------------
    # Base 1: (Profit & Tax Only)
    # -----------------------
    gb1 = df.dropna(subset=['profit_before_tax', tax_col]).copy()
    gb1 = gb1[gb1['profit_before_tax'] > 0] # Positive profits only
    
    # Calculate ETR
    gb1['ETR'] = gb1[tax_col] / gb1['profit_before_tax']
    
    # Apply ETR Limits
    gb1 = gb1[(gb1['ETR'] >= ETR_MIN) & (gb1['ETR'] < ETR_MAX)]
    
    # Create Regression Variables
    gb1['ln_profits'] = np.log(gb1['profit_before_tax'])
    gb1['ETR_sq'] = gb1['ETR'] ** 2
    
    # -----------------------
    # Base 2: (With Controls)
    # -----------------------
    gb2 = df.dropna(subset=['profit_before_tax', tax_col] + valid_controls).copy()
    gb2 = gb2[gb2['profit_before_tax'] > 0]
    gb2['ETR'] = gb2[tax_col] / gb2['profit_before_tax']
    gb2 = gb2[(gb2['ETR'] >= ETR_MIN) & (gb2['ETR'] < ETR_MAX)]
    
    gb2['ln_profits'] = np.log(gb2['profit_before_tax'])
    gb2['ETR_sq'] = gb2['ETR'] ** 2
    
    logged_controls = []
    for c in valid_controls:
        # Ensure positive values before Log
        gb2 = gb2[gb2[c] > 0] 
        gb2[f'ln_{c}'] = np.log(gb2[c])
        logged_controls.append(f'ln_{c}')
        
    return gb1, gb2, logged_controls

# ==============================================================================
# 3. REGRESSION FUNCTION
# ==============================================================================
def run_regression(df, formula, title):
    logger.info(f"Running Regression: {title} (N={len(df)})")
    out = f"\n\n{'-'*20}\n{title}\n{'-'*20}\n"
    if len(df) < 5:
        return out + "\nNot enough data points (<5) to run regression."
        
    try:
        model = smf.ols(formula, data=df).fit()
        out += model.summary().as_text()
        
        # U-Test Check (Quadratic Relationship)
        if 'ETR_sq' in model.params:
            b1 = model.params.get('ETR', 0)
            b2 = model.params.get('ETR_sq', 0)
            
            # Turning Point
            tp = -b1 / (2*b2) if b2 != 0 else 0
            min_etr, max_etr = df['ETR'].min(), df['ETR'].max()
            is_valid = min_etr <= tp <= max_etr
            
            out += f"\n\n>> U-TEST RESULTS:"
            out += f"\n   Coeff ETR (Linear): {b1:.4f}"
            out += f"\n   Coeff ETR^2 (Quad): {b2:.4f}"
            out += f"\n   Turning Point: {tp:.4%}"
            out += f"\n   In Range [{min_etr:.2%}, {max_etr:.2%}]? {'YES' if is_valid else 'NO'}"
            
    except Exception as e:
        logger.error(f"Regression Failed: {e}")
        out += f"\nRegression Failed: {str(e)}"
    return out

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
def main():
    try:
        df_raw = load_and_standardize_data(INPUT_FILE)
    except Exception as e:
        logger.critical(f"Execution stopped: {e}")
        return

    # ==========================================================================
    # FILTERING LOGIC
    # UPE Name = Italy | Jur Name = All | Sector = Utilities
    # ==========================================================================
    logger.info("Applying Filters...")
    
    # 1. Filter UPE Name
    if 'upe_name' in df_raw.columns:
        # Convert to string and lowercase for comparison to handle inconsistencies
        mask_upe = df_raw['upe_name'].astype(str).str.lower() == TARGET_UPE.lower()
        df_filtered = df_raw[mask_upe]
        logger.info(f"-> After UPE Filter ({TARGET_UPE}): {len(df_filtered)} rows")
    else:
        logger.error("Column 'upe_name' not found!")
        return

    # 2. Filter Jurisdiction
    if 'jurisdiction' in df_raw.columns:
        # Check if the cell contains "all" (e.g., "All Jurisdictions")
        mask_jur = df_filtered['jurisdiction'].astype(str).str.lower().str.contains("all")
        df_filtered = df_filtered[mask_jur]
        logger.info(f"-> After Jurisdiction Filter ({TARGET_JUR}): {len(df_filtered)} rows")
    else:
        logger.warning("Column 'jurisdiction' not found. Skipping Jur filter.")

    # 3. Filter Sector
    if 'sector' in df_raw.columns:
        mask_sec = df_filtered['sector'].astype(str).str.lower() == TARGET_SECTOR.lower()
        df_filtered = df_filtered[mask_sec]
        logger.info(f"-> After Sector Filter ({TARGET_SECTOR}): {len(df_filtered)} rows")
    else:
        logger.warning("Column 'sector' not found. Skipping Sector filter.")

    if len(df_filtered) == 0:
        logger.critical("No data left after filtering! Check your filter values in the configuration section.")
        return

    # ==========================================================================
    # PROCESSING & ANALYSIS
    # ==========================================================================
    all_sheets = {}
    report_lines = []
    
    report_lines.append(f"ANALYSIS PARAMETERS:")
    report_lines.append(f"Target UPE: {TARGET_UPE}")
    report_lines.append(f"Target Jurisdiction: {TARGET_JUR}")
    report_lines.append(f"Target Sector: {TARGET_SECTOR}")
    report_lines.append("-" * 30)

    # --- PROCESS ACCRUED TAX ---
    b1_acc, b2_acc, acc_ctrls = create_global_clean_bases(df_filtered, 'tax_accrued')
    
    all_sheets['Accrued_Base1'] = b1_acc
    all_sheets['Accrued_Base2'] = b2_acc
    
    report_lines.append("\n*** RESULTS: ACCRUED TAX ***")
    # Linear
    report_lines.append(run_regression(b1_acc, "ln_profits ~ ETR", "Accrued Base 1 (Linear)"))
    # Non-Linear (Quadratic)
    report_lines.append(run_regression(b1_acc, "ln_profits ~ ETR + ETR_sq", "Accrued Base 1 (Non-Linear)"))
    
    # With Controls
    if len(b2_acc) > 0:
        ctrls_str = " + ".join(acc_ctrls)
        report_lines.append(run_regression(b2_acc, f"ln_profits ~ ETR + ETR_sq + {ctrls_str}", "Accrued Base 2 (Controls)"))

    # --- PROCESS PAID TAX ---
    b1_paid, b2_paid, paid_ctrls = create_global_clean_bases(df_filtered, 'tax_paid')
    
    all_sheets['Paid_Base1'] = b1_paid
    all_sheets['Paid_Base2'] = b2_paid
    
    report_lines.append("\n*** RESULTS: PAID TAX ***")
    report_lines.append(run_regression(b1_paid, "ln_profits ~ ETR", "Paid Base 1 (Linear)"))
    report_lines.append(run_regression(b1_paid, "ln_profits ~ ETR + ETR_sq", "Paid Base 1 (Non-Linear)"))
    
    if len(b2_paid) > 0:
        ctrls_str = " + ".join(paid_ctrls)
        report_lines.append(run_regression(b2_paid, f"ln_profits ~ ETR + ETR_sq + {ctrls_str}", "Paid Base 2 (Controls)"))

    # SAVE TO EXCEL
    logger.info(f"Saving to {OUTPUT_FILE}...")
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        for name, data in all_sheets.items():
            if not data.empty:
                data.to_excel(writer, sheet_name=name, index=False)
        
        # Save Text Report
        pd.DataFrame("\n".join(report_lines).split('\n')).to_excel(writer, sheet_name='Regression_Report', index=False, header=False)
        
    logger.info("DONE.")

if __name__ == "__main__":
    main()