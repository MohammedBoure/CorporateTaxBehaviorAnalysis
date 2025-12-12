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
INPUT_FILE = 'EUTO_Public_CbCR_Database_2021.xlsx' 
OUTPUT_FILE = 'Germany_Baseline_Results_Full.xlsx' # تم تعديل الاسم
SHEET_NAME = 'Public_CbCRs' 
TARGET_COUNTRY = 'Germany' 

# الشروط
ETR_MIN = 0.0
ETR_MAX = 0.5

# ==============================================================================
# 1. DATA LOADER & COLUMN MAPPING
# ==============================================================================
def load_and_standardize_data(filepath):
    logger.info("="*60)
    logger.info(f"STEP 1: LOADING DATA FROM [{filepath}]")
    logger.info("="*60)

    if not os.path.exists(filepath):
        filepath = os.path.join('..', filepath)
        if not os.path.exists(filepath):
            logger.error(f"File '{filepath}' not found.")
            raise FileNotFoundError(f"File '{filepath}' not found.")
            
    df = pd.read_excel(filepath, sheet_name=SHEET_NAME, engine='openpyxl')
    logger.info(f"-> Raw Data Loaded. Total Rows: {len(df)}")
    
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
    return df

# ==============================================================================
# 2. GLOBAL CLEANING LOGIC (إنشاء الجداول المنظفة العامة)
# ==============================================================================
def create_global_clean_bases(df, tax_col):
    logger.info(f"\n--- GENERATING GLOBAL CLEAN DATA FOR: {tax_col} ---")
    
    # -----------------------
    # GLOBAL BASE 1
    # -----------------------
    cols_to_num = ['profit_before_tax', tax_col]
    for c in cols_to_num:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    gb1 = df.dropna(subset=['profit_before_tax', tax_col]).copy()
    gb1 = gb1[gb1['profit_before_tax'] > 0]
    gb1['ETR'] = gb1[tax_col] / gb1['profit_before_tax']
    gb1 = gb1[(gb1['ETR'] >= ETR_MIN) & (gb1['ETR'] < ETR_MAX)]
    
    gb1['ln_profits'] = np.log(gb1['profit_before_tax'])
    gb1['ETR_sq'] = gb1['ETR'] ** 2
    
    logger.info(f"-> Global Base 1 ({tax_col}) Created. Valid Firms: {len(gb1)}")

    # -----------------------
    # GLOBAL BASE 2
    # -----------------------
    controls = ['employees', 'tangible_assets', 'related_revenues']
    valid_controls = [c for c in controls if c in df.columns]
    
    # تحويل Controls إلى أرقام
    for c in valid_controls:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Strict Listwise Deletion
    gb2 = df.dropna(subset=['profit_before_tax', tax_col] + valid_controls).copy()
    
    gb2 = gb2[gb2['profit_before_tax'] > 0]
    gb2['ETR'] = gb2[tax_col] / gb2['profit_before_tax']
    gb2 = gb2[(gb2['ETR'] >= ETR_MIN) & (gb2['ETR'] < ETR_MAX)]
    
    gb2['ln_profits'] = np.log(gb2['profit_before_tax'])
    gb2['ETR_sq'] = gb2['ETR'] ** 2
    
    logged_controls = []
    for c in valid_controls:
        gb2 = gb2[gb2[c] > 0] # Ensure positive for Log
        gb2[f'ln_{c}'] = np.log(gb2[c])
        logged_controls.append(f'ln_{c}')
        
    logger.info(f"-> Global Base 2 ({tax_col}) Created. Valid Firms: {len(gb2)}")
    
    return gb1, gb2, logged_controls

# ==============================================================================
# 3. REGRESSION
# ==============================================================================
def run_regression(df, formula, title):
    logger.info(f"Running Regression: {title}")
    out = f"\n\n{'-'*20}\n{title}\n{'-'*20}\n"
    try:
        model = smf.ols(formula, data=df).fit()
        out += model.summary().as_text()
        
        if 'ETR_sq' in model.params:
            b1, b2 = model.params['ETR'], model.params['ETR_sq']
            tp = -b1 / (2*b2) if b2 != 0 else 0
            min_etr, max_etr = df['ETR'].min(), df['ETR'].max()
            is_valid = min_etr <= tp <= max_etr
            out += f"\n\n>> U-TEST: Coeff ETR^2={b2:.4f}, TP={tp:.4%}, In Range? {'YES' if is_valid else 'NO'}"
    except Exception as e:
        logger.error(f"Regression Failed: {e}")
        out += f"\nRegression Failed: {str(e)}"
    return out

# ==============================================================================
# 4. MAIN execution for GERMANY
# ==============================================================================
def main():
    try:
        df_raw = load_and_standardize_data(INPUT_FILE)
    except Exception as e:
        logger.critical(f"Execution stopped: {e}")
        return

    all_sheets = {}
    report_lines = []

    # --- PROCESS ACCRUED TAX ---
    g_acc_b1, g_acc_b2, acc_ctrls = create_global_clean_bases(df_raw, 'tax_accrued')
    
    # Save Global Data
    all_sheets['Global_Accrued_BASE_1'] = g_acc_b1
    all_sheets['Global_Accrued_BASE_2'] = g_acc_b2
    
    # Filter for GERMANY
    de_acc_b1 = g_acc_b1[g_acc_b1['upe_name'] == TARGET_COUNTRY]
    de_acc_b2 = g_acc_b2[g_acc_b2['upe_name'] == TARGET_COUNTRY]
    
    all_sheets['DE_Accrued_BASE_1'] = de_acc_b1
    all_sheets['DE_Accrued_BASE_2'] = de_acc_b2
    
    report_lines.append(f"*** {TARGET_COUNTRY} ACCRUED TAX ***")
    if len(de_acc_b1) > 10:
        report_lines.append(run_regression(de_acc_b1, "ln_profits ~ ETR", "DE Accrued B1 Linear"))
        report_lines.append(run_regression(de_acc_b1, "ln_profits ~ ETR + ETR_sq", "DE Accrued B1 Non-Linear"))
        
    if len(de_acc_b2) > 10:
        ctrls = " + ".join(acc_ctrls)
        report_lines.append(run_regression(de_acc_b2, f"ln_profits ~ ETR + {ctrls}", "DE Accrued B2 Linear"))
        report_lines.append(run_regression(de_acc_b2, f"ln_profits ~ ETR + ETR_sq + {ctrls}", "DE Accrued B2 Non-Linear"))
    else:
        logger.warning(f"Not enough data for DE Accrued Base 2 (N={len(de_acc_b2)})")

    # --- PROCESS PAID TAX ---
    g_paid_b1, g_paid_b2, paid_ctrls = create_global_clean_bases(df_raw, 'tax_paid')
    
    # Save Global Data
    all_sheets['Global_Paid_BASE_1'] = g_paid_b1
    all_sheets['Global_Paid_BASE_2'] = g_paid_b2
    
    # Filter for GERMANY
    de_paid_b1 = g_paid_b1[g_paid_b1['upe_name'] == TARGET_COUNTRY]
    de_paid_b2 = g_paid_b2[g_paid_b2['upe_name'] == TARGET_COUNTRY]
    
    all_sheets['DE_Paid_BASE_1'] = de_paid_b1
    all_sheets['DE_Paid_BASE_2'] = de_paid_b2
    
    report_lines.append(f"\n*** {TARGET_COUNTRY} PAID TAX ***")
    if len(de_paid_b1) > 10:
        report_lines.append(run_regression(de_paid_b1, "ln_profits ~ ETR", "DE Paid B1 Linear"))
        report_lines.append(run_regression(de_paid_b1, "ln_profits ~ ETR + ETR_sq", "DE Paid B1 Non-Linear"))
        
    if len(de_paid_b2) > 10:
        ctrls = " + ".join(paid_ctrls)
        report_lines.append(run_regression(de_paid_b2, f"ln_profits ~ ETR + {ctrls}", "DE Paid B2 Linear"))
        report_lines.append(run_regression(de_paid_b2, f"ln_profits ~ ETR + ETR_sq + {ctrls}", "DE Paid B2 Non-Linear"))
    else:
        logger.warning(f"Not enough data for DE Paid Base 2 (N={len(de_paid_b2)})")

    # Save to Excel
    logger.info(f"Saving to {OUTPUT_FILE}...")
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        for name, data in all_sheets.items():
            data.to_excel(writer, sheet_name=name, index=False)
        
        pd.DataFrame("\n".join(report_lines).split('\n')).to_excel(writer, sheet_name='Regression_Output', index=False, header=False)
        
    logger.info("DONE.")

if __name__ == "__main__":
    main()