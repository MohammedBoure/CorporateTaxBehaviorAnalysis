import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")  # إخفاء التحذيرات البسيطة

# إنشاء مجلد جديد للرسوم المصححة بالكامل
os.makedirs('plots_final', exist_ok=True)

# قراءة البيانات (تأكد من أن الملفات في نفس المجلد)
original_df = pd.read_excel('EUTO_Public_CbCR_Database_2021.xlsx', sheet_name='Public_CbCRs')
cc_accrued_df = pd.read_excel('Germany_Baseline_Results_Full.xlsx', sheet_name='DE_Accrued_BASE_1')
imputed_accrued_df = pd.read_excel('Germany_Imputation_Results.xlsx', sheet_name='Accrued_Imputation')
cc_paid_df = pd.read_excel('Germany_Imputation_Results.xlsx', sheet_name='Paid_CC')
imputed_paid_df = pd.read_excel('Germany_Imputation_Results.xlsx', sheet_name='Paid_Imputation')

# دالة تنظيف آمنة
def clean_df(df):
    if df.empty:
        return df
    numeric_cols = ['profit_before_tax', 'tax_accrued', 'employees', 'tangible_assets', 'related_revenues',
                    'ETR', 'ln_profits', 'ln_employees', 'ln_tangible_assets', 'ln_related_revenues', 'ETR_sq']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['ln_profits', 'ETR'], how='any')
    return df.copy()

cc_accrued_df = clean_df(cc_accrued_df)
imputed_accrued_df = clean_df(imputed_accrued_df)
cc_paid_df = clean_df(cc_paid_df)
imputed_paid_df = clean_df(imputed_paid_df)

# استخدام Imputed كأساس رئيسي (لأن CC Accrued غالباً فارغ)
main_df = imputed_accrued_df if len(imputed_accrued_df) > len(cc_accrued_df) else cc_accrued_df

# 1. Scatter Plot مع خطوط الانحدار
plt.figure(figsize=(12, 8))
sns.scatterplot(data=main_df, x='ETR', y='ln_profits', alpha=0.7, color='darkblue')
sns.regplot(data=main_df, x='ETR', y='ln_profits', scatter=False, color='blue', label='Linear Fit', ci=None)
sns.regplot(data=main_df, x='ETR', y='ln_profits', order=2, scatter=False, color='red', label='Quadratic Fit', ci=None)
plt.title('ETR vs ln(Profits): Linear and Quadratic Fits', fontsize=16)
plt.xlabel('Effective Tax Rate (ETR)', fontsize=14)
plt.ylabel('ln(Profits)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots_final/01_scatter_etr_lnprofits.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Residual Plot و Predicted vs Actual (إذا توفرت المتغيرات التحكمية)
controls = ['ln_employees', 'ln_tangible_assets', 'ln_related_revenues']
if all(c in main_df.columns for c in controls):
    try:
        X_lin = sm.add_constant(main_df[['ETR'] + controls])
        y = main_df['ln_profits']
        model_lin = sm.OLS(y, X_lin).fit()
        residuals = model_lin.resid

        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=main_df['ETR'], y=residuals, alpha=0.7, color='darkblue')
        plt.axhline(0, color='red', linestyle='--')
        plt.title('Residuals vs ETR (Linear Model)', fontsize=16)
        plt.xlabel('ETR', fontsize=14)
        plt.ylabel('Residuals', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots_final/02_residual_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

        X_nonlin = sm.add_constant(main_df[['ETR', 'ETR_sq'] + controls])
        model_nonlin = sm.OLS(y, X_nonlin).fit()
        predicted = model_nonlin.fittedvalues

        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=y, y=predicted, alpha=0.7, color='darkblue')
        plt.plot(y.sort_values(), y.sort_values(), color='red', linestyle='--')
        plt.title('Predicted vs Actual ln(Profits) (Nonlinear Model)', fontsize=16)
        plt.xlabel('Actual', fontsize=14)
        plt.ylabel('Predicted', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots_final/03_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
    except:
        pass

# 3. مقارنة Coefficients و SE/t-stats (باستخدام Paid إذا توفرت)
def safe_ols_nonlin(df):
    if len(df) < 20 or not all(c in df.columns for c in controls):
        return None
    try:
        X = sm.add_constant(df[['ETR', 'ETR_sq'] + controls])
        y = df['ln_profits']
        return sm.OLS(y, X).fit()
    except:
        return None

model_cc = safe_ols_nonlin(cc_paid_df)
model_imp = safe_ols_nonlin(imputed_paid_df)

if model_cc is not None and model_imp is not None:
    params = ['const', 'ETR', 'ETR_sq', 'ln_employees', 'ln_tangible_assets', 'ln_related_revenues']
    df_coef = pd.DataFrame({
        'CC': model_cc.params.reindex(params, fill_value=0),
        'Imputed': model_imp.params.reindex(params, fill_value=0)
    })

    df_coef.plot(kind='bar', figsize=(14, 8), alpha=0.8, color=['steelblue', 'darkorange'])
    plt.title('Nonlinear Coefficients: Complete Cases vs Imputed (Paid Tax)', fontsize=16)
    plt.ylabel('Coefficient Value', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('plots_final/04_coefficients_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # منحنيات متراكبة
    etr_range = np.linspace(0, 0.5, 200)
    mean_controls = imputed_paid_df[controls].mean()
    X_pred = sm.add_constant(pd.DataFrame({
        'ETR': etr_range,
        'ETR_sq': etr_range**2,
        'ln_employees': mean_controls['ln_employees'],
        'ln_tangible_assets': mean_controls['ln_tangible_assets'],
        'ln_related_revenues': mean_controls['ln_related_revenues']
    }))

    pred_cc = model_cc.predict(X_pred)
    pred_imp = model_imp.predict(X_pred)

    plt.figure(figsize=(12, 8))
    plt.plot(etr_range, pred_cc, label='Complete Cases Curve', color='steelblue', linewidth=3)
    plt.plot(etr_range, pred_imp, label='Imputed Curve', color='darkorange', linewidth=3)
    plt.title('Predicted ln(Profits) vs ETR: CC vs Imputed', fontsize=16)
    plt.xlabel('ETR', fontsize=14)
    plt.ylabel('Predicted ln(Profits)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots_final/05_overlaid_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. توزيعات Original vs Imputed (مع تحقق آمن)
original_germany = original_df[original_df['jur_code'] == 'DEU'].copy()
original_germany['ETR_calc'] = pd.to_numeric(original_germany['tax_accrued'], errors='coerce') / pd.to_numeric(original_germany['profit_before_tax'], errors='coerce')

for var, title in [('profit_before_tax', 'Profit Before Tax'), ('ETR_calc', 'Effective Tax Rate (ETR)')]:
    orig_data = original_germany[var].dropna()
    imp_data = main_df[var.replace('_calc', '')] if var != 'ETR_calc' else main_df['ETR']
    imp_data = imp_data.dropna()

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Histogram
    if not orig_data.empty:
        sns.histplot(orig_data, kde=True, ax=axs[0], color='green', alpha=0.6, label='Original')
    if not imp_data.empty:
        sns.histplot(imp_data, kde=True, ax=axs[0], color='orange', alpha=0.6, label='Imputed')
    axs[0].set_title(f'Histogram: {title}', fontsize=16)
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Boxplot
    box_data = []
    box_labels = []
    box_colors = []
    if not orig_data.empty:
        box_data.append(orig_data)
        box_labels.append('Original')
        box_colors.append('green')
    if not imp_data.empty:
        box_data.append(imp_data)
        box_labels.append('Imputed')
        box_colors.append('orange')
    if box_data:
        sns.boxplot(data=box_data, ax=axs[1], palette=box_colors)
        axs[1].set_xticks(range(len(box_labels)))
        axs[1].set_xticklabels(box_labels)
    axs[1].set_title(f'Boxplot: {title}', fontsize=16)
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'plots_final/06_dist_{var.replace("_calc", "")}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("تم إنشاء جميع الرسوم البيانية بنجاح في مجلد 'plots_final'")
print("الرسوم المتولدة:")
print(os.listdir('plots_final'))