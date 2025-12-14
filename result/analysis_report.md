# Analysis of Multinational Enterprise Data in Germany (CbCR 2021)

## Introduction and Methodology

This report investigates two key aspects of the 2021 Country-by-Country Reporting (CbCR) data for Germany, based on provided guidelines:

1.  **Nonlinearity**: The existence of a nonlinear relationship between the Effective Tax Rate (ETR) and the natural logarithm of profits ($\ln(\text{Profits})$).
2.  **Imputation Impact**: The effect of data imputation on the dataset and subsequent results compared to using only complete cases.

**Data Used:**

*   **Original Data**: The EUTO Public CbCR Database 2021, filtered for Germany (jur\_code = DEU).
*   **Imputed Data**: Data after imputation using the `Accrued_Imputation` method, resulting in 178 observations.
*   **Control Variables**: $\ln(\text{employees})$, $\ln(\text{tangible\_assets})$, $\ln(\text{related\_revenues})$.

---

## 1. Evidence of Nonlinearity

### Scatter Plot: ETR vs $\ln(\text{Profits})$ with Linear and Quadratic Fits (Imputed Data)

*   **Analysis**:
    The blue line (Linear Fit) suggests a general positive relationship. However, the red line (Quadratic Fit) provides a better fit, especially at the low and high ends of the ETR range. This indicates a nonlinear relationship, specifically an inverted U-shape. Profits appear to decrease slightly at very low ETRs (potentially due to profit shifting) before gradually increasing.

### Residual Plot from Linear Regression (Imputed Data)

*   **Analysis**:
    The residuals exhibit a clear curved pattern around the zero line: positive at low ETRs, negative in the middle, and positive again at high ETRs. This pattern is strong evidence that a linear model is insufficient, and a squared term for ETR ($\text{ETR}^2$) should be included to correct for the deviation.

### Predicted vs Actual $\ln(\text{Profits})$ from Nonlinear Model (Imputed Data)

*   **Analysis**:
    The data points closely align with the red line (the line of perfect prediction), demonstrating a high degree of accuracy for the nonlinear model. This plot reinforces the significant improvement in model fit after adding the $\text{ETR}^2$ term and supports the findings from the U-test (negative and statistically significant coefficient for $\text{ETR}^2$).

---

## 2. Impact of Imputation

### Comparison of the Missing and Imputed Values (in ln)

*   **Analysis**:
    The bar chart displays the mean values of the variables in logarithmic form ($\ln$). Imputed data averages (orange) are slightly higher than the observed-only averages (blue) for most variables ($\ln(\text{employees})$, $\ln(\text{tangible\_assets})$, $\ln(\text{related\_revenues})$). This suggests that the imputation process filled missing values in a logical and consistent manner with the existing observations.

### Histogram and Boxplot of Profit Before Tax

*   **Analysis**:
    The original distribution (green) includes negative values and extreme outliers. After imputation (orange), the distribution becomes more concentrated around larger positive values and shows reduced variability on the negative end. The boxplot clearly indicates a significant reduction in outliers.

### Histogram and Boxplot of ETR

*   **Analysis**:
    The original data contains negative and extremely high ETR values (even >10) likely due to illogical calculations or missing data. After imputation, the ETR values are concentrated around 0.1–0.2 (reflecting realistic German tax rates of approximately 15–30%). The boxplot shows a substantial reduction in variance and outliers.

---

## Conclusion

*   **Nonlinearity is Confirmed**: The first three plots visually and statistically confirm a nonlinear (inverted U-shape) relationship between ETR and $\ln(\text{Profits})$, supporting the inclusion of $\text{ETR}^2$ and the application of the U-test.
*   **Imputation is Successful and Beneficial**: The imputation process improved the distribution of key variables ($\text{profit\_before\_tax}$ and ETR), reduced outliers, and increased the sample size, leading to a more accurate and reliable model.

**Recommendation**: The imputed dataset should be used for final analyses. It is advisable to report the results of the U-test to identify the turning point (point of diminishing returns) in the ETR-profit relationship.