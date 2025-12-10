import pandas as pd
import xml.etree.ElementTree as ET
import os

INPUT_FILE = 'Cbcr_data.xls'      
TARGET_COUNTRY = 'Germany'

def diagnose_data(file_path):
    print("--- BDIA (Basic Data Inspection) ---")
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        data = []
        for sheet in root.iter():
            if sheet.tag.endswith('Worksheet'):
                name = sheet.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Name', '')
                if 'CbCR' in name or 'Sheet1' in name:
                    for row in sheet.iter():
                        if row.tag.endswith('Row'):
                            row_data = []
                            for cell in row.iter():
                                if cell.tag.endswith('Cell'):
                                    cell_val = None
                                    for data_tag in cell.iter():
                                        if data_tag.tag.endswith('Data'):
                                            cell_val = data_tag.text
                                            break
                                    row_data.append(cell_val)
                            data.append(row_data)
                    break
        
        if not data:
            print("no data get.")
            return

        headers = [str(h).strip() for h in data[0]]
        df = pd.DataFrame(data[1:], columns=headers)
        
        if 'upe_name' in df.columns:
            df_germany = df[df['upe_name'] == TARGET_COUNTRY].copy()
            print(f"Total Companies found for {TARGET_COUNTRY}: {len(df_germany)}")
            
            cols_to_check = ['profit_before_tax', 'tax_accrued', 'employees', 'tangible_assets', 'related_revenues']
            
            print("\n--- Missing Values Count ---")
            for col in cols_to_check:
                if col in df_germany.columns:
                    numeric_vals = pd.to_numeric(df_germany[col], errors='coerce')
                    missing_count = numeric_vals.isnull().sum()
                    total_rows = len(df_germany)
                    print(f"{col}: {missing_count} missing out of {total_rows}")
                else:
                    print(f"{col}: COLUMN NOT FOUND!")
        else:
            print("Column 'upe_name' not found.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    diagnose_data(INPUT_FILE)