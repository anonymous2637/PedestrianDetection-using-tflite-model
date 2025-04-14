import pandas as pd
from datetime import datetime
from openpyxl import load_workbook

excel_file = "detections.xlsx"

# Create file if not exists
if not pd.io.common.file_exists(excel_file):
    df = pd.DataFrame(columns=["Date", "Time", "Person Count"])
    df.to_excel(excel_file, index=False, sheet_name="Detections")

def save_to_excel(person_count):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_12hr = now.strftime("%I:%M:%S %p")

    try:
        workbook = load_workbook(excel_file)
        sheet = workbook.active
        sheet.append([date, time_12hr, person_count])
        workbook.save(excel_file)
    except Exception as e:
        print(f"Error updating Excel file: {e}")
