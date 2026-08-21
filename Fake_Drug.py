import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# I LOAD GENUINE DATASET FROM EXCEL
df_genuine = pd.read_excel("NAFDAC_Registered_Products_Database.xlsx")
df_genuine["label"] = 0  # 0 = Genuine

# Fill missing values
df_genuine["Product Name"] = df_genuine["Product Name"].fillna("").astype(str)
df_genuine["NRN (NAFDAC Reg No)"] = (
    df_genuine["NRN (NAFDAC Reg No)"].fillna("").astype(str)
)

# I GENERATE SYNTHETIC COUNTERFEIT DATASET (100 Entries)
np.random.seed(42)
fake_prefixes = ["XX-", "99-", "NAFDAC-", "00-00", "ZZ-"]
fake_records = []

for _, row in df_genuine.iterrows():
    brand = str(row["Product Name"])

    fake_nrn = fake_prefixes[
        np.random.randint(0, len(fake_prefixes))
    ] + str(np.random.randint(100, 999))
    fake_brand = (
        brand.replace("a", "4")
        .replace("o", "0")
        .replace("e", "3")
        .replace("i", "1")
    )
    if fake_brand == brand:
        fake_brand += " Counterfeit"

    fake_records.append(
        {
            "Product Name": fake_brand,
            "NRN (NAFDAC Reg No)": fake_nrn,
            "label": 1,  # 1 = Suspicious / Counterfeit
        }
    )

df_suspicious = pd.DataFrame(fake_records)

# I Combine datasets into a balanced set (200 records total)
df_combined = pd.concat(
    [
        df_genuine[["Product Name", "NRN (NAFDAC Reg No)", "label"]],
        df_suspicious,
    ],
    ignore_index=True,
)

# FEATURE EXTRACTION (TF-IDF VECTORIZER)
df_combined["full_text"] = (
    df_combined["NRN (NAFDAC Reg No)"] + " " + df_combined["Product Name"]
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(df_combined["full_text"])
y = df_combined["label"]

# TRAIN AND EVALUATE MODEL
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation Report
y_pred = model.predict(X_test)
print("=== Model Evaluation Report ===")
print(classification_report(y_test, y_pred))


# UNIFIED PREDICTION FUNCTION (TEXT & BARCODE)
def verify_drug_product(entry_input, brand_name=""):
   
    input_str = f"{entry_input} {brand_name}".strip()

    # Direct database verification
    exact_db_match = df_genuine[
        df_genuine["NRN (NAFDAC Reg No)"].str.strip().str.lower()
        == entry_input.strip().lower()
    ]

    # Scikit-Learn Model Prediction
    vec_input = vectorizer.transform([input_str])
    pred = model.predict(vec_input)[0]
    prob = model.predict_proba(vec_input)[0]

    if not exact_db_match.empty:
        matched_row = exact_db_match.iloc[0]
        return {
            "Input Entry": entry_input,
            "Flag": "GENUINE (VERIFIED DB RECORD)",
            "Product Name": matched_row["Product Name"],
            "Category": matched_row["Product Category"],
            "Status": matched_row["Status"],
        }
    elif pred == 1:
        return {
            "Input Entry": entry_input,
            "Flag": "SUSPICIOUS / COUNTERFEIT",
            "Confidence": f"{prob[1] * 100:.1f}%",
            "Explanation": "Entry displays structural or spelling anomalies matching fake product patterns.",
        }
    else:
        return {
            "Input Entry": entry_input,
            "Flag": "UNVERIFIED ENTRY",
            "Confidence": f"{prob[0] * 100:.1f}%",
            "Explanation": "Format appears standard, but record is not found in the official registration sheet.",
        }


# --- TESTING THE EXAMPLE---
print("\n=== Test 1: Genuine Database Record ===")
print(verify_drug_product("03-1450"))

print("\n=== Test 2: Fake/Suspicious Barcode or NRN Entry ===")
print(verify_drug_product("NAFDAC-999", "Paracetam0l 500mg"))
