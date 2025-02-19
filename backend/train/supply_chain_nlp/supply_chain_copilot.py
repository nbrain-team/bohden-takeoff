import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import fastapi
from fastapi import FastAPI
import uvicorn

# Load data
df = pd.read_csv("C:\\Users\\Dell\\Downloads\\construction-optimisation\\construction_suppliers.csv")
print(df.columns)


# Text Preprocessing: Combine relevant columns
df["combined_features"] = df["Supplier Name"] + " " + df["Material Type"] + " " + df["Location"]


# Convert text data into numerical vectors (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["combined_features"])

# Train Nearest Neighbors model
model = NearestNeighbors(n_neighbors=5, metric="cosine")
model.fit(X)

# Save model & vectorizer
joblib.dump(model, "supplier_recommendation_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# FastAPI for serving recommendations
app = FastAPI()

@app.get("/recommend/")
def recommend_supplier(query: str):
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("supplier_recommendation_model.pkl")
    
    query_vector = vectorizer.transform([query])
    distances, indices = model.kneighbors(query_vector)

    results = []
    for i in range(len(indices[0])):
        supplier_idx = indices[0][i]
        results.append({
            "Supplier Name": df.iloc[supplier_idx]["Supplier Name"],
            "Material Type": df.iloc[supplier_idx]["Material Type"],
            "Location": df.iloc[supplier_idx]["Location"],
            "Distance Score": round(distances[0][i], 3)
        })
    
    return {"Recommendations": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
