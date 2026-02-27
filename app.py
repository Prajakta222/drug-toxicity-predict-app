import streamlit as st
import pickle
import pandas as pd

st.title("?? Drug Toxicity Prediction App")

st.markdown("""
Enter the SMILES string of a drug to predict whether it is **Safe** or **Toxic**.
""")

# Load the trained model
with open("toxicity_model.pkl", "rb") as f:
    model = pickle.load(f)

# Minimal featurization (placeholder)
def featurize(smiles):
    """
    This is a placeholder for feature extraction without RDKit.
    Uses SMILES length and character counts as features.
    """
    length = len(smiles)
    num_c = smiles.count("C")
    num_o = smiles.count("O")
    num_n = smiles.count("N")
    return [length, num_c, num_o, num_n]

# Single prediction
smiles = st.text_input("Enter Drug SMILES:")

if st.button("Predict"):
    if smiles:
        features = featurize(smiles)
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][1]  # probability of Toxic
        result = "Toxic ??" if prediction else "Safe ?"
        st.success(f"Prediction: {result}")
        st.info(f"Toxicity Probability: {probability:.2f}")
    else:
        st.warning("?? Please enter a SMILES string")

# CSV batch prediction
uploaded_file = st.file_uploader("Or upload CSV (with column 'SMILES')", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    if "SMILES" not in data.columns:
        st.error("CSV must have a 'SMILES' column")
    else:
        predictions = []
        probabilities = []
        for smi in data["SMILES"]:
            features = featurize(smi)
            pred = model.predict([features])[0]
            prob = model.predict_proba([features])[0][1]
            predictions.append("Toxic ??" if pred else "Safe ?")
            probabilities.append(prob)
        data["Prediction"] = predictions
        data["Probability"] = probabilities
        st.dataframe(data)
        st.download_button("Download Results", data.to_csv(index=False), file_name="predictions.csv")