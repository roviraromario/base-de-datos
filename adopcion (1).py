
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT = "graficas_simple"
os.makedirs(OUT, exist_ok=True)


df = pd.read_csv("dog_adoption_master.csv")



def limpiar_datos(df):


    df["size"] = df["size"].str.lower()
    df["breed_group"] = df["breed_group"].str.lower()
    df["intake_type"] = df["intake_type"].str.lower()
    df["medical_needs"] = df["medical_needs"].str.lower()
    df["home_type"] = df["home_type"].str.lower()
    df["return_reason"] = df["return_reason"].str.lower()

    
    df["age_years"] = pd.to_numeric(df["age_years"], errors='coerce')
    df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors='coerce')
    df["neutered"] = pd.to_numeric(df["neutered"], errors='coerce')
    df["aggression_score"] = pd.to_numeric(df["aggression_score"], errors='coerce')
    df["anxiety_separation"] = pd.to_numeric(df["anxiety_separation"], errors='coerce')
    df["energy_level"] = pd.to_numeric(df["energy_level"], errors='coerce')
    df["training_level"] = pd.to_numeric(df["training_level"], errors='coerce')
    df["house_trained"] = pd.to_numeric(df["house_trained"], errors='coerce')
    df["first_time_owner"] = pd.to_numeric(df["first_time_owner"], errors='coerce')
    df["household_has_kids"] = pd.to_numeric(df["household_has_kids"], errors='coerce')
    df["household_has_pets"] = pd.to_numeric(df["household_has_pets"], errors='coerce')
    df["has_yard"] = pd.to_numeric(df["has_yard"], errors='coerce')
    df["adopter_activity_level"] = pd.to_numeric(df["adopter_activity_level"], errors='coerce')
    df["visits_before_adoption"] = pd.to_numeric(df["visits_before_adoption"], errors='coerce')
    df["met_resident_pets"] = pd.to_numeric(df["met_resident_pets"], errors='coerce')
    df["adoption_counseling"] = pd.to_numeric(df["adoption_counseling"], errors='coerce')
    df["expectation_score"] = pd.to_numeric(df["expectation_score"], errors='coerce')
    df["energy_mismatch"] = pd.to_numeric(df["energy_mismatch"], errors='coerce')
    df["size_home_mismatch"] = pd.to_numeric(df["size_home_mismatch"], errors='coerce')
    df["days_to_return"] = pd.to_numeric(df["days_to_return"], errors='coerce')
    df["returned"] = pd.to_numeric(df["returned"], errors='coerce')


    df.fillna({"days_to_return": 0}, inplace=True)
    df.fillna({"medical_needs": "without_needs"}, inplace=True)
    df.replace("none", "without_needs", inplace=True)


    df = df.drop_duplicates().dropna()
    
    return df





def plot_agresion_entrenamiento(df):
    if not all(c in df.columns for c in ["aggression_score", "training_level", "returned"]):
        return
    bins = [-0.1, 2, 4, 6, 8, 10.1]
    labels = ["0-2","2-4","4-6","6-8","8-10"]
    df["ag_bin"] = pd.cut(df["aggression_score"], bins=bins, labels=labels)
    df["tr_bin"] = pd.cut(df["training_level"], bins=bins, labels=labels)
    a = df.groupby("ag_bin")["returned"].mean()*100
    t = df.groupby("tr_bin")["returned"].mean()*100

    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].bar(a.index.astype(str), a.values, color="#D9534F")
    ax[0].set_title("Devolución por agresión (%)")
    ax[0].set_ylim(0, max(15, a.max() if len(a) else 0)+5)

    ax[1].bar(t.index.astype(str), t.values, color="#2E5EAA")
    ax[1].set_title("Devolución por entrenamiento (%)")

    plt.tight_layout()
    plt.savefig(f"{OUT}/agresion_entrenamiento.png")
    plt.close()

    high_risk = df[(df["aggression_score"] >= 6) & (df["training_level"] < 4)]
    low_risk = df[(df["aggression_score"] < 2) & (df["training_level"] >= 6)]
    print("Alta agresión + bajo entrenamiento:", len(high_risk), "casos ->", f"{high_risk['returned'].mean()*100:.1f}%")
    print("Baja agresión + alto entrenamiento:", len(low_risk), "casos ->", f"{low_risk['returned'].mean()*100:.1f}%")


def plot_expectativas_energia(df):
    if not all(c in df.columns for c in ["expectation_score", "energy_mismatch", "returned"]):
        return
    e_bins = pd.cut(df["expectation_score"], bins=[-0.1,4,6,8,10.1], labels=["Baja","Media","Alta","Muy alta"])
    m_bins = pd.cut(df["energy_mismatch"], bins=[-0.1,2,4,6,10.1], labels=["Bajo","Medio","Alto","Muy alto"])
    tab = (df.assign(e=e_bins, m=m_bins)
             .groupby(["e","m"])["returned"]
             .mean()
             .unstack(fill_value=np.nan)*100)
    plt.figure(figsize=(6,4))
    plt.imshow(tab.values, cmap="Reds", aspect="auto")
    plt.xticks(range(len(tab.columns)), tab.columns)
    plt.yticks(range(len(tab.index)), tab.index)
    plt.colorbar(label="Devolución (%)")
    plt.title("Expectativas vs desajuste de energía")
    plt.tight_layout()
    plt.savefig(f"{OUT}/expectativas_energia.png")
    plt.close()
    # ejemplo numérico
    high_combo = df[(df["expectation_score"] >= 8) & (df["energy_mismatch"] >= 6)]
    print("Expectativa >=8 y desajuste >=6:", len(high_combo), "casos ->", f"{high_combo['returned'].mean()*100:.1f}%")


 


plot_agresion_entrenamiento(df)
plot_expectativas_energia(df)



df.to_csv("cleaned_adoptions_simple.csv", index=False)
print("Listo: gráficas guardadas en", OUT, "y CSV limpio como cleaned_adoptions_simple.csv")
