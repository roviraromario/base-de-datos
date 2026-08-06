import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sqlalchemy import create_engine


CARPETA_GRAFICAS = "graficas"

# Paleta de colores usada en todas las gráficas
COLOR_AGRESION = "#D9534F"
COLOR_ENTRENAMIENTO = "#2E5EAA"
COLOR_OK = "#5CB85C"

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


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


# ---------------------------------------------------------------------------
# PREGUNTA 1: ¿Cómo influye el nivel de agresión y la preparación
# (training_level) en la tasa de devolución (returned)?
# ---------------------------------------------------------------------------
def graficar_agresion_entrenamiento(df):

    df["aggression_bin"] = pd.cut(
        df["aggression_score"], bins=[-0.1, 2, 4, 6, 8, 10.1],
        labels=["0-2", "2-4", "4-6", "6-8", "8-10"]
    )
    df["training_bin"] = pd.cut(
        df["training_level"], bins=[-0.1, 2, 4, 6, 8, 10.1],
        labels=["0-2", "2-4", "4-6", "6-8", "8-10"]
    )

    tasa_agresion = df.groupby("aggression_bin", observed=True)["returned"].mean() * 100
    tasa_entrenamiento = df.groupby("training_bin", observed=True)["returned"].mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(tasa_agresion.index.astype(str), tasa_agresion.values, color=COLOR_AGRESION)
    axes[0].set_title("Tasa de devolución según\npuntaje de agresión")
    axes[0].set_xlabel("Puntaje de agresión (0-10)")
    axes[0].set_ylabel("Tasa de devolución (%)")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    for i, v in enumerate(tasa_agresion.values):
        axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)

    axes[1].bar(tasa_entrenamiento.index.astype(str), tasa_entrenamiento.values, color=COLOR_ENTRENAMIENTO)
    axes[1].set_title("Tasa de devolución según\nnivel de entrenamiento")
    axes[1].set_xlabel("Nivel de entrenamiento (0-10)")
    axes[1].set_ylabel("Tasa de devolución (%)")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    for i, v in enumerate(tasa_entrenamiento.values):
        axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)

    fig.suptitle("Pregunta 1 · Agresión y preparación como factores de devolución", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{CARPETA_GRAFICAS}/pregunta1_agresion_entrenamiento.png", bbox_inches="tight")
    plt.close(fig)

    # Combinación de riesgo: alta agresión + bajo entrenamiento
    alto_riesgo = df[(df["aggression_score"] >= 6) & (df["training_level"] < 4)]
    bajo_riesgo = df[(df["aggression_score"] < 2) & (df["training_level"] >= 6)]
    print("\n--- Pregunta 1: Agresión y entrenamiento ---")
    print(f"Alta agresión + bajo entrenamiento (n={len(alto_riesgo)}): "
          f"{alto_riesgo['returned'].mean() * 100:.1f}% de devolución")
    print(f"Baja agresión + alto entrenamiento (n={len(bajo_riesgo)}): "
          f"{bajo_riesgo['returned'].mean() * 100:.1f}% de devolución")


# ---------------------------------------------------------------------------
# PREGUNTA 2: ¿Es la incompatibilidad de energía combinada con altas
# expectativas (expectation_score) el mayor predictor de fracaso?
# ---------------------------------------------------------------------------
def graficar_expectativas_energia(df):

    df["expect_bin"] = pd.cut(
        df["expectation_score"], bins=[-0.1, 4, 6, 8, 10.1],
        labels=["Baja\n(0-4)", "Media\n(4-6)", "Alta\n(6-8)", "Muy alta\n(8-10)"]
    )
    df["mismatch_bin"] = pd.cut(
        df["energy_mismatch"], bins=[-0.1, 2, 4, 6, 10.1],
        labels=["Bajo\n(0-2)", "Moderado\n(2-4)", "Alto\n(4-6)", "Muy alto\n(6-10)"]
    )

    pivote = df.pivot_table(
        index="expect_bin", columns="mismatch_bin", values="returned",
        aggfunc="mean", observed=True
    ) * 100

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(pivote.values, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(pivote.columns)))
    ax.set_xticklabels(pivote.columns)
    ax.set_yticks(range(len(pivote.index)))
    ax.set_yticklabels(pivote.index)
    ax.set_xlabel("Desajuste de energía (perro vs adoptante)")
    ax.set_ylabel("Nivel de expectativas del adoptante")

    valores_validos = pivote.values[~np.isnan(pivote.values)]
    umbral = valores_validos.mean() if len(valores_validos) else 0
    for i in range(pivote.shape[0]):
        for j in range(pivote.shape[1]):
            val = pivote.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        color="white" if val > umbral else "black", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Tasa de devolución (%)")
    ax.set_title("Pregunta 2 · Expectativas altas + desajuste de energía", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{CARPETA_GRAFICAS}/pregunta2_expectativas_energia.png", bbox_inches="tight")
    plt.close(fig)

    combinado = df[(df["expectation_score"] >= 8) & (df["energy_mismatch"] >= 6)]
    print("\n--- Pregunta 2: Expectativas y desajuste de energía ---")
    print(f"Expectativa >=8 y desajuste >=6 (n={len(combinado)}): "
          f"{combinado['returned'].mean() * 100:.1f}% de devolución")


# ---------------------------------------------------------------------------
# PREGUNTA 3: ¿Qué tan rápido se rompe una adopción según el perfil
# del adoptante (first_time_owner) y cuál es la distribución de days_to_return?
# ---------------------------------------------------------------------------
def graficar_factor_tiempo(df):

    devueltos = df[df["returned"] == 1].copy()
    devueltos["perfil"] = np.where(
        devueltos["first_time_owner"] == 1, "Dueño primerizo", "Con experiencia previa"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    datos_por_perfil = [
        devueltos.loc[devueltos["perfil"] == p, "days_to_return"].values
        for p in ["Dueño primerizo", "Con experiencia previa"]
    ]
    bp = axes[0].boxplot(
        datos_por_perfil, tick_labels=["Dueño\nprimerizo", "Con experiencia\nprevia"],
        patch_artist=True, showfliers=False
    )
    for patch, color in zip(bp["boxes"], [COLOR_AGRESION, COLOR_ENTRENAMIENTO]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_title("Días hasta la devolución\nsegún experiencia del adoptante")
    axes[0].set_ylabel("Días hasta la devolución")

    bins = [0, 3, 7, 14, 30, 60, 90, 365]
    etiquetas = ["0-3", "4-7", "8-14", "15-30", "31-60", "61-90", "90+"]
    devueltos["periodo"] = pd.cut(devueltos["days_to_return"], bins=bins, labels=etiquetas)
    conteo_periodo = devueltos["periodo"].value_counts(sort=False, normalize=True) * 100

    axes[1].plot(etiquetas, conteo_periodo.values, marker="o", color=COLOR_AGRESION, linewidth=2)
    axes[1].fill_between(etiquetas, conteo_periodo.values, alpha=0.2, color=COLOR_AGRESION)
    axes[1].set_title("¿Cuándo ocurren las devoluciones?\n(distribución por días desde la adopción)")
    axes[1].set_xlabel("Días desde la adopción")
    axes[1].set_ylabel("% de las devoluciones")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())

    fig.suptitle("Pregunta 3 · Factor tiempo en la ruptura de la adopción", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{CARPETA_GRAFICAS}/pregunta3_factor_tiempo.png", bbox_inches="tight")
    plt.close(fig)

    print("\n--- Pregunta 3: Factor tiempo ---")
    print(f"Mediana días hasta devolución (primerizo): "
          f"{devueltos.loc[devueltos['perfil'] == 'Dueño primerizo', 'days_to_return'].median():.1f}")
    print(f"Mediana días hasta devolución (con experiencia): "
          f"{devueltos.loc[devueltos['perfil'] == 'Con experiencia previa', 'days_to_return'].median():.1f}")
    print(f"% de devoluciones que ocurren en <=30 días: "
          f"{(devueltos['days_to_return'] <= 30).mean() * 100:.1f}%")


# ---------------------------------------------------------------------------
# PREGUNTA 4: ¿Qué combinaciones de vivienda (home_type) y tiempo a solas
# (hours_alone_per_day) disparan las devoluciones por ansiedad?
# ---------------------------------------------------------------------------
def graficar_espacio_soledad(df):

    df["hours_bin"] = pd.cut(
        df["hours_alone_per_day"], bins=[-0.1, 4, 8, 12, 20],
        labels=["0-4h", "4-8h", "8-12h", "12h+"]
    )
    df["devolucion_ansiedad"] = (df["return_reason"] == "separation_anxiety_destruction").astype(int)

    etiquetas_vivienda = {
        "apartment": "Departamento",
        "house_rented": "Casa alquilada",
        "house_owned": "Casa propia",
    }
    pivote = df.pivot_table(
        index="home_type", columns="hours_bin", values="devolucion_ansiedad",
        aggfunc="mean", observed=True
    ) * 100
    pivote = pivote.rename(index=etiquetas_vivienda)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(pivote.columns))
    ancho = 0.25
    colores = [COLOR_AGRESION, "#F0AD4E", COLOR_OK]
    for i, (idx, fila) in enumerate(pivote.iterrows()):
        ax.bar(x + i * ancho, fila.values, ancho, label=idx, color=colores[i % len(colores)])

    ax.set_xticks(x + ancho)
    ax.set_xticklabels(pivote.columns)
    ax.set_ylabel("Tasa de devolución por ansiedad/destrucción (%)")
    ax.set_xlabel("Horas solo por día")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("Pregunta 4 · Espacio y soledad como disparadores\nde devolución por ansiedad", fontweight="bold")
    ax.legend(title="Tipo de vivienda")
    fig.tight_layout()
    fig.savefig(f"{CARPETA_GRAFICAS}/pregunta4_espacio_soledad.png", bbox_inches="tight")
    plt.close(fig)

    print("\n--- Pregunta 4: Espacio y soledad ---")
    print(pivote.round(1))


# ---------------------------------------------------------------------------
# PREGUNTA 5: ¿Cuánto protegen realmente las visitas previas
# (visits_before_adoption) y el contacto con otras mascotas (met_resident_pets)?
# ---------------------------------------------------------------------------
def graficar_protocolo_refugio(df):

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    tasa_visitas = df.groupby("visits_before_adoption")["returned"].mean() * 100
    axes[0].bar(tasa_visitas.index.astype(str), tasa_visitas.values, color=COLOR_ENTRENAMIENTO)
    axes[0].set_title("Tasa de devolución según\nvisitas previas al refugio")
    axes[0].set_xlabel("Número de visitas antes de adoptar")
    axes[0].set_ylabel("Tasa de devolución (%)")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    for i, v in enumerate(tasa_visitas.values):
        axes[0].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)

    tasa_mascotas = df.groupby("met_resident_pets")["returned"].mean() * 100
    tasa_asesoria = df.groupby("adoption_counseling")["returned"].mean() * 100

    x = np.arange(2)
    ancho = 0.35
    axes[1].bar(x - ancho / 2, tasa_mascotas.values, ancho,
                label="Contacto con mascotas\nresidentes", color=COLOR_OK)
    axes[1].bar(x + ancho / 2, tasa_asesoria.values, ancho,
                label="Asesoría de adopción", color="#5BC0DE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["No (0)", "Sí (1)"])
    axes[1].set_ylabel("Tasa de devolución (%)")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].set_title("Efecto de los factores protectores\ndel protocolo del refugio")
    axes[1].legend(fontsize=8)

    fig.suptitle("Pregunta 5 · Eficacia del protocolo del refugio", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{CARPETA_GRAFICAS}/pregunta5_protocolo_refugio.png", bbox_inches="tight")
    plt.close(fig)

    print("\n--- Pregunta 5: Protocolo del refugio ---")
    print("Tasa de devolución según visitas previas:")
    print(tasa_visitas.round(1))
    print(f"Conoció mascotas residentes: {tasa_mascotas[1]:.1f}% vs no conoció: {tasa_mascotas[0]:.1f}%")
    print(f"Con asesoría: {tasa_asesoria[1]:.1f}% vs sin asesoría: {tasa_asesoria[0]:.1f}%")


df = limpiar_datos(df) 

print(df)

# Save the clean data to your desktop folder
df.to_csv("cleaned_adoptions_staging.csv", index=False)
print("Clean CSV file exported successfully! Open DBeaver now.")

# Crear carpeta de salida para las gráficas
os.makedirs(CARPETA_GRAFICAS, exist_ok=True)

# Generar las gráficas que responden cada pregunta del análisis
graficar_agresion_entrenamiento(df)
graficar_expectativas_energia(df)
graficar_factor_tiempo(df)
graficar_espacio_soledad(df)
graficar_protocolo_refugio(df)

print(f"\nTodas las gráficas se guardaron en la carpeta '{CARPETA_GRAFICAS}/'")


 