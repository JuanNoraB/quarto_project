#%%
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "/home/juanchx/Desktop/Maestria_IA/machine learning/lab 1 data/mdt_colocados_2025agosto.csv"

# Cargar y limpiar
df = pd.read_csv(DATA_PATH, sep=";", encoding="latin1")
df = df.rename(
    columns={
        "A¤o": "Año",
        "Cant¢n": "Cantón",
        "N£mero de Personas": "Número de Personas",
    }
)

# Provincias con más y menos contrataciones
provincia_totales = df.groupby("Provincia")["Número de Personas"].sum().sort_values(ascending=False)
top4 = provincia_totales.head(4).index.tolist()
bottom4 = provincia_totales.tail(4).sort_values().index.tolist()


def plot_provincias(provincias, titulo, ranking_label):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    fig.suptitle(titulo, fontsize=14, weight="bold")

    for ax, provincia in zip(axes.flatten(), provincias):
        serie = (
            df[df["Provincia"] == provincia]
            .groupby("Cantón")["Número de Personas"]
            .sum()
            .sort_values(ascending=False)
        )
        ax.bar(serie.index, serie.values, color="#4C72B0")
        ax.set_title(provincia, fontsize=10)
        ax.tick_params(axis="x", rotation=70, labelsize=8)
        if ax in (axes[0, 0], axes[1, 0]):
            ax.set_ylabel("Personas")
        else:
            ax.set_ylabel("")
        ax.set_xlabel("Cantón", fontsize=9)

        total_provincia = provincia_totales.loc[provincia]
        ax.text(
            0.98,
            0.95,
            f"Total: {total_provincia:,}",
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="right",
            va="top",
        )

        posicion = provincias.index(provincia) + 1
        ax.text(
            0.98,
            0.85,
            f"{ranking_label} {posicion}",
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="top",
        )

        for patch in ax.patches:
            altura = patch.get_height()
            if altura > 0:
                ax.text(
                    patch.get_x() + patch.get_width() / 2,
                    altura,
                    f"{int(altura):,}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    for ax in axes.flatten()[len(provincias):]:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


plot_provincias(top4, "Provincias con más contrataciones", "Top")
plot_provincias(bottom4, "Provincias con menos contrataciones", "Bottom")
plt.show()



#%%
# Participación de contrataciones por provincia (top 8 + "Otros")
TOP_N = 8
pie_datos = provincia_totales.reset_index().rename(columns={"Provincia": "Provincia", "Número de Personas": "Personas"})
top_provincias = pie_datos.head(TOP_N)
resto_total = pie_datos["Personas"][TOP_N:].sum()

if resto_total > 0:
    pie_data = pd.concat(
        [top_provincias, pd.DataFrame({"Provincia": ["Otros"], "Personas": [resto_total]})],
        ignore_index=True,
    )
else:
    pie_data = top_provincias


def _autopct(values):
    total = values.sum()

    def inner(pct):
        valor = int(round(total * pct / 100.0))
        return f"{pct:.1f}%\n{valor:,}"

    return inner


plt.figure(figsize=(8, 8))
plt.pie(
    pie_data["Personas"],
    labels=pie_data["Provincia"],
    autopct=_autopct(pie_data["Personas"]),
    startangle=90,
    pctdistance=0.78,
    textprops={"fontsize": 9},
)
plt.title("Participación de contrataciones por provincia", fontsize=14, weight="bold")
centre_circle = plt.Circle((0, 0), 0.60, fc="white")
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.tight_layout()
plt.show()

# %%
