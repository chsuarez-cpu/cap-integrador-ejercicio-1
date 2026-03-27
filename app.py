import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Modelo M/M/1", layout="wide")

# =========================
# FUNCIONES
# =========================
def mm1_metrics(lmbda, mu):
    rho = lmbda / mu
    if rho >= 1:
        return None

    P0 = 1 - rho
    L = rho / (1 - rho)
    Lq = rho**2 / (1 - rho)
    W = 1 / (mu - lmbda)
    Wq = rho / (mu - lmbda)

    return rho, P0, L, Lq, W, Wq


def prob_n(rho, n):
    return (1 - rho) * rho**n


def prob_geq_n(rho, n):
    return rho**n


def prob_wait_greater(lmbda, mu, t):
    return np.exp(-(mu - lmbda) * t)


def simulate_mm1(lmbda, mu, n=1000):
    arrivals = np.cumsum(np.random.exponential(1/lmbda, n))
    service_times = np.random.exponential(1/mu, n)

    start = np.zeros(n)
    end = np.zeros(n)
    wait = np.zeros(n)

    for i in range(n):
        if i == 0:
            start[i] = arrivals[i]
        else:
            start[i] = max(arrivals[i], end[i-1])

        wait[i] = start[i] - arrivals[i]
        end[i] = start[i] + service_times[i]

    return pd.DataFrame({
        "Llegada": arrivals,
        "Inicio": start,
        "Fin": end,
        "Espera": wait
    })


# =========================
# INTERFAZ
# =========================
st.title("📞 Centro de Emergencias - M/M/1")

lmbda = st.number_input("Tasa de llegada (λ)", value=18.0)
mu = st.number_input("Tasa de servicio (μ)", value=24.0)
t_umbral = st.number_input("Tiempo umbral (min)", value=8.0)
n_prob = st.number_input("Número mínimo de llamadas", value=4)

# =========================
# RESULTADOS
# =========================
metrics = mm1_metrics(lmbda, mu)

if metrics is None:
    st.error("⚠️ Sistema inestable (λ ≥ μ)")
else:
    rho, P0, L, Lq, W, Wq = metrics

    st.subheader("📊 Métricas")

    st.write(f"Utilización (ρ): {round(rho,3)}")
    st.write(f"P0: {round(P0,3)}")
    st.write(f"L: {round(L,3)}")
    st.write(f"Lq: {round(Lq,3)}")
    st.write(f"W (min): {round(W*60,2)}")
    st.write(f"Wq (min): {round(Wq*60,2)}")

    p_geq = prob_geq_n(rho, n_prob)
    p_wait = prob_wait_greater(lmbda, mu, t_umbral/60)

    st.write(f"P(n ≥ {n_prob}) = {round(p_geq,4)}")
    st.write(f"P(espera > {t_umbral} min) = {round(p_wait,4)}")

    # =========================
    # GRÁFICO Pn
    # =========================
    n_vals = np.arange(0, 10)
    p_vals = [prob_n(rho, n) for n in n_vals]

    fig, ax = plt.subplots()
    ax.bar(n_vals, p_vals)
    ax.set_title("Distribución Pn")
    st.pyplot(fig)

    # =========================
    # SIMULACIÓN
    # =========================
    st.subheader("🎲 Simulación")

    n_sim = st.slider("Número de simulaciones", 100, 5000, 1000)
    df = simulate_mm1(lmbda, mu, n_sim)

    st.dataframe(df.head())

    st.write("Espera promedio simulada (min):",
             round(df["Espera"].mean()*60, 2))

    fig2, ax2 = plt.subplots()
    ax2.hist(df["Espera"]*60, bins=30)
    ax2.set_title("Distribución de espera")
    st.pyplot(fig2)

    # =========================
    # INTERPRETACIÓN
    # =========================
    st.subheader("🧠 Interpretación")

    if rho < 0.7:
        st.success("Sistema eficiente")
    elif rho < 0.85:
        st.warning("Sistema moderado")
    else:
        st.error("Sistema crítico (alto riesgo)")

    if Wq*60 > 5:
        st.error("Tiempo de espera elevado para emergencias")