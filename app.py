import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Modelo M/M/1", layout="wide")

# =========================
# FUNCIONES
# =========================
def mm1_metrics(lmbda, mu):
    if mu <= lmbda:
        return None

    rho = lmbda / mu
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
st.title("📞 Centro de Emergencias - Modelo M/M/1")

st.sidebar.header("Parámetros")

lmbda = st.sidebar.number_input("Tasa de llegada (λ)", min_value=0.1, value=18.0)
mu = st.sidebar.number_input("Tasa de servicio (μ)", min_value=0.1, value=24.0)
t_umbral = st.sidebar.number_input("Tiempo umbral (min)", min_value=0.1, value=8.0)
n_prob = st.sidebar.number_input("Número mínimo de llamadas", min_value=1, value=4)

# =========================
# RESULTADOS
# =========================
metrics = mm1_metrics(lmbda, mu)

if metrics is None:
    st.error("⚠️ Sistema inestable: λ ≥ μ (la cola crece indefinidamente)")
else:
    rho, P0, L, Lq, W, Wq = metrics

    st.subheader("📊 Métricas del sistema")

    col1, col2, col3 = st.columns(3)
    col1.metric("Utilización (ρ)", round(rho, 3))
    col2.metric("L (sistema)", round(L, 3))
    col3.metric("Lq (cola)", round(Lq, 3))

    col4, col5 = st.columns(2)
    col4.metric("W (min)", round(W*60, 2))
    col5.metric("Wq (min)", round(Wq*60, 2))

    # Probabilidades
    st.subheader("📈 Probabilidades")

    p_geq = prob_geq_n(rho, n_prob)
    p_wait = prob_wait_greater(lmbda, mu, t_umbral/60)

    st.write(f"Probabilidad de n ≥ {n_prob}: {round(p_geq,4)}")
    st.write(f"Probabilidad de espera > {t_umbral} min: {round(p_wait,4)}")

    # =========================
    # GRÁFICO Pn
    # =========================
    st.subheader("📊 Distribución de estados")

    n_vals = np.arange(0, 10)
    p_vals = [prob_n(rho, n) for n in n_vals]

    fig, ax = plt.subplots()
    ax.bar(n_vals, p_vals)
    ax.set_xlabel("Número de llamadas")
    ax.set_ylabel("Probabilidad")
    ax.set_title("Distribución Pn")
    st.pyplot(fig)

    # =========================
    # SIMULACIÓN
    # =========================
    st.subheader("🎲 Simulación Monte Carlo")

    n_sim = st.slider("Número de simulaciones", 100, 5000, 1000)

    df = simulate_mm1(lmbda, mu, n_sim)

    st.dataframe(df.head())

    st.write("Espera promedio simulada (min):",
             round(df["Espera"].mean()*60, 2))

    fig2, ax2 = plt.subplots()
    ax2.hist(df["Espera"]*60, bins=30)
    ax2.set_title("Distribución de tiempos de espera")
    st.pyplot(fig2)

    # =========================
    # INTERPRETACIÓN
    # =========================
    st.subheader("🧠 Interpretación automática")

    if rho < 0.7:
        st.success("Sistema eficiente")
    elif rho < 0.85:
        st.warning("Sistema moderadamente cargado")
    else:
        st.error("Sistema crítico - alto riesgo")

    if Wq*60 > 5:
        st.error("Tiempo de espera elevado para emergencias")