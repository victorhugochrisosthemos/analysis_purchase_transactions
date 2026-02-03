import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer

import networkx as nx


# -----------------------------
# Config Streamlit
# -----------------------------
st.set_page_config(page_title="Varejo - Regras, Clusters e Temporal", layout="wide")

st.title("Dashboard de Análises de Trasações de compras")


# -----------------------------
# Funções 
# -----------------------------
def split_products(product_string: str):
    if pd.isna(product_string):
        return []
    return [p.strip() for p in str(product_string).split(",") if p.strip()]


def validate_columns(df: pd.DataFrame):
    required = {"CustomerID", "Products", "Timestamp"}
    missing = required - set(df.columns)
    return missing


@st.cache_data(show_spinner=False)
def prepare_transactions(df: pd.DataFrame):
    df = df.copy()
    df["ProductList"] = df["Products"].apply(split_products)

    te = TransactionEncoder()
    te_array = te.fit(df["ProductList"]).transform(df["ProductList"])
    onehot = pd.DataFrame(te_array, columns=te.columns_)

    return df, onehot


@st.cache_data(show_spinner=False)
def compute_association_rules(onehot: pd.DataFrame, min_support: float, min_lift: float):
    freq = apriori(onehot, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="lift", min_threshold=min_lift)
    rules_sorted = rules.sort_values(["lift", "confidence"], ascending=[False, False])
    return freq, rules, rules_sorted


@st.cache_data(show_spinner=False)
def compute_customer_features(df: pd.DataFrame):
    # lista única de produtos por cliente
    compras_de_clientes = (
        df.groupby("CustomerID")["ProductList"]
        .agg(lambda x: list(set([item for sublist in x for item in sublist])))
        .reset_index()
    )

    frequencia_compras = df.groupby("CustomerID").size().reset_index(name="frequencia_compras")

    compras_de_clientes["total_produtos_unicos"] = compras_de_clientes["ProductList"].apply(len)
    total_produtos_unicos = compras_de_clientes[["CustomerID", "total_produtos_unicos"]]

    features = pd.merge(frequencia_compras, total_produtos_unicos, on="CustomerID")

    # binarização de produtos
    mlb = MultiLabelBinarizer()
    product_binary = pd.DataFrame(
        mlb.fit_transform(compras_de_clientes["ProductList"]),
        columns=mlb.classes_
    )

    features = pd.merge(features, product_binary, left_index=True, right_index=True)
    return features


@st.cache_data(show_spinner=False)
def run_kmeans(features: pd.DataFrame, n_clusters: int, random_state: int = 42):
    features_for_clustering = features.drop("CustomerID", axis=1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features_for_clustering)

    features_out = features.copy()
    features_out["Cluster"] = labels

    # características médias por cluster
    cluster_characteristics = features_out.groupby("Cluster").mean(numeric_only=True)
    return features_out, cluster_characteristics


@st.cache_data(show_spinner=False)
def temporal_aggregations(df: pd.DataFrame):
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])

    df["Mes"] = df["Timestamp"].dt.month
    df["Ano"] = df["Timestamp"].dt.year
    df["Dia da semana"] = df["Timestamp"].dt.dayofweek

    # transações mensais e total de produtos
    transacoes_mensais = df.groupby("Mes").size().reset_index(name="total_transacoes")
    vendas_mensais = df.groupby("Mes")["ProductList"].apply(lambda x: sum(len(i) for i in x)).reset_index(name="total_produtos_vendidos")
    total_mensal = pd.merge(transacoes_mensais, vendas_mensais, on="Mes")

    # diário
    transacoes_diarias = df.groupby("Dia da semana").size().reset_index(name="total_transacoes")
    vendas_diarias = df.groupby("Dia da semana")["ProductList"].apply(lambda x: sum(len(i) for i in x)).reset_index(name="total_produtos_vendidos")
    total_diario = pd.merge(transacoes_diarias, vendas_diarias, on="Dia da semana")

    # volume por produto e mês
    df_exploded = df.explode("ProductList")
    volume_mensal = df_exploded.groupby(["Mes", "ProductList"]).size().reset_index(name="volume_vendas")

    return df, total_mensal, total_diario, volume_mensal


def fig_scatter_clusters(features_out: pd.DataFrame, jitter=True):
    x = features_out["frequencia_compras"].to_numpy()
    y = features_out["total_produtos_unicos"].to_numpy()
    c = features_out["Cluster"].to_numpy()

    if jitter:
        rng = np.random.default_rng(42)
        x = x + rng.normal(0, 0.15, size=len(x))
        y = y + rng.normal(0, 0.08, size=len(y))

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=70, alpha=0.6, edgecolors="k", linewidths=0.4)

    ax.set_title("Clusters de Clientes (KMeans)")
    ax.set_xlabel("Frequência de Compras")
    ax.set_ylabel("Total de Produtos Únicos")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.colorbar(sc, ax=ax, label="Cluster")
    return fig


def fig_cluster_bars(cluster_characteristics: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_characteristics[["frequencia_compras", "total_produtos_unicos"]].plot(kind="bar", ax=ax, colormap="viridis")
    ax.set_title("Características Médias dos Clusters")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Média dos Valores")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(["Frequência de Compras", "Total de Produtos Únicos"])
    return fig


def fig_rules_bar(rules_sorted: pd.DataFrame, top_n=10):
    top = rules_sorted.head(top_n).copy()
    top["antecedents"] = top["antecedents"].apply(lambda s: ", ".join(list(s)))
    top["consequents"] = top["consequents"].apply(lambda s: ", ".join(list(s)))
    top["rule"] = top["antecedents"] + " → " + top["consequents"]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top, y="rule", x="lift", ax=ax)
    ax.set_title(f"Top {top_n} Regras (Lift)")
    ax.set_xlabel("Lift")
    ax.set_ylabel("Regra")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    return fig


# -----------------------------
# Layout radial por frequência
# -----------------------------
def radial_positions_by_frequency(G, freq_map, center_node, r_min=0.7, r_max=4.0, seed=42):
    """
    centro = produto mais frequente
    raio = inversamente proporcional à frequência:
      - mais frequente -> mais perto do centro (raio ~ r_min)
      - menos frequente -> mais longe (raio ~ r_max)
    """
    nodes = list(G.nodes())
    others = [n for n in nodes if n != center_node]

    if not others:
        return {center_node: np.array([0.0, 0.0])}

    freqs = np.array([max(1, freq_map.get(n, 1)) for n in others], dtype=float)

    fmin, fmax = freqs.min(), freqs.max()
    if fmax - fmin < 1e-9:
        norm = np.zeros_like(freqs)
    else:
        norm = (freqs - fmin) / (fmax - fmin)

    # norm=1 (mais freq) -> r_min ; norm=0 (menos freq) -> r_max
    radii = r_max - norm * (r_max - r_min)

    # usa spring só pra definir ângulos bons (sem controlar distância)
    spring = nx.spring_layout(G, seed=seed)
    angles = [np.arctan2(spring[n][1], spring[n][0]) for n in others]

    pos = {center_node: np.array([0.0, 0.0])}
    for n, r, ang in zip(others, radii, angles):
        pos[n] = np.array([r * np.cos(ang), r * np.sin(ang)])

    return pos


def fig_network_rules(rules: pd.DataFrame, df_base: pd.DataFrame, lift_thr: float, conf_thr: float,
                      r_min=0.7, r_max=4.0):
    """
    Ajustado:
    - Nó central = produto mais frequente (no dataset)
    - Distância ao centro = quanto maior a frequência do produto, mais perto do centro
    - Tamanho do nó = proporcional à frequência (com central maior)
    """
    filtered = rules[(rules["lift"] > lift_thr) & (rules["confidence"] > conf_thr)].copy()
    if filtered.empty:
        return None

    # Frequência de produtos no dataset (quantas vezes aparece nas transações)
    # df_base precisa ter ProductList
    freq_map = df_base.explode("ProductList")["ProductList"].value_counts().to_dict()

    G = nx.DiGraph()
    for _, row in filtered.iterrows():
        ant = list(row["antecedents"])
        con = list(row["consequents"])
        if len(ant) == 0 or len(con) == 0:
            continue
        a = ant[0]
        b = con[0]
        G.add_edge(a, b, lift=row["lift"], confidence=row["confidence"], support=row["support"])

    if G.number_of_nodes() == 0:
        return None

    # Nó central: produto mais frequente entre os nós do grafo
    nodes_in_graph = list(G.nodes())
    main_node = max(nodes_in_graph, key=lambda n: freq_map.get(n, 0))

    # Posições radiais por frequência
    pos = radial_positions_by_frequency(G, freq_map, center_node=main_node, r_min=r_min, r_max=r_max, seed=42)

    # Tamanho dos nós por frequência (e central maior)
    node_sizes = []
    for n in G.nodes():
        f = freq_map.get(n, 1)
        size = 700 + 55 * np.sqrt(f)     # ajuste aqui se quiser maior/menor
        if n == main_node:
            size *= 2.2                  # central bem maior
        node_sizes.append(size)

    # cores das arestas por lift
    lifts = np.array([G[u][v]["lift"] for u, v in G.edges()])
    norm = (lifts - lifts.min()) / (lifts.max() - lifts.min() + 1e-8)
    cmap = plt.cm.coolwarm
    edge_colors = [cmap(v) for v in norm]

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color="#0077b6",
        edgecolors="#023047",
        linewidths=2,
        alpha=0.95
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors,
        width=4,
        arrowsize=18,
        alpha=0.85
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=10,
        font_weight="bold",
        font_color="black"
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=lifts.min(), vmax=lifts.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Lift (Força da Associação)")

    ax.set_title(f"Grafo de Regras (centro = mais frequente | Lift>{lift_thr} e Confiança>{conf_thr})")
    ax.axis("off")
    plt.tight_layout()
    return fig


def fig_monthly_trend(total_mensal: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=total_mensal, x="Mes", y="total_transacoes", marker="o", ax=ax, label="total_transacoes")
    sns.lineplot(data=total_mensal, x="Mes", y="total_produtos_vendidos", marker="o", ax=ax, label="total_produtos_vendidos")
    ax.set_title("Tendência Mensal: Transações e Produtos Vendidos")
    ax.set_xlabel("Mês")
    ax.set_ylabel("Quantidade")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xticks(sorted(total_mensal["Mes"].unique()))
    return fig


def fig_daily_bar(total_diario: pd.DataFrame):
    daily_melted = total_diario.melt(id_vars="Dia da semana", var_name="Metric", value_name="Count")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=daily_melted, x="Dia da semana", y="Count", hue="Metric", ax=ax)
    ax.set_title("Tendência Diária: Transações e Produtos Vendidos")
    ax.set_xlabel("Dia da semana")
    ax.set_ylabel("Quantidade")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_xticks(sorted(total_diario["Dia da semana"].unique()))
    ax.set_xticklabels(["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"])
    return fig


def fig_top10_products_monthly(volume_mensal: pd.DataFrame, top_n=10):
    overall = volume_mensal.groupby("ProductList")["volume_vendas"].sum().sort_values(ascending=False)
    top_products = overall.head(top_n).index.tolist()
    df_top = volume_mensal[volume_mensal["ProductList"].isin(top_products)]

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=df_top, x="Mes", y="volume_vendas", hue="ProductList", marker="o", ax=ax)
    ax.set_title(f"Volume mensal de vendas - Top {top_n} produtos")
    ax.set_xlabel("Mês")
    ax.set_ylabel("Volume de vendas")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xticks(sorted(df_top["Mes"].unique()))
    ax.legend(title="Produto", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return fig


# -----------------------------
# Sidebar - upload e parâmetros
# -----------------------------
st.sidebar.header("Configurações")

uploaded = st.sidebar.file_uploader("Envie seu CSV", type=["csv"])

min_support = st.sidebar.slider("Min support (Apriori)", 0.001, 0.1, 0.01, 0.001)
min_lift = st.sidebar.slider("Min lift (regras)", 1.0, 5.0, 1.2, 0.1)

lift_thr = st.sidebar.slider("Filtro grafo: Lift >", 1.0, 10.0, 2.5, 0.1)
conf_thr = st.sidebar.slider("Filtro grafo: Confiança >", 0.0, 1.0, 0.5, 0.05)

# NOVO: controla quão perto/longe ficam os nós do grafo
r_min = st.sidebar.slider("Raio mínimo (mais freq)", 0.2, 2.0, 0.7, 0.1)
r_max = st.sidebar.slider("Raio máximo (menos freq)", 2.0, 8.0, 4.0, 0.2)

n_clusters = st.sidebar.slider("N clusters (KMeans)", 2, 8, 4, 1)
top_rules_n = st.sidebar.slider("Top regras (tabela/gráfico)", 3, 30, 10, 1)

st.sidebar.divider()
st.sidebar.caption("Colunas esperadas: CustomerID, Products, Timestamp")


if uploaded is None:
    st.info("Envie um CSV na barra lateral para gerar automaticamente todos os gráficos.")
    st.stop()


# -----------------------------
# Leitura do CSV
# -----------------------------
df = pd.read_csv(uploaded)

missing = validate_columns(df)
if missing:
    st.error(f"Faltam colunas no CSV: {', '.join(missing)}")
    st.stop()

with st.spinner("Preparando dados..."):
    df, products_onehot = prepare_transactions(df)

# -----------------------------
# Layout por abas
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "1) Regras de Associação",
    "2) Segmentação de Clientes",
    "3) Análises Temporais",
    "4) Exportar"
])

# -----------------------------
# TAB 1 - Regras
# -----------------------------
with tab1:
    st.subheader("1) Regras de Associação")

    freq, rules, rules_sorted = compute_association_rules(products_onehot, min_support, min_lift)

    colA, colB = st.columns([1, 1])
    with colA:
        st.write("**Top regras (ordenadas por lift e confidence)**")
        st.dataframe(rules_sorted.head(top_rules_n), use_container_width=True)
    with colB:
        st.write("**Itemsets frequentes (primeiras linhas)**")
        st.dataframe(freq.head(10), use_container_width=True)

    st.write("**Gráfico: Top regras por Lift**")
    fig = fig_rules_bar(rules_sorted, top_n=top_rules_n)
    st.pyplot(fig)

    st.write("**Grafo das regras filtradas (centro = produto mais frequente)**")
    fig_g = fig_network_rules(rules, df_base=df, lift_thr=lift_thr, conf_thr=conf_thr, r_min=r_min, r_max=r_max)
    if fig_g is None:
        st.warning("Nenhuma regra passou no filtro do grafo. Tente reduzir Lift/Confiança na lateral.")
    else:
        st.pyplot(fig_g)


# -----------------------------
# TAB 2 - Clusters
# -----------------------------
with tab2:
    st.subheader("2) Segmentação de clientes (KMeans)")

    features = compute_customer_features(df)
    features_out, cluster_characteristics = run_kmeans(features, n_clusters=n_clusters)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("**Amostra de features**")
        st.dataframe(features_out.head(10), use_container_width=True)
    with col2:
        st.write("**Características médias por cluster**")
        st.dataframe(cluster_characteristics[["frequencia_compras", "total_produtos_unicos"]], use_container_width=True)

    st.write("**Gráfico: clusters (scatter)**")
    st.pyplot(fig_scatter_clusters(features_out, jitter=True))

    st.write("**Gráfico: médias dos clusters**")
    st.pyplot(fig_cluster_bars(cluster_characteristics))

    st.write("**Produtos mais comuns por cluster (top 5)**")
    base_cols = {"frequencia_compras", "total_produtos_unicos"}
    product_cols = [c for c in cluster_characteristics.columns if c not in base_cols]
    for c in sorted(cluster_characteristics.index):
        top5 = cluster_characteristics.loc[c, product_cols].sort_values(ascending=False).head(5)
        st.write(
            f"**Cluster {c}** — freq média: {cluster_characteristics.loc[c, 'frequencia_compras']:.2f} | "
            f"prod únicos médios: {cluster_characteristics.loc[c, 'total_produtos_unicos']:.2f}"
        )
        st.dataframe(top5.to_frame("proporção").T, use_container_width=True)


# -----------------------------
# TAB 3 - Temporal
# -----------------------------
with tab3:
    st.subheader("3) Análise temporal de vendas")

    df_t, total_mensal, total_diario, volume_mensal = temporal_aggregations(df)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("**Resumo mensal**")
        st.dataframe(total_mensal, use_container_width=True)
    with col2:
        st.write("**Resumo diário**")
        st.dataframe(total_diario, use_container_width=True)

    st.write("**Gráfico: tendência mensal**")
    st.pyplot(fig_monthly_trend(total_mensal))

    st.write("**Gráfico: tendência diária**")
    st.pyplot(fig_daily_bar(total_diario))

    st.write("**Gráfico: Top 10 produtos mais vendidos (mensal)**")
    st.pyplot(fig_top10_products_monthly(volume_mensal, top_n=10))

    st.write("**Top 10 produtos (geral)**")
    overall = volume_mensal.groupby("ProductList")["volume_vendas"].sum().sort_values(ascending=False)
    st.dataframe(overall.head(10).to_frame("volume_total"), use_container_width=True)

    st.write("**Bottom 10 produtos (geral)**")
    st.dataframe(overall.tail(10).to_frame("volume_total"), use_container_width=True)


# -----------------------------
# TAB 4 - Exportar
# -----------------------------
with tab4:
    st.subheader("4) Exportar resultados")

    _, rules, rules_sorted = compute_association_rules(products_onehot, min_support, min_lift)
    features = compute_customer_features(df)
    features_out, _ = run_kmeans(features, n_clusters=n_clusters)
    _, total_mensal, _, volume_mensal = temporal_aggregations(df)

    st.write("Você pode baixar tabelas em CSV:")

    st.download_button(
        "1. Baixar Top Regras",
        rules_sorted.head(top_rules_n).to_csv(index=False).encode("utf-8"),
        file_name="top_regras.csv",
        mime="text/csv"
    )

    st.download_button(
        "2. Baixar Features + Cluster",
        features_out.to_csv(index=False).encode("utf-8"),
        file_name="clientes_clusters.csv",
        mime="text/csv"
    )

    st.download_button(
        "3. Baixar Resumo Mensal",
        total_mensal.to_csv(index=False).encode("utf-8"),
        file_name="resumo_mensal.csv",
        mime="text/csv"
    )

    st.download_button(
        "4. Baixar Volume Mensal por Produto",
        volume_mensal.to_csv(index=False).encode("utf-8"),
        file_name="volume_mensal_produto.csv",
        mime="text/csv"
    )

# streamlit run t4-layout.py
