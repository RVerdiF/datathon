"""
Streamlit monitoring dashboard for Passos Mágicos.
Displays model metrics, feature importance, and data drift analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Passos Mágicos - Monitoramento",
    page_icon="🎓",
    layout="wide",
)

# --- Helper functions ---

@st.cache_data
def load_json(filepath: str) -> dict:
    """Load a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV file."""
    return pd.read_csv(filepath)


def find_file(pattern: str) -> str | None:
    """Find a file in common locations."""
    for base in [".", "/app"]:
        for p in Path(base).rglob(pattern):
            return str(p)
    return None


# --- Main Dashboard ---

st.title("🎓 Passos Mágicos — Dashboard de Monitoramento")
st.markdown("Monitoramento do modelo de predição de risco de defasagem escolar")
st.divider()

# --- Sidebar ---

st.sidebar.title("Navegação")
page = st.sidebar.radio(
    "Seção",
    ["📊 Métricas do Modelo", "📈 Feature Importances", "🔍 Análise de Drift", "📋 Dados"],
)

# --- Metrics Page ---

if page == "📊 Métricas do Modelo":
    st.header("📊 Métricas do Modelo")

    metrics_path = find_file("metrics.json")
    report_path = find_file("evaluation_report.json")

    if metrics_path:
        metrics = load_json(metrics_path)

        # Model info
        model_name = metrics.get("model_name", "N/A")
        st.info(f"**Modelo selecionado:** `{model_name}`")

        # CV Results
        cv = metrics.get("cv_results", {})
        if cv:
            st.subheader("Cross-Validation (Treino)")
            col1, col2, col3 = st.columns(3)
            col1.metric("F1-Score", f"{cv.get('f1_mean', 0):.4f}", f"± {cv.get('f1_std', 0):.4f}")
            col2.metric("Accuracy", f"{cv.get('accuracy_mean', 0):.4f}", f"± {cv.get('accuracy_std', 0):.4f}")
            col3.metric("ROC-AUC", f"{cv.get('roc_auc_mean', 0):.4f}", f"± {cv.get('roc_auc_std', 0):.4f}")

        # Validation metrics
        val = metrics.get("validation_metrics", {})
        if val:
            st.subheader("Validação")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{val.get('accuracy', 0):.4f}")
            col2.metric("Precision", f"{val.get('precision', 0):.4f}")
            col3.metric("Recall", f"{val.get('recall', 0):.4f}")
            col4.metric("F1-Score", f"{val.get('f1_score', 0):.4f}")

        # Test metrics
        test = metrics.get("test_metrics", {})
        if test:
            st.subheader("Teste")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{test.get('accuracy', 0):.4f}")
            col2.metric("Precision", f"{test.get('precision', 0):.4f}")
            col3.metric("Recall", f"{test.get('recall', 0):.4f}")
            col4.metric("F1-Score", f"{test.get('f1_score', 0):.4f}")
            col5.metric("ROC-AUC", f"{test.get('roc_auc', 0):.4f}")

            # Confusion Matrix
            cm = test.get("confusion_matrix")
            if cm:
                st.subheader("Matriz de Confusão (Teste)")
                cm_df = pd.DataFrame(
                    cm,
                    index=["Real: Sem Risco", "Real: Risco"],
                    columns=["Pred: Sem Risco", "Pred: Risco"],
                )
                st.dataframe(cm_df, use_container_width=True)
    else:
        st.warning("Arquivo `metrics.json` não encontrado. Execute o pipeline de treinamento primeiro.")

# --- Feature Importance Page ---

elif page == "📈 Feature Importances":
    st.header("📈 Feature Importances")

    model_path = find_file("model.joblib")
    train_path = find_file("train.csv")

    if model_path and train_path:
        model = joblib.load(model_path)
        train_df = load_csv(train_path)
        feature_cols = [c for c in train_df.columns if c != "target"]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            importances = None

        if importances is not None:
            imp_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importância": importances,
            }).sort_values("Importância", ascending=True)

            st.bar_chart(imp_df.set_index("Feature"), horizontal=True, height=500)

            st.subheader("Tabela Detalhada")
            st.dataframe(
                imp_df.sort_values("Importância", ascending=False).reset_index(drop=True),
                use_container_width=True,
            )
        else:
            st.info("O modelo não suporta extração de importância de features.")
    else:
        st.warning("Modelo ou dados de treino não encontrados.")

# --- Drift Analysis Page ---

elif page == "🔍 Análise de Drift":
    st.header("🔍 Análise de Drift")

    drift_path = find_file("drift_report.json")

    if drift_path:
        drift = load_json(drift_path)

        # Summary
        is_drifted = drift.get("drift_detected", False)
        n_drifted = drift.get("n_drifted", 0)
        n_total = drift.get("n_total_features", 0)

        if is_drifted:
            st.error(f"⚠️ Drift detectado em **{n_drifted}/{n_total}** features!")
            st.warning(f"Features com drift: `{', '.join(drift.get('drifted_features', []))}`")
        else:
            st.success(f"✅ Nenhum drift detectado ({n_total} features analisadas)")

        st.caption(f"Referência: {drift.get('n_reference', 'N/A')} amostras | "
                   f"Atual: {drift.get('n_current', 'N/A')} amostras | "
                   f"Threshold: {drift.get('threshold', 'N/A')}")

        # Feature details
        features = drift.get("features", {})
        if features:
            st.subheader("Detalhes por Feature")

            drift_df = pd.DataFrame([
                {
                    "Feature": name,
                    "Drift Score": info["drift_score"],
                    "Drifted": "🔴" if info["is_drifted"] else "🟢",
                    "Ref Mean": info["ref_mean"],
                    "Cur Mean": info["cur_mean"],
                    "Ref Std": info["ref_std"],
                    "Cur Std": info["cur_std"],
                }
                for name, info in features.items()
            ]).sort_values("Drift Score", ascending=False)

            st.dataframe(drift_df, use_container_width=True, hide_index=True)

            # Drift scores chart
            st.subheader("Drift Scores")
            chart_df = drift_df[["Feature", "Drift Score"]].set_index("Feature").sort_values("Drift Score", ascending=True)
            st.bar_chart(chart_df, horizontal=True, height=400)

    else:
        st.info("Nenhum relatório de drift encontrado. Execute `monitoring/drift_detection.py` primeiro.")

        if st.button("🔄 Executar Detecção de Drift"):
            try:
                from monitoring.drift_detection import run_drift_check
                results = run_drift_check()
                st.rerun()
            except Exception as e:
                st.error(f"Erro: {e}")

# --- Data Page ---

elif page == "📋 Dados":
    st.header("📋 Visão dos Dados")

    train_path = find_file("train.csv")
    val_path = find_file("val.csv")
    test_path = find_file("test.csv")

    if train_path:
        train_df = load_csv(train_path)
        val_df = load_csv(val_path) if val_path else pd.DataFrame()
        test_df = load_csv(test_path) if test_path else pd.DataFrame()

        col1, col2, col3 = st.columns(3)
        col1.metric("Treino", f"{len(train_df)} amostras")
        col2.metric("Validação", f"{len(val_df)} amostras")
        col3.metric("Teste", f"{len(test_df)} amostras")

        # Class distribution
        st.subheader("Distribuição de Classes (Treino)")
        target_counts = train_df["target"].value_counts().rename({0: "Sem Risco", 1: "Risco"})
        st.bar_chart(target_counts)

        # Feature stats
        st.subheader("Estatísticas das Features")
        st.dataframe(train_df.describe().T, use_container_width=True)

        # Preview
        st.subheader("Preview dos Dados")
        tab1, tab2, tab3 = st.tabs(["Treino", "Validação", "Teste"])
        with tab1:
            st.dataframe(train_df.head(20), use_container_width=True)
        with tab2:
            st.dataframe(val_df.head(20), use_container_width=True)
        with tab3:
            st.dataframe(test_df.head(20), use_container_width=True)
    else:
        st.warning("Dados processados não encontrados. Execute o pipeline de pré-processamento primeiro.")

# --- Footer ---
st.divider()
st.caption("Passos Mágicos — Datathon FIAP | Dashboard de Monitoramento v1.0")
