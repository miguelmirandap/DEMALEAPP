import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import io
import base64
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, r2_score, mean_absolute_error, mean_squared_error, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, learning_curve
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import seaborn as sns
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import shap
from datetime import datetime

PROJECT_DIR = Path(__file__).parent
DEFAULT_DATASET_PATH = PROJECT_DIR / "DEMALE-HSJM_2025_data.xlsx"
MODEL_PATH = PROJECT_DIR / "model.joblib"
META_PATH = PROJECT_DIR / "feature_metadata.json"
LOGO_PATH = PROJECT_DIR / "logo.png"
DICT_PATH = PROJECT_DIR / "DEMALE-HSJM_2025_dictionary.pdf"
EXP_DIR = PROJECT_DIR / "experiments"
EXP_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Oráculo DEMALE – IA de Predicción Clínica", page_icon="🧠", layout="wide")

# --- Minimal custom styling ---
st.markdown(
    """
    <style>
      :root {
        --brand:#10b981; /* emerald */
        --brand2:#0ea5e9; /* sky */
        --bg:#0b1221;
        --card:#0f172a; /* slate-900 */
        --card2:#111827; /* gray-900 */
        --muted:#9ca3af;
      }
      .stApp { background: radial-gradient(1000px 500px at 10% -10%, rgba(16,185,129,.15), transparent 60%),
                               radial-gradient(1000px 600px at 110% 10%, rgba(14,165,233,.12), transparent 60%), var(--bg) !important; }
      .stMetric { background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(14,165,233,0.12)); padding: 10px 14px; border-radius: 12px; border:1px solid rgba(255,255,255,.06); }
      .card { background: linear-gradient(180deg, rgba(255,255,255,.025), rgba(255,255,255,.015)); border:1px solid rgba(255,255,255,.06); padding:18px; border-radius:16px; box-shadow: 0 10px 30px rgba(0,0,0,.25); }
      .pill { display:inline-block; padding:4px 10px; border-radius:999px; background: rgba(16,185,129,.15); color:#d1fae5; font-size:.8rem; border:1px solid rgba(16,185,129,.35); }
      .hero { padding: 18px 20px; border-radius: 18px; border:1px solid rgba(255,255,255,.06);
              background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(14,165,233,0.15)); }
      .hero h1 { font-size: 2.1rem; background: linear-gradient(90deg, #6ee7b7, #93c5fd); -webkit-background-clip:text; background-clip:text; color:transparent; margin:0 0 8px; }
      .hero .sub { color: var(--muted); margin:0; }
      /* Animated accent */
      .glow {
        position: relative;
        overflow: hidden;
        border-radius: 14px;
      }
      .glow:before {
        content: "";
        position: absolute; inset: -2px;
        background: conic-gradient(from 90deg, rgba(16,185,129,.6), rgba(14,165,233,.6), rgba(16,185,129,.6));
        filter: blur(18px); opacity:.25; animation: spin 8s linear infinite;
      }
      @keyframes spin { to { transform: rotate(360deg); } }
      /* Floating CTA */
      .fab { position: fixed; right: 22px; bottom: 22px; background: linear-gradient(135deg, var(--brand), var(--brand2));
             color:white; padding: 12px 16px; border-radius: 999px; box-shadow: 0 12px 28px rgba(16,185,129,.35); cursor:pointer; z-index:9999; }
      .fab:hover { transform: translateY(-1px); }
      .block-title { font-size: 1.05rem; font-weight: 600; opacity: .85; margin: .25rem 0 .5rem; }
      .small { opacity: .75; font-size: .9rem; }
      /* Buttons */
      .stButton>button { background: linear-gradient(135deg, var(--brand), var(--brand2)); color:white; border:0; padding: .55rem 1rem; border-radius: 10px; }
      .stButton>button:hover { filter: brightness(1.05); box-shadow:0 8px 20px rgba(16,185,129,.25); }
      /* Tables */
      .stDataFrame { border-radius: 12px; overflow:hidden; }
      /* Sidebar */
      section[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.01)); border-right:1px solid rgba(255,255,255,.06); }
      /* Footer */
      .footer { text-align:center; color:var(--muted); margin-top: 18px; font-size:.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

def inject_top_right_logo(path: Path, *, width_px: int = 36, right_px: int = 70, top_px: int = 10):
    if not path.exists():
        return
    try:
        b64 = base64.b64encode(path.read_bytes()).decode()
        html = f"""
        <img src='data:image/png;base64,{b64}' style='position:fixed; top:{top_px}px; right:{right_px}px; width:{width_px}px; height:auto; z-index:1000; opacity:0.95;'/>
        """
        st.markdown(html, unsafe_allow_html=True)
    except Exception:
        pass

def render_theme_controls():
    st.sidebar.markdown("**Apariencia**")
    theme = st.sidebar.selectbox("Tema", ["Emerald/Sky", "Purple/Pink", "Orange/Amber"], index=0, help="Cambia la paleta de colores de toda la app")
    pres = st.sidebar.toggle("Modo presentación", value=False, help="Oculta elementos y aumenta tamaños para exponer en clase")
    if theme == "Purple/Pink":
        st.markdown(
            """
            <style>
            :root{ --brand:#8b5cf6; --brand2:#ec4899; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    elif theme == "Orange/Amber":
        st.markdown(
            """
            <style>
            :root{ --brand:#f59e0b; --brand2:#fb7185; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    if pres:
        st.markdown(
            """
            <style>
              #MainMenu {visibility: hidden;}
              header {visibility: hidden;}
              section[data-testid="stSidebar"] {opacity:.95}
              .stApp { font-size: 1.05rem; }
              .stMarkdown h1, .hero h1 { font-size: 2.6rem !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

def save_experiment(name: str, meta: dict, metrics: dict, model_choice: str, params: dict):
    payload = {
        "name": name,
        "meta": meta,
        "metrics": metrics,
        "model_choice": model_choice,
        "params": params,
    }
    (EXP_DIR / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def list_experiments():
    return sorted([p.stem for p in EXP_DIR.glob("*.json")])

def load_experiment(name: str):
    p = EXP_DIR / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None

def render_shap_summary(model_pipeline, X_sample: pd.DataFrame):
    try:
        model = model_pipeline.named_steps["model"]
        pre = model_pipeline.named_steps["pre"]
        # Transform and densify if sparse
        X_trans = pre.transform(X_sample)
        try:
            import scipy.sparse as sp
            if sp.issparse(X_trans):
                X_trans = X_trans.toarray()
        except Exception:
            pass

        is_tree = hasattr(model, "feature_importances_")
        is_linear = model.__class__.__name__ in ["LogisticRegression", "LinearRegression"]

        plt.close('all')
        if is_tree:
            # Tree-based models: TreeExplainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_trans)
            shap.summary_plot(shap_values if isinstance(shap_values, list) else shap_values, X_trans, show=False)
            st.pyplot(plt.gcf())
        elif is_linear and hasattr(model, "predict_proba"):
            # LogisticRegression: use Kernel on predict_proba for class 1
            f = lambda X: model_pipeline.predict_proba(pd.DataFrame(X, columns=X_sample.columns))[:,1]
            explainer = shap.KernelExplainer(f, X_sample.iloc[:50, :])
            shap_values = explainer.shap_values(X_sample.iloc[:200, :], nsamples=100)
            shap.summary_plot(shap_values, X_sample.iloc[:200, :], show=False)
            st.pyplot(plt.gcf())
        else:
            # Generic fallback: Kernel on predict
            f = lambda X: model_pipeline.predict(pd.DataFrame(X, columns=X_sample.columns))
            explainer = shap.KernelExplainer(f, X_sample.iloc[:50, :])
            shap_values = explainer.shap_values(X_sample.iloc[:200, :], nsamples=100)
            shap.summary_plot(shap_values, X_sample.iloc[:200, :], show=False)
            st.pyplot(plt.gcf())
    except Exception as e:
        st.info(f"SHAP no disponible: {e}")

def generate_pdf_report(title: str, metrics: dict, cm_fig=None, roc_fig=None):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, h-50, title)
    c.setFont("Helvetica", 10)
    y = h-80
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            c.drawString(40, y, f"{k}: {v:.3f}")
        else:
            c.drawString(40, y, f"{k}: {v}")
        y -= 14
    def draw_fig(fig, y_start):
        if fig is None:
            return y_start
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=150)
        img_buf.seek(0)
        img = ImageReader(img_buf)
        c.drawImage(img, 40, y_start-220, width=520, height=200, preserveAspectRatio=True, anchor='sw')
        return y_start-240
    y = draw_fig(cm_fig, y)
    y = draw_fig(roc_fig, y)
    c.showPage()
    c.save()
    buf.seek(0)
    return buf

# --- Analytics helpers ---
def plot_threshold_curves(model_pipeline, X, y):
    try:
        y_prob = model_pipeline.predict_proba(X)[:,1]
    except Exception:
        return None
    thresholds = np.linspace(0.01, 0.99, 50)
    precs, recs, f1s = [], [], []
    for t in thresholds:
        y_hat = (y_prob >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y, y_hat, average="binary", zero_division=0)
        precs.append(p); recs.append(r); f1s.append(f1)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(thresholds, precs, label='Precision')
    ax.plot(thresholds, recs, label='Recall')
    ax.plot(thresholds, f1s, label='F1')
    ax.set_xlabel('Umbral'); ax.set_title('Curvas de umbral')
    ax.legend();
    return fig, thresholds, np.array(precs), np.array(recs), np.array(f1s)

def plot_learning_validation_curves(estimator, X, y, classification=True):
    cv = StratifiedKFold(5, shuffle=True, random_state=42) if classification else KFold(5, shuffle=True, random_state=42)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.2, 1.0, 5))
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Score entrenamiento')
    ax.plot(train_sizes, test_scores.mean(axis=1), 'o-', label='Score validación')
    ax.set_xlabel('Tamaño de entrenamiento')
    ax.set_ylabel('Score')
    ax.set_title('Learning curve (CV)')
    ax.legend()
    return fig

def compute_permutation_importance_fig(model, X, y, classification=True):
    try:
        result = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
        importances = result.importances_mean
        idx = np.argsort(importances)[-15:]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.barh(range(len(idx)), importances[idx])
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([str(c) for c in X.columns[idx]])
        ax.set_title('Permutation Importance (top 15)')
        return fig
    except Exception:
        return None

def plot_pdp_top_features(model_pipeline, X, top_k=3):
    # Best-effort: try to plot PDP for top numeric features
    num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    if not num_cols:
        return None
    feats = num_cols[:top_k]
    try:
        fig, ax = plt.subplots(nrows=1, ncols=len(feats), figsize=(5*len(feats),4))
        if len(feats)==1:
            ax = [ax]
        for i, f in enumerate(feats):
            PartialDependenceDisplay.from_estimator(model_pipeline, X, [f], ax=ax[i])
        fig.suptitle('Partial Dependence (num. features)')
        return fig
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def read_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(io.BytesIO(file_bytes))
    elif ext == ".csv":
        return pd.read_csv(io.BytesIO(file_bytes))
    else:
        raise ValueError("Formato no soportado. Use .csv o .xlsx")

@st.cache_data(show_spinner=False)
def read_table_from_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError("Formato no soportado. Use .csv o .xlsx")

def infer_feature_types(df: pd.DataFrame, target: str):
    features = [c for c in df.columns if c != target]
    cat_cols = [c for c in features if str(df[c].dtype) in ["object", "category"]]
    num_cols = [c for c in features if c not in cat_cols]
    return num_cols, cat_cols

def build_pipeline(problem_type: str, num_cols, cat_cols, model_choice: str = "RandomForest", params: dict | None = None):
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer(transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    params = params or {}
    if problem_type == "Clasificación":
        if model_choice == "RandomForest":
            model = RandomForestClassifier(random_state=42, **params)
        elif model_choice == "GradientBoosting":
            model = GradientBoostingClassifier(random_state=42, **params)
        elif model_choice == "MLP":
            # Ensure defaults for stability
            params = {"max_iter": 300, **params}
            model = MLPClassifier(random_state=42, **params)
        else:  # LogisticRegression
            model = LogisticRegression(max_iter=1000, **params)
    else:
        if model_choice == "RandomForest":
            model = RandomForestRegressor(random_state=42, **params)
        elif model_choice == "GradientBoosting":
            model = GradientBoostingRegressor(random_state=42, **params)
        elif model_choice == "MLP":
            params = {"max_iter": 300, **params}
            model = MLPRegressor(random_state=42, **params)
        else:  # LinearRegression
            model = LinearRegression(**params)
    return Pipeline(steps=[("pre", pre), ("model", model)])


def save_metadata(path: Path, *, problem_type: str, target: str, features: list, num_cols: list, cat_cols: list, class_labels=None, cat_categories=None):
    meta = {
        "problem_type": problem_type,
        "target": target,
        "features": features,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "class_labels": class_labels,
        "cat_categories": cat_categories,
    }
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_metadata(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def train_workflow(df: pd.DataFrame, target: str, problem_type: str, test_size: float = 0.2, random_state: int = 42, model_choice: str = "RandomForest", params: dict | None = None):
    num_cols, cat_cols = infer_feature_types(df, target)
    X = df.drop(columns=[target])
    y = df[target]

    stratify = y if problem_type == "Clasificación" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    pipe = build_pipeline(problem_type, num_cols, cat_cols, model_choice=model_choice, params=params)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    if problem_type == "Clasificación":
        acc = accuracy_score(y_test, y_pred)
        pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
        labels = np.unique(pd.concat([pd.Series(y_test), pd.Series(y_pred)])).tolist()
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        # ROC/PR (solo binario)
        roc_auc = None
        pr_auc = None
        if len(np.unique(y_test)) == 2:
            try:
                y_prob = pipe.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                prec, recl, _ = precision_recall_curve(y_test, y_prob)
                roc_auc = roc_auc_score(y_test, y_prob)
                pr_auc = np.trapz(prec, recl)
            except Exception:
                pass

        ohe = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"] if len(cat_cols) > 0 else None
        cat_categories = {col: (ohe.categories_[i].tolist() if ohe is not None else []) for i, col in enumerate(cat_cols)}

        joblib.dump(pipe, MODEL_PATH)
        save_metadata(
            META_PATH,
            problem_type=problem_type,
            target=target,
            features=X.columns.tolist(),
            num_cols=num_cols,
            cat_cols=cat_cols,
            class_labels=labels,
            cat_categories=cat_categories,
        )
        return pipe, {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1, "labels": labels, "cm": cm, "roc_auc": roc_auc, "pr_auc": pr_auc, "X_test": X_test, "y_test": y_test}
    else:
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        ohe = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"] if len(cat_cols) > 0 else None
        cat_categories = {col: (ohe.categories_[i].tolist() if ohe is not None else []) for i, col in enumerate(cat_cols)}

        joblib.dump(pipe, MODEL_PATH)
        save_metadata(
            META_PATH,
            problem_type=problem_type,
            target=target,
            features=X.columns.tolist(),
            num_cols=num_cols,
            cat_cols=cat_cols,
            class_labels=None,
            cat_categories=cat_categories,
        )
        return pipe, {"r2": r2, "mae": mae, "rmse": rmse, "X_test": X_test, "y_test": y_test, "y_pred": y_pred}


def load_model_and_meta():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    meta = load_metadata(META_PATH)
    return model, meta


def section_train():
    st.header("Entrenar/Actualizar modelo")

    st.markdown("Seleccione el dataset, la columna objetivo y el tipo de problema.")

    # Top toolbar (horizontal) shown after entrenar
    if st.session_state.get("last_metrics") is not None and st.session_state.get("last_problem_type") is not None:
        mt = st.session_state["last_metrics"]
        ptype = st.session_state["last_problem_type"]
        cta1, cta2, cta3, cta4 = st.columns([1,1,1,1])
        with cta1:
            try:
                cm_fig = None
                if ptype == "Clasificación" and "cm" in mt:
                    fig_cm, ax_cm = plt.subplots(figsize=(3,2))
                    sns.heatmap(mt["cm"], annot=False, cmap="Blues", ax=ax_cm, cbar=False)
                    ax_cm.axis('off')
                    cm_fig = fig_cm
                pdf = generate_pdf_report(f"Reporte - {ptype}", {k: v for k, v in mt.items() if isinstance(v, (int,float))}, cm_fig=cm_fig)
                st.download_button("Reporte PDF", data=pdf.getvalue(), file_name="reporte_modelo.pdf", mime="application/pdf")
            except Exception:
                st.button("Reporte PDF", disabled=True, help="Entrene para habilitar")
        with cta2:
            st.download_button("Modelo .joblib", data=open(MODEL_PATH, "rb").read(), file_name="model.joblib", disabled=not MODEL_PATH.exists())
        with cta3:
            st.download_button("Metadatos .json", data=open(META_PATH, "rb").read(), file_name="feature_metadata.json", disabled=not META_PATH.exists())
        with cta4:
            if DICT_PATH.exists():
                st.download_button("Diccionario PDF", data=open(DICT_PATH, "rb").read(), file_name=DICT_PATH.name)
            else:
                st.button("Diccionario PDF", disabled=True)

    # Cargar SIEMPRE por archivo subido
    df = None
    upl = st.file_uploader("Suba .csv o .xlsx", type=["csv", "xlsx"]) 
    if upl is not None:
        try:
            df = read_table(upl.getvalue(), upl.name)
            st.success(f"Archivo cargado: {upl.name}")
        except Exception as e:
            st.error(str(e))

    if df is None:
        st.info("Suba un archivo para continuar con el entrenamiento.")
        return

    st.write("Vista previa")
    st.dataframe(df.head())

    # Data Health quick card
    try:
        n_null = float(df.isna().mean().mean()*100)
        n_dup = int(df.duplicated().sum())
        high_card = [c for c in df.columns if df[c].nunique()>max(100, int(len(df)*0.7))]
        with st.container():
            st.markdown("<div class='block-title'>Salud de datos</div>", unsafe_allow_html=True)
            cdh1, cdh2, cdh3 = st.columns(3)
            cdh1.metric("Nulos promedio", f"{n_null:.1f}%")
            cdh2.metric("Duplicados", f"{n_dup}")
            cdh3.metric("Alta cardinalidad", f"{len(high_card)}")
            if high_card:
                st.caption("Columnas de alta cardinalidad: " + ", ".join(high_card[:5]) + ("…" if len(high_card)>5 else ""))
    except Exception:
        pass

    target = st.selectbox("Columna objetivo (target)", options=[c for c in df.columns])
    # Heurística para columna de paciente (opcional)
    guesses = [c for c in df.columns if any(k in c.lower() for k in ["pac", "patient", "dni", "hist", "id"]) and c != target]
    default_patient = guesses[0] if guesses else None
    patient_col = st.selectbox("Columna paciente (opcional)", options=["(ninguna)"] + [c for c in df.columns if c != target], index=0 if default_patient is None else (["(ninguna)"] + [c for c in df.columns if c != target]).index(default_patient))
    if patient_col == "(ninguna)":
        patient_col = None

    # Alias de visualización (nombres en español)
    default_target_alias = "Diagnóstico" if target.lower() in ["diagnosis", "diagnostic", "dx"] else target
    target_alias = st.text_input("Mostrar nombre de objetivo (alias)", value=default_target_alias, help="Solo cambia cómo se muestra en tablas y encabezados")
    patient_alias = None
    if patient_col:
        default_patient_alias = "Paciente" if any(k in patient_col.lower() for k in ["pac", "patient", "dni", "hist", "id"]) else patient_col
        patient_alias = st.text_input("Mostrar nombre de paciente (alias)", value=default_patient_alias)

    suggested = "Clasificación" if df[target].nunique() <= max(20, int(0.05 * len(df))) or str(df[target].dtype) in ["object", "category"] else "Regresión"
    problem_type = st.radio("Tipo de problema", ["Clasificación", "Regresión"], index=0 if suggested == "Clasificación" else 1, horizontal=True, help="Se infiere por cardinalidad/tipo, puedes cambiarlo")

    # Selección de modelo
    if problem_type == "Clasificación":
        model_choice = st.selectbox("Modelo", ["RandomForest", "GradientBoosting", "LogisticRegression", "MLP"], index=0, help="Selecciona el algoritmo base")
    else:
        model_choice = st.selectbox("Modelo", ["RandomForest", "GradientBoosting", "LinearRegression", "MLP"], index=0, help="Selecciona el algoritmo base")

    test_size = st.slider("Tamaño de test (%)", 10, 40, 20, step=5, help="Porcentaje de datos para evaluación") / 100.0
    st.markdown("---")
    st.subheader("Hiperparámetros del modelo")
    params = {}
    if model_choice == "RandomForest":
        c1, c2, c3, c4 = st.columns(4)
        params["n_estimators"] = c1.slider("n_estimators", 100, 1000, 300, step=50, help="Número de árboles")
        params["max_depth"] = c2.slider("max_depth", 2, 50, 10, step=1, help="Profundidad máxima")
        params["min_samples_split"] = c3.slider("min_samples_split", 2, 10, 2, help="Mínimo para dividir nodo")
        params["min_samples_leaf"] = c4.slider("min_samples_leaf", 1, 10, 1, help="Mínimo por hoja")
        if problem_type == "Clasificación":
            cw = st.selectbox("class_weight", [None, "balanced"], index=0, help="Rebalancea clases desbalanceadas")
            if cw is not None:
                params["class_weight"] = cw
    elif model_choice == "GradientBoosting":
        c1, c2, c3 = st.columns(3)
        params["n_estimators"] = c1.slider("n_estimators", 50, 500, 200, step=50, help="Número de etapas")
        params["learning_rate"] = c2.slider("learning_rate", 0.01, 0.5, 0.1, step=0.01, help="Tasa de aprendizaje")
        params["max_depth"] = c3.slider("max_depth", 2, 10, 3, step=1, help="Profundidad de árboles")
    elif model_choice == "LogisticRegression":
        c1, c2 = st.columns(2)
        params["C"] = c1.slider("C (inversa regularización)", 0.01, 5.0, 1.0, step=0.01, help="Mayor C = menor regularización")
        params["penalty"] = c2.selectbox("penalty", ["l2", "l1"], index=0, help="Tipo de penalización")
        params["solver"] = st.selectbox("solver", ["lbfgs", "liblinear", "saga"], index=0, help="Optimizador")
    elif model_choice == "LinearRegression":
        st.caption("LinearRegression sin hiperparámetros importantes.")
    elif model_choice == "MLP":
        st.caption("Red neuronal MLP (multi-layer perceptron)")
        c1, c2, c3 = st.columns(3)
        hls_text = c1.text_input("hidden_layer_sizes", value="64,32", help="Capas ocultas separadas por coma")
        try:
            hls = tuple(int(x.strip()) for x in hls_text.split(',') if x.strip())
        except Exception:
            hls = (64, 32)
        params["hidden_layer_sizes"] = hls
        params["activation"] = c2.selectbox("activation", ["relu", "tanh", "logistic", "identity"], index=0)
        params["alpha"] = c3.slider("alpha (L2)", 0.0001, 0.1, 0.0001, step=0.0001)
        c4, c5 = st.columns(2)
        params["learning_rate_init"] = c4.slider("learning_rate_init", 0.0001, 0.1, 0.001, step=0.0001)
        params["max_iter"] = int(c5.slider("max_iter", 100, 1000, 300, step=50))

    if st.button("Entrenar modelo", type="primary"):
        with st.spinner("Entrenando modelo..."):
            try:
                model, metrics = train_workflow(df, target, problem_type, test_size=test_size, model_choice=model_choice, params=params)
                st.success("Modelo entrenado y guardado.")
                # Persist last results for top toolbar
                st.session_state["last_metrics"] = metrics
                st.session_state["last_problem_type"] = problem_type
                # Guardar columna de paciente y alias en metadatos
                meta_saved = load_metadata(META_PATH)
                if meta_saved is not None:
                    meta_saved["patient_id_col"] = patient_col
                    meta_saved["target_display_name"] = target_alias
                    if patient_alias:
                        meta_saved["patient_display_name"] = patient_alias
                    META_PATH.write_text(json.dumps(meta_saved, ensure_ascii=False, indent=2), encoding="utf-8")

                if problem_type == "Clasificación":
                    c1, c2 = st.columns([1,1])
                    with c1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                        st.metric("Precision (weighted)", f"{metrics['precision']:.3f}")
                        st.metric("Recall (weighted)", f"{metrics['recall']:.3f}")
                        st.metric("F1 (weighted)", f"{metrics['f1']:.3f}")
                    with c2:
                        fig, ax = plt.subplots(figsize=(5,4))
                        sns.heatmap(metrics["cm"], annot=True, fmt="d", cmap="Blues", xticklabels=metrics["labels"], yticklabels=metrics["labels"], ax=ax)
                        ax.set_xlabel("Predicho")
                        ax.set_ylabel("Real")
                        ax.set_title("Matriz de confusión")
                        st.pyplot(fig)
                    # Tabla de pacientes y objetivo (real/predicha)
                    try:
                        X_test = metrics["X_test"]
                        y_test = metrics["y_test"]
                        y_pred = model.predict(X_test)
                        prob1 = None
                        try:
                            proba = model.predict_proba(X_test)
                            if proba.shape[1] == 2:
                                prob1 = proba[:,1]
                        except Exception:
                            pass
                        # Crea tabla con alias en nombres de columnas para visualización
                        real_col = target_alias
                        pred_col = f"{target_alias} (predicho)"
                        df_res = pd.DataFrame({
                            real_col: y_test,
                            pred_col: y_pred,
                        }, index=X_test.index)
                        if prob1 is not None:
                            df_res["Prob. clase 1"] = prob1
                        # Adjuntar columna paciente desde df original si existe
                        if patient_col and patient_col in df.columns:
                            patient_col_name = patient_alias or patient_col
                            df_res.insert(0, patient_col_name, df.loc[df_res.index, patient_col])
                        st.markdown(f"### Pacientes y {target_alias.lower()}")
                        st.dataframe(df_res.head(200))
                        buf = io.StringIO()
                        df_res.to_csv(buf, index=False)
                        st.download_button(f"Descargar pacientes/{target_alias.lower()} (CSV)", data=buf.getvalue(), file_name="pacientes_objetivo.csv", mime="text/csv")
                    except Exception:
                        pass
                    # Curvas ROC/PR si es binario
                    if metrics.get("roc_auc") is not None:
                        st.markdown("### Curvas")
                        X_test = metrics["X_test"]; y_test = metrics["y_test"]
                        try:
                            y_prob = model.predict_proba(X_test)[:, 1]
                            fpr, tpr, _ = roc_curve(y_test, y_prob)
                            prec, recl, _ = precision_recall_curve(y_test, y_prob)
                            col1, col2 = st.columns(2)
                            with col1:
                                fig1, ax1 = plt.subplots(figsize=(5,4))
                                ax1.plot(fpr, tpr, label=f"ROC AUC = {metrics['roc_auc']:.3f}")
                                ax1.plot([0,1],[0,1],'--', color='gray')
                                ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR'); ax1.set_title('ROC')
                                ax1.legend(); st.pyplot(fig1)
                            with col2:
                                fig2, ax2 = plt.subplots(figsize=(5,4))
                                ax2.plot(recl, prec, label=f"PR AUC = {metrics['pr_auc']:.3f}")
                                ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision'); ax2.set_title('Precision-Recall')
                                ax2.legend(); st.pyplot(fig2)
                        except Exception:
                            pass
                    # Importancia de variables si es RandomForest
                    try:
                        rf = model.named_steps["model"]
                        importances = rf.feature_importances_
                        # Obtener nombres de features procesados no es trivial; mostramos importancia global
                        st.markdown("### Importancia de variables (modelo)")
                        fig, ax = plt.subplots(figsize=(6,4))
                        ax.bar(range(len(importances)), sorted(importances, reverse=True)[:20])
                        ax.set_title("Top importancias (primeras 20)")
                        st.pyplot(fig)
                    except Exception:
                        pass
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R2", f"{metrics['r2']:.3f}")
                    c2.metric("MAE", f"{metrics['mae']:.3f}")
                    c3.metric("RMSE", f"{metrics['rmse']:.3f}")
                    try:
                        fig, ax = plt.subplots(figsize=(5,4))
                        ax.scatter(metrics["y_test"], metrics["y_pred"], alpha=0.6)
                        ax.set_xlabel("Real"); ax.set_ylabel("Predicho"); ax.set_title("Real vs Predicho")
                        st.pyplot(fig)
                    except Exception:
                        pass
                # Reporte PDF y SHAP
                try:
                    cm_fig = None
                    roc_fig = None
                    if problem_type == "Clasificación":
                        # intentar recrear cm_fig a partir de metrics['cm']
                        fig_cm, ax_cm = plt.subplots(figsize=(5,4))
                        sns.heatmap(metrics["cm"], annot=True, fmt="d", cmap="Blues", xticklabels=metrics["labels"], yticklabels=metrics["labels"], ax=ax_cm)
                        ax_cm.set_xlabel("Predicho"); ax_cm.set_ylabel("Real"); ax_cm.set_title("Matriz de confusión")
                        cm_fig = fig_cm
                    pdf = generate_pdf_report(
                        f"Reporte - {problem_type}",
                        {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                        cm_fig=cm_fig,
                        roc_fig=roc_fig,
                    )
                    st.download_button("Descargar reporte (PDF)", data=pdf.getvalue(), file_name="reporte_modelo.pdf", mime="application/pdf")
                except Exception:
                    pass

                # ZIP export de artefactos
                try:
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                        if MODEL_PATH.exists():
                            zf.writestr('model.joblib', open(MODEL_PATH, 'rb').read())
                        if META_PATH.exists():
                            zf.writestr('feature_metadata.json', open(META_PATH, 'rb').read())
                        # metrics.json
                        zf.writestr('metrics.json', json.dumps({k: (float(v) if isinstance(v, (int,float,np.floating)) else str(v)) for k,v in metrics.items()}, ensure_ascii=False, indent=2))
                    zip_buf.seek(0)
                    st.download_button("Descargar paquete (ZIP)", data=zip_buf.getvalue(), file_name="artefactos_modelo.zip", mime="application/zip")
                except Exception:
                    pass

                st.markdown("### Explicabilidad (SHAP)")
                try:
                    render_shap_summary(model, metrics["X_test"].head(200))
                except Exception:
                    st.caption("SHAP no disponible para este modelo o tamaño de muestra.")

                # Descargas de artefactos
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.download_button("Descargar modelo (.joblib)", data=open(MODEL_PATH, "rb").read(), file_name="model.joblib")
                with col_b:
                    st.download_button("Descargar metadatos (.json)", data=open(META_PATH, "rb").read(), file_name="feature_metadata.json")
                with col_c:
                    if DICT_PATH.exists():
                        st.download_button("Diccionario (PDF)", data=open(DICT_PATH, "rb").read(), file_name=DICT_PATH.name)

                # Guardar experimento
                st.markdown("### Guardar experimento")
                default_name = f"exp_{model_choice}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                exp_name = st.text_input("Nombre del experimento", value=default_name)
                if st.button("Guardar experimento", key="save_exp"):
                    meta_saved = load_metadata(META_PATH)
                    basic_metrics = {k: float(v) for k, v in metrics.items() if k in ["accuracy","precision","recall","f1","r2","mae","rmse"] and isinstance(v, (int,float,np.floating))}
                    save_experiment(exp_name, meta_saved, basic_metrics, model_choice, params)
                    st.success(f"Experimento guardado: {exp_name}")
                    # Navegar a Experimentos para que se vea inmediatamente
                    st.session_state["nav"] = "Experimentos"
                    st.experimental_rerun()

                # Optimización con GridSearchCV
                with st.expander("Optimización (GridSearchCV)"):
                    do_cv = st.checkbox("Ejecutar búsqueda de hiperparámetros", value=False)
                    if do_cv:
                        if model_choice == "RandomForest":
                            grid = {"model__n_estimators": [200, 400], "model__max_depth": [None, 10]}
                        elif model_choice == "GradientBoosting":
                            grid = {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]}
                        elif model_choice == "LogisticRegression":
                            grid = {"model__C": [0.5, 1.0, 2.0]}
                        elif model_choice == "MLP":
                            grid = {
                                "model__hidden_layer_sizes": [(64,), (64,32)],
                                "model__alpha": [0.0001, 0.001],
                                "model__learning_rate_init": [0.001, 0.01],
                                "model__max_iter": [300]
                            }
                        elif model_choice == "LinearRegression":
                            grid = {}
                        else:
                            grid = {}
                        st.write(grid)
                        pipe = build_pipeline(problem_type, *infer_feature_types(df, target), model_choice=model_choice, params=params)
                        scoring = 'accuracy' if problem_type == "Clasificación" else 'r2'
                        cv = GridSearchCV(pipe, param_grid=grid, cv=3, n_jobs=-1, scoring=scoring)
                        with st.spinner("Buscando mejores hiperparámetros..."):
                            cv.fit(df.drop(columns=[target]), df[target])
                        st.success(f"Mejor puntaje CV: {cv.best_score_:.3f}")
                        st.json(cv.best_params_)

                # Comparación rápida de modelos
                with st.expander("Comparar modelos rápidamente"):
                    choices = ["RandomForest", "GradientBoosting", "MLP"] + (["LogisticRegression"] if problem_type=="Clasificación" else ["LinearRegression"])
                    sel = st.multiselect("Modelos a evaluar", choices, default=choices)
                    if st.button("Evaluar modelos", key="cmp"):
                        rows = []
                        for m in sel:
                            pipe = build_pipeline(problem_type, *infer_feature_types(df, target), model_choice=m, params={})
                            Xtr, Xte, ytr, yte = train_test_split(
                                df.drop(columns=[target]), df[target], test_size=test_size, random_state=42,
                                stratify=df[target] if problem_type=="Clasificación" and df[target].nunique()>1 else None,
                            )
                            pipe.fit(Xtr, ytr)
                            yp = pipe.predict(Xte)
                            if problem_type=="Clasificación":
                                rows.append({"modelo": m, "accuracy": accuracy_score(yte, yp)})
                            else:
                                rows.append({"modelo": m, "r2": r2_score(yte, yp)})
                        st.dataframe(pd.DataFrame(rows))
            except Exception as e:
                st.error(str(e))


def render_input_widgets(meta: dict):
    inputs = {}
    vitals = []
    labs = []
    demos = []
    others = []
    for col in meta["features"]:
        cl = col.lower()
        if cl in ["edad", "age", "sexo", "genero", "sexo_biologico"]:
            demos.append(col)
        elif any(k in cl for k in ["sistol", "diast", "bp", "presion", "hr", "pulso", "heart_rate", "rr", "resp", "spo2", "sat", "o2", "temp", "temperatura"]):
            vitals.append(col)
        elif any(k in cl for k in ["hb", "hemog", "hto", "hemat", "leuco", "wbc", "neut", "linf", "plaquet", "platelet", "gluc", "glyc", "creat", "urea", "sodio", "na", "potasio", "k_", "pcr", "crp", "ddim", "bilirr", "alt", "ast", "ldh", "ferrit"]):
            labs.append(col)
        else:
            others.append(col)

    def render_num(col):
        cl = col.lower()
        if cl in ["edad", "age"]:
            return st.number_input("Edad", min_value=0, max_value=100, value=0, step=1, key=f"num_{col}")
        if any(k in cl for k in ["sistol", "bp_sys", "systolic"]):
            return st.number_input("Presión sistólica (mmHg)", min_value=60, max_value=240, value=120, step=1, key=f"num_{col}")
        if any(k in cl for k in ["diast", "bp_dia", "diastolic"]):
            return st.number_input("Presión diastólica (mmHg)", min_value=30, max_value=150, value=80, step=1, key=f"num_{col}")
        if any(k in cl for k in ["hr", "pulso", "heart_rate"]):
            return st.number_input("Frecuencia cardiaca (bpm)", min_value=30, max_value=220, value=80, step=1, key=f"num_{col}")
        if any(k in cl for k in ["rr", "resp"]):
            return st.number_input("Frecuencia respiratoria (rpm)", min_value=6, max_value=60, value=16, step=1, key=f"num_{col}")
        if any(k in cl for k in ["spo2", "sat", "o2"]):
            return st.number_input("Saturación O2 (%)", min_value=50, max_value=100, value=98, step=1, key=f"num_{col}")
        if any(k in cl for k in ["temp", "temperatura"]):
            return st.number_input("Temperatura (°C)", min_value=30.0, max_value=43.0, value=36.5, step=0.1, key=f"num_{col}")
        if any(k in cl for k in ["gluc"]):
            return st.number_input("Glucosa (mg/dL)", min_value=20.0, max_value=600.0, value=90.0, step=1.0, key=f"num_{col}")
        if any(k in cl for k in ["hb", "hemog"]):
            return st.number_input("Hemoglobina (g/dL)", min_value=3.0, max_value=22.0, value=13.5, step=0.1, key=f"num_{col}")
        if any(k in cl for k in ["plaquet", "platelet"]):
            return st.number_input("Plaquetas (x10³/µL)", min_value=10.0, max_value=1000.0, value=250.0, step=1.0, key=f"num_{col}")
        if any(k in cl for k in ["creat"]):
            return st.number_input("Creatinina (mg/dL)", min_value=0.2, max_value=15.0, value=1.0, step=0.1, key=f"num_{col}")
        if any(k in cl for k in ["pcr", "crp"]):
            return st.number_input("PCR/CRP (mg/L)", min_value=0.0, max_value=500.0, value=5.0, step=0.5, key=f"num_{col}")
        return st.number_input(col, value=0.0, key=f"num_{col}")

    def render_cat(col):
        options = meta.get("cat_categories", {}).get(col, [])
        default = options[0] if options else ""
        label = "Sexo" if col.lower() in ["sexo", "genero", "sexo_biologico"] else col
        return st.selectbox(label, options=options if options else [default], key=f"cat_{col}")

    with st.expander("Signos vitales", expanded=True):
        for col in [c for c in vitals if c in meta["num_cols"]]:
            inputs[col] = render_num(col)
        for col in [c for c in vitals if c in meta["cat_cols"]]:
            inputs[col] = render_cat(col)

    with st.expander("Laboratorio", expanded=False):
        for col in [c for c in labs if c in meta["num_cols"]]:
            inputs[col] = render_num(col)
        for col in [c for c in labs if c in meta["cat_cols"]]:
            inputs[col] = render_cat(col)

    with st.expander("Demográficos", expanded=False):
        for col in [c for c in demos if c in meta["num_cols"]]:
            inputs[col] = render_num(col)
        for col in [c for c in demos if c in meta["cat_cols"]]:
            inputs[col] = render_cat(col)

    with st.expander("Otros", expanded=False):
        for col in [c for c in others if c in meta["num_cols"]]:
            inputs[col] = render_num(col)
        for col in [c for c in others if c in meta["cat_cols"]]:
            inputs[col] = render_cat(col)
        for col in meta["features"]:
            if col not in inputs:
                inputs[col] = st.text_input(col, value="", key=f"txt_{col}")
    return inputs


def section_predict_single():
    st.header("Predicción individual")
    model, meta = load_model_and_meta()
    if model is None or meta is None:
        st.warning("Primero entrene y guarde un modelo en la pestaña Entrenar.")
        return

    with st.form("single_pred_form"):
        st.markdown("Ingrese los valores de las variables predictoras.")
        inputs = render_input_widgets(meta)
        threshold = None
        if meta.get("problem_type") == "Clasificación":
            threshold = st.slider("Umbral de clasificación (si hay probabilidades)", 0.05, 0.95, 0.50, step=0.05)
        submitted = st.form_submit_button("Predecir")

    if submitted:
        try:
            df_input = pd.DataFrame([inputs], columns=meta["features"])
            pred = model.predict(df_input)[0]
            prob_info = ""
            if threshold is not None:
                try:
                    proba = model.predict_proba(df_input)[0]
                    if proba.shape[0] == 2:
                        pred_bin = int(proba[1] >= threshold)
                        st.write(f"Probabilidad clase 1: {proba[1]:.3f} | Umbral: {threshold:.2f}")
                        st.success(f"Predicción (umbral): {pred_bin}")
                    else:
                        st.success(f"Predicción: {pred}")
                except Exception:
                    st.success(f"Predicción: {pred}")
            else:
                st.success(f"Predicción: {pred}")
        except Exception as e:
            st.error(str(e))


def section_predict_batch():
    st.header("Predicción por lotes")
    model, meta = load_model_and_meta()
    if model is None or meta is None:
        st.warning("Primero entrene y guarde un modelo en la pestaña Entrenar.")
        return

    # Descargar plantilla
    with st.expander("Descargar plantilla de columnas", expanded=False):
        cols_csv = ",".join(meta["features"] + ([meta["target"]] if meta.get("target") else []))
        st.code(cols_csv, language="text")
        st.download_button("Descargar cabeceras (CSV)", data=cols_csv + "\n", file_name="plantilla_columnas.csv", mime="text/csv")

    upl = st.file_uploader("Suba archivo .csv o .xlsx", type=["csv", "xlsx"])
    if upl is None:
        return

    try:
        df = read_table(upl.getvalue(), upl.name)
    except Exception as e:
        st.error(str(e))
        return

    st.write("Vista previa")
    st.dataframe(df.head())

    has_target = meta["target"] in df.columns
    if has_target:
        st.info(f"Se detectó la columna objetivo: {meta['target']}")
    threshold = None
    auto = False
    if meta.get("problem_type") == "Clasificación":
        cth1, cth2 = st.columns([3,1])
        with cth1:
            threshold = st.slider("Umbral de clasificación (si hay probabilidades)", 0.05, 0.95, 0.50, step=0.01, help="Ajusta el compromiso Precision/Recall")
        with cth2:
            auto = st.toggle("Auto", value=False, help="Recalcular automáticamente con el slider")

    def run_batch():
        try:
            X = df.copy()
            y_true = None
            if has_target:
                y_true = X.pop(meta["target"]) 
            missing_cols = [c for c in meta["features"] if c not in X.columns]
            for c in missing_cols:
                X[c] = np.nan
            X = X[meta["features"]]
            y_pred = model.predict(X)
            # Si hay umbral y probabilidades (binario)
            y_prob1 = None
            if threshold is not None:
                try:
                    prob = model.predict_proba(X)
                    if prob.shape[1] == 2:
                        y_prob1 = prob[:,1]
                        y_pred = (y_prob1 >= threshold).astype(int)
                except Exception:
                    pass

            out = X.copy()
            out["prediccion"] = y_pred
            if y_prob1 is not None:
                out["prob_clase_1"] = y_prob1
            # Adjuntar columna paciente si existe en el archivo y en metadatos
            patcol = meta.get("patient_id_col")
            if patcol and patcol in df.columns and patcol not in out.columns:
                out.insert(0, patcol, df.loc[out.index, patcol])

            if meta["problem_type"] == "Clasificación" and y_true is not None:
                labels = meta.get("class_labels") or np.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)])).tolist()
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
                acc = accuracy_score(y_true, y_pred)
                c1, c2 = st.columns([1,1])
                with c1:
                    st.metric("Accuracy", f"{acc:.3f}")
                    st.metric("Precision (weighted)", f"{pr:.3f}")
                    st.metric("Recall (weighted)", f"{rc:.3f}")
                    st.metric("F1 (weighted)", f"{f1:.3f}")
                with c2:
                    fig, ax = plt.subplots(figsize=(5,4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
                    ax.set_xlabel("Predicho")
                    ax.set_ylabel("Real")
                    ax.set_title("Matriz de confusión (lote)")
                    st.pyplot(fig)

            csv_buf = io.StringIO()
            out.to_csv(csv_buf, index=False)
            st.download_button("Descargar predicciones (CSV)", data=csv_buf.getvalue(), file_name="predicciones.csv", mime="text/csv")

            st.success("Predicción por lotes finalizada.")
        except Exception as e:
            st.error(str(e))

    # Validación clínica opcional y clipping
    def get_limits(col: str):
        cl = col.lower()
        # Demográficos / Vitals
        if cl in ["edad", "age"]: return (0, 100, "Edad (años)")
        if any(k in cl for k in ["sistol", "bp_sys", "systolic"]): return (60, 240, "Presión sistólica (mmHg)")
        if any(k in cl for k in ["diast", "bp_dia", "diastolic"]): return (30, 150, "Presión diastólica (mmHg)")
        if any(k in cl for k in ["hr", "pulso", "heart_rate"]): return (30, 220, "Frecuencia cardiaca (bpm)")
        if any(k in cl for k in ["rr", "resp"]): return (6, 60, "Frecuencia respiratoria (rpm)")
        if any(k in cl for k in ["spo2", "sat", "o2"]): return (50, 100, "Saturación O2 (%)")
        if any(k in cl for k in ["temp", "temperatura"]): return (30.0, 43.0, "Temperatura (°C)")
        # Laboratorio
        if "gluc" in cl: return (20.0, 600.0, "Glucosa (mg/dL)")
        if "hb" in cl or "hemog" in cl: return (3.0, 22.0, "Hemoglobina (g/dL)")
        if "plaquet" in cl or "platelet" in cl: return (10.0, 1000.0, "Plaquetas (x10³/µL)")
        if "creat" in cl: return (0.2, 15.0, "Creatinina (mg/dL)")
        if "pcr" in cl or "crp" in cl: return (0.0, 500.0, "PCR/CRP (mg/L)")
        return None

    clip = st.checkbox("Corregir valores fuera de rango (clip)", value=True, help="Ajusta automáticamente a los límites razonables clínicos")

    # Reporte de outliers por columna
    try:
        reports = []
        for c in [c for c in df.columns if c in meta["features"]]:
            lim = get_limits(c)
            if lim is None:
                continue
            lo, hi, label = lim
            if pd.api.types.is_numeric_dtype(df[c]):
                below = int((df[c] < lo).sum())
                above = int((df[c] > hi).sum())
                if below > 0 or above > 0:
                    reports.append({"col": c, "label": label, "<min": below, ">max": above})
                    if clip:
                        df[c] = df[c].clip(lower=lo, upper=hi)
        if reports:
            st.warning("Se detectaron valores fuera de rango clínico. Se muestra el resumen y se aplicó clip si estaba activado.")
            st.dataframe(pd.DataFrame(reports))
    except Exception:
        pass

    if auto:
        run_batch()
    else:
        if st.button("Ejecutar predicción por lotes", type="primary"):
            run_batch()


def main():
    st.title("Oráculo DEMALE – IA de Predicción Clínica")
    # Logo fijo cerca del botón Deploy (arriba derecha)
    inject_top_right_logo(LOGO_PATH, width_px=28, right_px=16, top_px=8)

    if "nav" not in st.session_state:
        st.session_state["nav"] = "Inicio"
    # Theme picker
    render_theme_controls()
    tab = st.sidebar.radio("Navegación", ["Inicio", "EDA", "Entrenar", "Predicción individual", "Predicción por lotes", "Experimentos"], key="nav") 

    if tab == "Inicio":
        # Hero
        st.markdown("""
        <div class="hero glow">
          <span class="pill">DEMALE • HSJM 2025</span>
          <h1>Predicción Inteligente para tu proyecto</h1>
          <p class="sub">Entrena, evalúa y explica tus modelos con una interfaz moderna.</p>
        </div>
        """, unsafe_allow_html=True)

        # Onboarding asistente
        st.markdown("**Cómo empezar**")
        step = st.segmented_control("Paso", options=["Subir datos","EDA","Entrenar","Predecir"], default="Subir datos") if hasattr(st, 'segmented_control') else st.radio("Paso", ["Subir datos","EDA","Entrenar","Predecir"], horizontal=True)
        if step == "Subir datos":
            st.caption("En Entrenar, elige 'Subir archivo' y carga tu CSV/XLSX.")
        elif step == "EDA":
            st.caption("Explora nulos, correlaciones y categorías en la pestaña EDA.")
        elif step == "Entrenar":
            st.caption("Selecciona target, modelo e hiperparámetros y presiona Entrenar.")
        else:
            st.caption("Usa Predicción individual o por lotes para obtener resultados.")

        # Feature cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="card">
              <div class="block-title">⚙️ Modelos y tuning</div>
              <div class="small">RandomForest, GradientBoosting, Logistic/Linear con hiperparámetros y GridSearchCV.</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="card">
              <div class="block-title">📊 Métricas y reportes</div>
              <div class="small">ROC/PR, matriz de confusión, batch metrics y reporte PDF descargable.</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="card">
              <div class="block-title">🧠 Explicabilidad</div>
              <div class="small">SHAP para interpretar tus predicciones y feature importance.</div>
            </div>
            """, unsafe_allow_html=True)

        # CTA button
        if st.button("Comenzar a entrenar", type="primary"):
            st.session_state["nav"] = "Entrenar"
            st.experimental_rerun()

        st.markdown("<div class='footer'>Proyecto: DEMALE HSJM 2025</div>", unsafe_allow_html=True)
    elif tab == "EDA":
        # EDA requiere archivo subido
        st.header("Exploración de datos (EDA)")
        df_eda = None
        up = st.file_uploader("Suba .csv o .xlsx", type=["csv","xlsx"]) 
        if up:
            try:
                df_eda = read_table(up.getvalue(), up.name)
                st.success(f"Archivo cargado: {up.name}")
            except Exception as e:
                st.error(str(e))
        if df_eda is None:
            st.info("Cargue un archivo para analizar.")
        else:
            st.subheader("Vista previa y resumen")
            st.dataframe(df_eda.head())
            c1, c2, c3 = st.columns(3)
            c1.metric("Filas", f"{len(df_eda)}")
            c2.metric("Columnas", f"{df_eda.shape[1]}")
            c3.metric("Nulos (%)", f"{df_eda.isna().mean().mean()*100:.1f}")
            st.markdown("### Nulos por columna")
            st.bar_chart(df_eda.isna().mean().sort_values(ascending=False))
            st.markdown("### Correlación (numéricas)")
            num = df_eda.select_dtypes(include=[np.number])
            if not num.empty:
                fig, ax = plt.subplots(figsize=(6,4))
                sns.heatmap(num.corr(numeric_only=True), cmap="viridis", ax=ax)
                st.pyplot(fig)
            st.markdown("### Distribución de variables categóricas (top 5)")
            cat_cols = df_eda.select_dtypes(include=["object","category"]).columns[:5]
            for col in cat_cols:
                st.write(f"- {col}")
                st.bar_chart(df_eda[col].value_counts().head(10))
    elif tab == "Entrenar":
        section_train()
    elif tab == "Predicción individual":
        section_predict_single()
    elif tab == "Predicción por lotes":
        section_predict_batch()
    elif tab == "Experimentos":
        st.header("Experimentos guardados")
        ctop1, ctop2 = st.columns([1,1])
        with ctop1:
            if st.button("Refrescar lista"):
                pass
        items = list_experiments()
        if not items:
            st.info("No hay experimentos aún. Entrena y guarda uno en la pestaña Entrenar.")
        else:
            name = st.selectbox("Selecciona un experimento", items)
            data = load_experiment(name)
            if data:
                st.json(data["metrics"])
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("Descargar experimento (.json)", data=json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'), file_name=f"{name}.json")
                with col2:
                    if st.button("Ir a Entrenar con este modelo"):
                        st.session_state["nav"] = "Entrenar"
                        st.experimental_rerun()
    elif tab == "Demo":
        st.info("La sección Demo ha sido deshabilitada. Utiliza Entrenar o EDA subiendo tu propio archivo.")
    else:  # Demo
        st.header("Demo rápida (datos sintéticos)")
        mode = st.radio("Tipo", ["Clasificación", "Regresión"], horizontal=True)
        n_samples = st.slider("Muestras", 200, 3000, 800, step=100)
        n_features = st.slider("Features", 5, 30, 12)
        rng = np.random.default_rng(42)
        if mode == "Clasificación":
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=int(n_features*0.6), n_redundant=int(n_features*0.2), random_state=42)
            df_demo = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
            df_demo["target"] = y
        else:
            from sklearn.datasets import make_regression
            X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=int(n_features*0.6), noise=12.0, random_state=42)
            df_demo = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
            df_demo["target"] = y
        st.dataframe(df_demo.head())
        if st.button("Entrenar con demo", type="primary"):
            with st.spinner("Entrenando demo..."):
                model, metrics = train_workflow(df_demo, "target", mode, test_size=0.2, model_choice="RandomForest", params={"n_estimators":200})
                st.success("Demo entrenada")
                st.write(metrics)

if __name__ == "__main__":
    main()
