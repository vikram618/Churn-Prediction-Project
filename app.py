import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
import joblib
import os

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        border-left: 5px solid #e94560;
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; font-weight: 700; }
    .main-header p  { margin: 0.5rem 0 0; opacity: 0.75; font-size: 1rem; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #e8ecf0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        text-align: center;
    }
    .metric-card .label { font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #0f3460; }

    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f3460;
        border-bottom: 3px solid #e94560;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    .prediction-box-churn {
        background: linear-gradient(135deg, #ff6b6b, #e94560);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .prediction-box-no-churn {
        background: linear-gradient(135deg, #56ab2f, #a8e063);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }

    .sidebar-note {
        background: #f0f4ff;
        border-left: 4px solid #0f3460;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #374151;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #f3f4f6;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #0f3460 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📡 Customer Churn Prediction</h1>
    <p>Telecom churn analysis · 6 ML models · Real-time predictions</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown('<div class="sidebar-note">Upload your dataset (Excel) to train the models or use a pre-trained model.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload Dataset (.xlsx)", type=["xlsx"])

    st.markdown("---")
    st.markdown("### 🔬 Model Selection")
    selected_model_name = st.selectbox("Choose model for prediction:", [
        "Random Forest (Tuned)",
        "Logistic Regression",
        "Decision Tree",
        "Gradient Boosting",
        "K-Nearest Neighbors",
        "Support Vector Machine"
    ])

    st.markdown("---")
    st.markdown("### 📖 About")
    st.info("Built with Streamlit · Scikit-learn · Project: Customer Churn Prediction")

# ─────────────────────────────────────────────
#  DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(file):
    if file is not None:
        data = pd.read_excel(file)
    else:
        st.warning("⚠️ No file uploaded. Please upload your P670-dataset.xlsx to continue.")
        return None, None, None, None, None, None

    data = data.drop(data.columns[0], axis=1)

    # Convert numeric-like object columns
    obj_cols = data.select_dtypes(include='object').columns
    for col in obj_cols:
        converted = pd.to_numeric(data[col], errors='coerce')
        if converted.notna().mean() > 0.8:
            data[col] = converted

    # Fill missing values
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns
    data[num_cols] = data[num_cols].fillna(data[num_cols].median())

    # Clip outliers (IQR)
    for col in num_cols:
        Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
        IQR = Q3 - Q1
        data[col] = data[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # Encode categorical
    cat_cols = data.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        data[col] = le.fit_transform(data[col].astype(str))

    X = data.drop('churn', axis=1)
    y = data['churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    return data, X, X_scaled, X_train, X_test, y_train, y_test, scaler

@st.cache_resource
def train_all_models(X_train, X_test, y_train, y_test, X_scaled, y):
    models = {
        'Logistic Regression':    LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree':          DecisionTreeClassifier(random_state=42),
        'Random Forest':          RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting':      GradientBoostingClassifier(random_state=42),
        'K-Nearest Neighbors':    KNeighborsClassifier(),
        'Support Vector Machine': SVC(probability=True, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            'model':     model,
            'y_pred':    y_pred,
            'y_prob':    y_prob,
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall':    recall_score(y_test, y_pred, zero_division=0),
            'f1':        f1_score(y_test, y_pred, zero_division=0),
            'roc_auc':   roc_auc_score(y_test, y_prob)
        }

    # Tune best model (Random Forest)
    param_grid = {
        'n_estimators':      [100, 200],
        'max_depth':         [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf':  [1, 2]
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=5, scoring='f1', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred_t = best_model.predict(X_test)
    y_prob_t = best_model.predict_proba(X_test)[:, 1]
    results['Random Forest (Tuned)'] = {
        'model':      best_model,
        'y_pred':     y_pred_t,
        'y_prob':     y_prob_t,
        'accuracy':   accuracy_score(y_test, y_pred_t),
        'precision':  precision_score(y_test, y_pred_t, zero_division=0),
        'recall':     recall_score(y_test, y_pred_t, zero_division=0),
        'f1':         f1_score(y_test, y_pred_t, zero_division=0),
        'roc_auc':    roc_auc_score(y_test, y_prob_t),
        'best_params': grid_search.best_params_
    }
    return results

# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
result = load_and_preprocess(uploaded_file)

if result[0] is None:
    st.markdown("""
    ### 👋 Welcome! Let's get started.
    **Steps:**
    1. Upload your `P670-dataset.xlsx` in the sidebar
    2. The app will automatically train all 6 models
    3. Explore results in the tabs below
    4. Use the Predict tab to test individual customers
    """)
    st.stop()

data, X, X_scaled, X_train, X_test, y_train, y_test, scaler = result

# Train models
with st.spinner("🔧 Training models... this may take a minute on first run."):
    results = train_all_models(X_train, X_test, y_train, y_test, X_scaled, data['churn'])

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📈 Model Comparison",
    "🔍 Detailed Evaluation",
    "⭐ Feature Importance",
    "🎯 Predict"
])

# ══════════════════════════════════════════════
#  TAB 1 · OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="label">Total Records</div><div class="value">{len(data):,}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="label">Features</div><div class="value">{X.shape[1]}</div></div>', unsafe_allow_html=True)
    with col3:
        churn_rate = data['churn'].mean() * 100
        st.markdown(f'<div class="metric-card"><div class="label">Churn Rate</div><div class="value">{churn_rate:.1f}%</div></div>', unsafe_allow_html=True)
    with col4:
        best_roc = max(v['roc_auc'] for v in results.values())
        st.markdown(f'<div class="metric-card"><div class="label">Best ROC-AUC</div><div class="value">{best_roc:.3f}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Churn Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = data['churn'].value_counts()
        bars = ax.bar(['No Churn', 'Churn'], counts.values,
                      color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    str(val), ha='center', fontweight='bold', fontsize=12)
        ax.set_ylabel('Count')
        ax.set_title('Churn Distribution', fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown('<div class="section-header">Dataset Sample</div>', unsafe_allow_html=True)
        st.dataframe(data.head(10), use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 2 · MODEL COMPARISON
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']

    summary = pd.DataFrame({
        name: {
            'Accuracy':  round(v['accuracy'], 4),
            'Precision': round(v['precision'], 4),
            'Recall':    round(v['recall'], 4),
            'F1 Score':  round(v['f1'], 4),
            'ROC-AUC':   round(v['roc_auc'], 4)
        }
        for name, v in results.items()
    }).T.sort_values('ROC-AUC', ascending=False)

    # Highlight best
    def highlight_best(s):
        is_best = s == s.max()
        return ['background-color: #d4edda; font-weight: bold' if v else '' for v in is_best]

    st.dataframe(
        summary.style.apply(highlight_best, axis=0).format("{:.4f}"),
        use_container_width=True
    )

    st.markdown("---")

    # Bar charts
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    model_names = list(results.keys())
    for ax, metric, label in zip(axes, metrics, metric_labels):
        vals = [results[m][metric] for m in model_names]
        bars = ax.bar(range(len(model_names)), vals, color=colors[:len(model_names)],
                      edgecolor='black', alpha=0.85)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([m.replace(' ', '\n') for m in model_names], fontsize=6)
        ax.set_ylim(0, 1.1)
        ax.set_title(label, fontweight='bold')
        ax.set_ylabel('Score')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=6, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ROC Curves
    st.markdown('<div class="section-header">ROC Curves</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    for (name, v), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, v['y_prob'])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={v['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════
#  TAB 3 · DETAILED EVALUATION
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Confusion Matrices — All Models', fontsize=16, fontweight='bold')
    for ax, (name, v) in zip(axes.flatten(), results.items()):
        cm = confusion_matrix(y_test, v['y_pred'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(name, fontweight='bold', fontsize=10)
    for ax in axes.flatten()[len(results):]:
        ax.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown('<div class="section-header">Classification Reports</div>', unsafe_allow_html=True)
    selected_for_report = st.selectbox("Select model:", list(results.keys()))
    v = results[selected_for_report]
    report = classification_report(y_test, v['y_pred'], target_names=['No Churn', 'Churn'], output_dict=True)
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 4 · FEATURE IMPORTANCE
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Feature Importance — Tuned Random Forest</div>', unsafe_allow_html=True)

    tuned_model = results['Random Forest (Tuned)']['model']
    importances = pd.Series(tuned_model.feature_importances_, index=X.columns)
    importances_sorted = importances.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors_fi = ['#e74c3c' if i < 5 else '#3498db' for i in range(len(importances_sorted))]
    importances_sorted.plot(kind='bar', color=colors_fi, edgecolor='black', ax=ax)
    ax.set_title('Feature Importance — Tuned Random Forest', fontsize=14, fontweight='bold')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance Score')
    ax.axhline(y=importances_sorted.mean(), color='black', linestyle='--', label='Mean Importance')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔴 Top 5 Features (most impactful)**")
        top5 = importances_sorted.head(5).reset_index()
        top5.columns = ['Feature', 'Importance']
        st.dataframe(top5.style.format({'Importance': '{:.4f}'}), use_container_width=True)
    with col2:
        st.markdown("**🔵 Bottom 5 Features (least impactful)**")
        bot5 = importances_sorted.tail(5).reset_index()
        bot5.columns = ['Feature', 'Importance']
        st.dataframe(bot5.style.format({'Importance': '{:.4f}'}), use_container_width=True)

    if 'best_params' in results['Random Forest (Tuned)']:
        st.markdown("---")
        st.markdown("**⚙️ Best Hyperparameters (GridSearchCV)**")
        params_df = pd.DataFrame([results['Random Forest (Tuned)']['best_params']])
        st.dataframe(params_df, use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 5 · PREDICT
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">🎯 Predict Customer Churn</div>', unsafe_allow_html=True)
    st.markdown(f"**Using model:** `{selected_model_name}`")
    st.markdown("Fill in the customer details below and click **Predict**.")

    feature_names = X.columns.tolist()
    st.markdown("---")

    # Create two-column input form
    input_data = {}
    cols = st.columns(3)
    for i, feat in enumerate(feature_names):
        col = cols[i % 3]
        min_val = float(X[feat].min())
        max_val = float(X[feat].max())
        mean_val = float(X[feat].mean())
        step = 1.0 if X[feat].dtype in ['int64'] else 0.01
        input_data[feat] = col.number_input(
            label=feat,
            min_value=min_val,
            max_value=max_val,
            value=round(mean_val, 2),
            step=step,
            key=feat
        )

    st.markdown("---")
    if st.button("🔮 Predict Churn", use_container_width=True):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)

        # Resolve model key
        model_key_map = {
            "Random Forest (Tuned)":   "Random Forest (Tuned)",
            "Logistic Regression":     "Logistic Regression",
            "Decision Tree":           "Decision Tree",
            "Gradient Boosting":       "Gradient Boosting",
            "K-Nearest Neighbors":     "K-Nearest Neighbors",
            "Support Vector Machine":  "Support Vector Machine"
        }
        model_key = model_key_map[selected_model_name]
        model = results[model_key]['model']

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        churn_prob = probability[1] * 100
        no_churn_prob = probability[0] * 100

        col_pred, col_prob = st.columns([1, 1])

        with col_pred:
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box-churn">
                    ⚠️ HIGH CHURN RISK<br>
                    <span style="font-size:1rem;opacity:0.9">This customer is likely to churn</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box-no-churn">
                    ✅ LOW CHURN RISK<br>
                    <span style="font-size:1rem;opacity:0.9">This customer is likely to stay</span>
                </div>
                """, unsafe_allow_html=True)

        with col_prob:
            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.barh(['No Churn', 'Churn'], [no_churn_prob, churn_prob],
                           color=['#2ecc71', '#e74c3c'], edgecolor='black')
            ax.set_xlim(0, 100)
            ax.set_xlabel('Probability (%)')
            ax.set_title('Prediction Probabilities', fontweight='bold')
            for bar, val in zip(bars, [no_churn_prob, churn_prob]):
                ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                        f'{val:.1f}%', va='center', fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            plt.close()

        # Metrics for selected model
        st.markdown("---")
        st.markdown(f"**Model Performance on Test Set ({model_key}):**")
        v = results[model_key]
        m1, m2, m3, m4, m5 = st.columns(5)
        for col, label, val in zip(
            [m1, m2, m3, m4, m5],
            ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
            [v['accuracy'], v['precision'], v['recall'], v['f1'], v['roc_auc']]
        ):
            col.markdown(f'<div class="metric-card"><div class="label">{label}</div><div class="value">{val:.3f}</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#9ca3af;font-size:0.85rem'>Customer Churn Prediction · Built with Streamlit & Scikit-learn</center>",
    unsafe_allow_html=True
)
