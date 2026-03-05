"""
Support Ticket Classifier — Streamlit App
Requires: streamlit, pandas, scikit-learn, plotly
Install : pip install streamlit pandas scikit-learn plotly
Run     : streamlit run app.py
"""

from typing import Optional
import streamlit as st
import pandas as pd

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ── MUST be the very first Streamlit call ─────────────────────────────────────
st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="wide",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f0f4ff;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
    border: 1px solid #d0d9f5;
    margin-bottom: 10px;
}
.metric-card h2 { margin: 0; font-size: 2rem; }
.metric-card p  { margin: 4px 0 0; color: #555; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "Billing":         "#e74c3c",
    "Technical":       "#3498db",
    "Account":         "#2ecc71",
    "General Inquiry": "#f39c12",
    "Shipping":        "#9b59b6",
    "Returns":         "#1abc9c",
    "Other":           "#95a5a6",
}
PRIORITY_COLORS = {
    "High":   "#e74c3c",
    "Medium": "#f39c12",
    "Low":    "#2ecc71",
}

KEYWORD_RULES = {
    "Billing": [
        "bill", "charge", "invoice", "payment", "refund", "price",
        "cost", "fee", "subscription", "overcharged",
    ],
    "Technical": [
        "error", "bug", "crash", "broken", "not working", "issue",
        "problem", "fail", "slow", "login", "password", "reset",
        "cannot", "can't", "unable", "glitch", "down",
    ],
    "Account": [
        "account", "profile", "username", "email", "settings",
        "delete", "deactivate", "access", "permission", "locked",
    ],
    "Shipping": [
        "ship", "delivery", "deliver", "package", "tracking",
        "track", "order", "arrived", "delay", "lost", "courier",
    ],
    "Returns": [
        "return", "refund", "exchange", "replace", "warranty",
        "broken item", "damaged", "wrong item",
    ],
    "General Inquiry": [
        "question", "info", "information", "how", "what", "when",
        "where", "help", "support", "enquiry", "inquiry",
    ],
}

SUGGESTED_REPLIES = {
    "Billing": (
        "Hi! I can see you have a billing concern. "
        "I'll look into your account charges right away and follow up within 24 hours."
    ),
    "Technical": (
        "Thanks for reaching out! Our technical team has been notified. "
        "Could you share any error messages or screenshots you're seeing?"
    ),
    "Account": (
        "I'll help you with your account. "
        "For security, please verify your identity and we'll resolve this promptly."
    ),
    "Shipping": (
        "I'll check the status of your shipment right away. "
        "Could you provide your order number so I can track it?"
    ),
    "Returns": (
        "I'm sorry to hear about the issue with your item. "
        "I'll initiate the return/exchange process for you straight away."
    ),
    "General Inquiry": (
        "Great question! I'm happy to help. "
        "Let me find the right information for you."
    ),
}

# ── Helper functions ──────────────────────────────────────────────────────────
def keyword_classify(text: str) -> str:
    """Keyword-based fallback classifier."""
    t = str(text).lower()
    scores = {cat: sum(1 for kw in kws if kw in t) for cat, kws in KEYWORD_RULES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"


def estimate_priority(text: str) -> str:
    """Heuristic priority estimator."""
    t = str(text).lower()
    high_kws = [
        "urgent", "asap", "immediately", "critical", "emergency",
        "fraud", "not working", "cannot access", "broken", "severe",
    ]
    low_kws = [
        "curious", "wondering", "no rush", "when you get a chance",
        "just asking", "whenever",
    ]
    if any(k in t for k in high_kws):
        return "High"
    if any(k in t for k in low_kws):
        return "Low"
    return "Medium"


def find_text_column(df: pd.DataFrame) -> Optional[str]:
    """Guess the most likely ticket-text column."""
    hints = ["subject", "text", "description", "body", "message", "ticket", "content", "detail"]
    for hint in hints:
        for col in df.columns:
            if hint in col.lower() and df[col].dtype == object:
                return col
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if obj_cols:
        return max(obj_cols, key=lambda c: df[c].dropna().str.len().mean())
    return None


def find_label_column(df: pd.DataFrame) -> Optional[str]:
    """Guess the most likely label/category column."""
    hints = ["category", "label", "type", "class", "tag", "topic", "issue_type"]
    for hint in hints:
        for col in df.columns:
            if hint in col.lower():
                return col
    return None


def metric_card(value, label: str, color: str = "#1a3c8f") -> None:
    """Render a styled metric card using HTML."""
    st.markdown(
        f'<div class="metric-card">'
        f'<h2 style="color:{color}">{value}</h2>'
        f'<p>{label}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def safe_value_counts(series: pd.Series, name_col: str, count_col: str) -> pd.DataFrame:
    """pandas-version-safe value_counts that works for pandas 1.x and 2.x."""
    return (
        series
        .value_counts()
        .rename_axis(name_col)
        .reset_index(name=count_col)
    )

# ── Session-state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "df":                None,
    "model":             None,
    "model_trained":     False,
    "_uploaded_file_id": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎫 Ticket Classifier")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "📊 Dashboard",
            "📁 Upload & Classify",
            "🤖 Train Model",
            "✏️ Classify Single Ticket",
        ],
    )
    st.markdown("---")

    if st.session_state["df"] is not None:
        st.success(f"Dataset: {len(st.session_state['df']):,} rows ✅")
    else:
        st.caption("Upload your CSV to get started.")

    if st.session_state["model_trained"]:
        st.success("ML model ready ✅")
    else:
        st.info("Using keyword classifier\n(train a model for better accuracy)")

    if st.button("🗑️ Clear All Data"):
        for k, v in _DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Support Ticket Dashboard")

    if st.session_state["df"] is None:
        st.info("No data loaded yet. Go to **📁 Upload & Classify** to load your CSV.")
        st.stop()

    df = st.session_state["df"]

    cat_col  = next((c for c in ["predicted_category", "category", "Category", "issue_type"] if c in df.columns), None)
    pri_col  = next((c for c in ["predicted_priority", "priority", "Priority"] if c in df.columns), None)
    text_col = find_text_column(df)

    if cat_col is None:
        st.warning(
            "Tickets haven't been classified yet. "
            "Go to **📁 Upload & Classify** and click **Classify All Tickets** first."
        )

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(f"{len(df):,}", "Total Tickets")
    with c2:
        metric_card(df[cat_col].nunique() if cat_col else "–", "Categories")
    with c3:
        high_n = f"{int((df[pri_col] == 'High').sum()):,}" if pri_col else "–"
        metric_card(high_n, "High Priority", color="#e74c3c")
    with c4:
        if text_col and df[text_col].dropna().shape[0] > 0:
            avg_len = int(df[text_col].dropna().str.len().mean())
            metric_card(avg_len, "Avg Text Length")
        else:
            metric_card("–", "Avg Text Length")

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    # Category pie chart
    if cat_col:
        cat_counts = safe_value_counts(df[cat_col].dropna(), "Category", "Count")
        with col_l:
            st.subheader("Ticket Categories")
            if PLOTLY_AVAILABLE:
                fig = px.pie(
                    cat_counts, values="Count", names="Category", hole=0.4,
                    color="Category", color_discrete_map=CATEGORY_COLORS,
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(showlegend=True, margin=dict(t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(cat_counts.set_index("Category"))
    else:
        with col_l:
            st.subheader("Ticket Categories")
            st.info("Classify tickets first to see category distribution.")

    # Priority bar chart
    if pri_col:
        pri_counts = safe_value_counts(df[pri_col].dropna(), "Priority", "Count")
        order = ["High", "Medium", "Low"]
        pri_counts["Priority"] = pd.Categorical(
            pri_counts["Priority"], categories=order, ordered=True
        )
        pri_counts = pri_counts.sort_values("Priority").dropna(subset=["Priority"])
        with col_r:
            st.subheader("Priority Breakdown")
            if PLOTLY_AVAILABLE:
                fig = px.bar(
                    pri_counts, x="Priority", y="Count", text="Count",
                    color="Priority", color_discrete_map=PRIORITY_COLORS,
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False, margin=dict(t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(pri_counts.set_index("Priority"))
    else:
        with col_r:
            st.subheader("Priority Breakdown")
            st.info("Classify tickets first to see priority distribution.")

    # Heatmap
    if cat_col and pri_col and PLOTLY_AVAILABLE:
        st.subheader("Category × Priority Heatmap")
        hm = df.groupby([cat_col, pri_col]).size().unstack(fill_value=0)
        ordered_pri = [c for c in ["High", "Medium", "Low"] if c in hm.columns]
        if ordered_pri:
            hm = hm[ordered_pri]
        fig = px.imshow(
            hm, text_auto=True, aspect="auto",
            color_continuous_scale="Blues",
            labels=dict(x="Priority", y="Category", color="Count"),
        )
        fig.update_layout(margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 View Raw Data"):
        st.dataframe(df, use_container_width=True)

    if cat_col:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Classified CSV",
            data=csv_bytes,
            file_name="classified_tickets.csv",
            mime="text/csv",
        )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – Upload & Classify
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Upload & Classify":
    st.title("📁 Upload & Classify Tickets")

    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

    # Parse only when a NEW file is uploaded (avoid re-parsing on every rerun)
    if uploaded is not None:
        file_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state["_uploaded_file_id"] != file_id:
            try:
                df_new = pd.read_csv(uploaded)
                st.session_state["df"] = df_new
                st.session_state["_uploaded_file_id"] = file_id
                st.toast(f"Loaded {len(df_new):,} rows!", icon="✅")
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

    # Render from session state so data persists across page switches
    if st.session_state["df"] is not None:
        df = st.session_state["df"]

        col_info1, col_info2 = st.columns(2)
        col_info1.metric("Rows", f"{len(df):,}")
        col_info2.metric("Columns", len(df.columns))

        with st.expander("👁️ Preview data (first 10 rows)"):
            st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Column Mapping")
        obj_cols = [c for c in df.columns if df[c].dtype == object]

        if not obj_cols:
            st.error("No text (string) columns found in this CSV.")
        else:
            guessed_text = find_text_column(df)
            default_text_idx = obj_cols.index(guessed_text) if guessed_text in obj_cols else 0

            text_col = st.selectbox(
                "Which column contains the ticket text?",
                obj_cols,
                index=default_text_idx,
                key="upload_text_col",
            )

            st.markdown("---")
            if st.button("🚀 Classify All Tickets", type="primary"):
                with st.spinner("Classifying tickets…"):
                    series = df[text_col].fillna("").astype(str)
                    if st.session_state["model_trained"] and SKLEARN_AVAILABLE:
                        df["predicted_category"] = st.session_state["model"].predict(series)
                        method_used = "ML Model"
                    else:
                        df["predicted_category"] = series.apply(keyword_classify)
                        method_used = "Keyword Rules"
                    df["predicted_priority"] = series.apply(estimate_priority)

                st.session_state["df"] = df
                st.success(
                    f"Classification complete using **{method_used}**! "
                    "Head to **📊 Dashboard** to explore results."
                )
                st.dataframe(df.head(20), use_container_width=True)

                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Classified CSV",
                    data=csv_bytes,
                    file_name="classified_tickets.csv",
                    mime="text/csv",
                )
    else:
        st.info("⬆️ Upload a CSV file above to get started.")
        st.markdown(
            "**Expected format:** Any CSV with at least one text column. "
            "Optional: a `category` / `label` column for model training."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Train Model
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Train Model":
    st.title("🤖 Train Classification Model")

    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn is not installed. Run: `pip install scikit-learn`")
        st.stop()

    if st.session_state["df"] is None:
        st.warning("No dataset loaded. Please go to **📁 Upload & Classify** first.")
        st.stop()

    df = st.session_state["df"]

    st.markdown(
        "Train a **TF-IDF + Logistic Regression** classifier on your labelled data. "
        "Once trained it will be used automatically when classifying tickets."
    )
    st.markdown("---")

    all_cols = list(df.columns)
    obj_cols = [c for c in all_cols if df[c].dtype == object]

    if not obj_cols:
        st.error("No text columns found in the dataset.")
        st.stop()

    # Column selectors
    col1, col2 = st.columns(2)
    with col1:
        guessed_text = find_text_column(df)
        default_text = obj_cols.index(guessed_text) if guessed_text in obj_cols else 0
        text_col = st.selectbox(
            "📝 Text column (ticket content)",
            obj_cols,
            index=default_text,
            key="train_text_col",
        )
    with col2:
        label_candidates = [c for c in all_cols if c != text_col]
        if not label_candidates:
            st.error("Need at least 2 columns (text + label).")
            st.stop()
        guessed_label = find_label_column(df)
        default_label = (
            label_candidates.index(guessed_label)
            if guessed_label in label_candidates else 0
        )
        label_col = st.selectbox(
            "🏷️ Label column (category/class)",
            label_candidates,
            index=default_label,
            key="train_label_col",
        )

    # Label distribution preview
    with st.expander("📊 Label distribution"):
        label_counts = safe_value_counts(
            df[label_col].dropna().astype(str), "Label", "Count"
        )
        st.dataframe(label_counts, use_container_width=True)
        if PLOTLY_AVAILABLE:
            fig = px.bar(
                label_counts, x="Label", y="Count", text="Count",
                color="Count", color_continuous_scale="Blues",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(margin=dict(t=10, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Hyperparameters")
    col_hp1, col_hp2 = st.columns(2)
    with col_hp1:
        test_size = st.slider("Test split %", 10, 40, 20, key="train_test_size") / 100
    with col_hp2:
        max_features = st.slider("TF-IDF max features", 500, 10000, 3000, 500, key="train_max_feat")

    st.markdown("---")

    if st.button("🏋️ Train Model", type="primary"):

        X = df[text_col].fillna("").astype(str)
        y = df[label_col].fillna("Other").astype(str)

        can_train = True

        # ── Validation ────────────────────────────────────────────────────────
        n_classes = y.nunique()
        if n_classes < 2:
            st.error(f"Only {n_classes} unique class found. Need at least 2 to train.")
            can_train = False

        if can_train and len(X) < 20:
            st.error("Dataset too small (fewer than 20 rows).")
            can_train = False

        X_train = X_test = y_train = y_test = None

        if can_train:
            class_counts    = y.value_counts()
            min_class_count = int(class_counts.min())
            n_test_samples  = max(1, int(len(y) * test_size))
            use_stratify    = (min_class_count >= 2) and (n_test_samples >= n_classes)

            if not use_stratify:
                st.warning(
                    f"Some classes have very few samples (min={min_class_count}). "
                    "Using random split instead of stratified."
                )
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=42,
                    stratify=(y if use_stratify else None),
                )
            except ValueError as e:
                st.error(f"Train/test split failed: {e}")
                can_train = False

        if can_train and X_train is not None:
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True,
                )),
                ("clf", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    C=1.0,
                )),
            ])

            try:
                with st.spinner("Training… please wait."):
                    pipeline.fit(X_train, y_train)
            except Exception as e:
                st.error(f"Training failed: {e}")
                can_train = False

        if can_train and X_train is not None:
            y_pred = pipeline.predict(X_test)

            # FIX: use accuracy_score — handles index alignment automatically
            acc = round(accuracy_score(y_test, y_pred) * 100, 1)

            st.session_state["model"]         = pipeline
            st.session_state["model_trained"] = True

            st.success("✅ Model trained successfully!")

            # ── Metrics row ───────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Test Accuracy",  f"{acc}%")
            m2.metric("Train Samples",  f"{len(X_train):,}")
            m3.metric("Test Samples",   f"{len(X_test):,}")
            m4.metric("Classes",        len(pipeline.classes_))

            st.markdown("---")

            # ── Classification report ─────────────────────────────────────────
            st.subheader("📋 Classification Report")
            report_dict = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            )
            report_df = pd.DataFrame(report_dict).T.round(3)

            # Drop scalar 'accuracy' row (it has NaN for precision/recall columns)
            report_df = report_df.drop(index=["accuracy"], errors="ignore")

            # Convert 'support' to int to avoid displaying as 12.0
            if "support" in report_df.columns:
                report_df["support"] = report_df["support"].fillna(0).astype(int)

            float_cols = [
                c for c in ["precision", "recall", "f1-score"]
                if c in report_df.columns
            ]
            fmt_dict = {c: "{:.3f}" for c in float_cols}
            if "support" in report_df.columns:
                fmt_dict["support"] = "{:,}"

            styled = report_df.style.format(fmt_dict, na_rep="–")
            if float_cols:
                styled = styled.background_gradient(
                    cmap="RdYlGn", subset=float_cols, vmin=0, vmax=1
                )
            st.dataframe(styled, use_container_width=True)

            st.markdown("---")

            # ── Confusion matrix ──────────────────────────────────────────────
            if PLOTLY_AVAILABLE:
                st.subheader("🔢 Confusion Matrix")

                # FIX: derive labels from values actually present in y_test & y_pred
                # avoids: "At least one label specified must be in y_true"
                present_labels = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
                cm = confusion_matrix(y_test, y_pred, labels=present_labels)

                fig = px.imshow(
                    cm,
                    x=present_labels,
                    y=present_labels,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    aspect="auto",
                )
                fig.update_layout(
                    xaxis_title="Predicted Label",
                    yaxis_title="Actual Label",
                    margin=dict(t=20, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

            # ── Top terms per class ───────────────────────────────────────────
            st.markdown("---")
            st.subheader("🔍 Top Predictive Terms per Class")
            try:
                tfidf         = pipeline.named_steps["tfidf"]
                clf           = pipeline.named_steps["clf"]
                feature_names = tfidf.get_feature_names_out()
                n_top = 8

                rows = []
                for i, cls in enumerate(clf.classes_):
                    top_idx = clf.coef_[i].argsort()[-n_top:][::-1]
                    for idx in top_idx:
                        rows.append({
                            "Class":  cls,
                            "Term":   feature_names[idx],
                            "Weight": round(float(clf.coef_[i][idx]), 4),
                        })

                feat_df = pd.DataFrame(rows)
                if PLOTLY_AVAILABLE:
                    fig = px.bar(
                        feat_df, x="Weight", y="Term", color="Class",
                        facet_col="Class", facet_col_wrap=3,
                        orientation="h",
                        color_discrete_map=CATEGORY_COLORS,
                        height=max(400, 120 * len(clf.classes_)),
                    )
                    fig.update_layout(showlegend=False, margin=dict(t=40, b=10))
                    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.dataframe(feat_df, use_container_width=True)
            except Exception:
                st.info("Feature importance display not available for this model.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – Classify Single Ticket
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "✏️ Classify Single Ticket":
    st.title("✏️ Classify a Single Ticket")

    st.markdown(
        "Paste any support ticket text below and get an instant "
        "category + priority prediction."
    )

    ticket_text = st.text_area(
        "Ticket text",
        height=160,
        placeholder=(
            "e.g. I was charged twice for my subscription last month "
            "and need an immediate refund…"
        ),
        label_visibility="collapsed",
    )

    if st.button("🔍 Classify Ticket", type="primary"):
        if not ticket_text.strip():
            st.warning("Please enter some ticket text before classifying.")
        else:
            proba   = None
            classes = None

            if st.session_state["model_trained"] and SKLEARN_AVAILABLE:
                model    = st.session_state["model"]
                category = str(model.predict([ticket_text])[0])
                proba    = model.predict_proba([ticket_text])[0].tolist()
                classes  = [str(c) for c in model.classes_]
                method   = "ML Model"
            else:
                category = keyword_classify(ticket_text)
                method   = "Keyword Rules"

            priority = estimate_priority(ticket_text)

            # Result cards
            st.markdown("### 🎯 Result")
            r1, r2, r3 = st.columns(3)
            with r1:
                metric_card(
                    category, "Predicted Category",
                    color=CATEGORY_COLORS.get(category, "#555"),
                )
            with r2:
                metric_card(
                    priority, "Estimated Priority",
                    color=PRIORITY_COLORS.get(priority, "#555"),
                )
            with r3:
                metric_card("🔧", f"Method: {method}")

            # Confidence bar chart
            if proba is not None and classes is not None and PLOTLY_AVAILABLE:
                st.subheader("📊 Confidence Scores")
                conf_df = (
                    pd.DataFrame({"Category": classes, "Confidence": proba})
                    .sort_values("Confidence", ascending=True)
                )
                fig = px.bar(
                    conf_df, x="Confidence", y="Category", orientation="h",
                    color="Confidence", color_continuous_scale="RdYlGn",
                    range_x=[0, 1], text="Confidence",
                )
                fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
                fig.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

            # Suggested reply
            if category in SUGGESTED_REPLIES:
                st.subheader("💬 Suggested Response")
                st.info(SUGGESTED_REPLIES[category])
                st.code(SUGGESTED_REPLIES[category], language=None)