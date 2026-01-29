import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import matplotlib.pyplot as plt

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(page_title="NYC Airbnb ARM", layout="wide")
st.title("🏙️ NYC Airbnb Association Rule Mining")


# ---------------------------
# Page Navigation
# ---------------------------
page = st.sidebar.radio(
    "📄 Select Page",
    ["🔍 ARM Explorer", "📊 Simple Dashboard"]
)

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

# ---------------------------
# Preprocessing (from notebook)
# ---------------------------
@st.cache_data
def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Implements the same cleaning + feature engineering used in the provided notebook.
    Returns a cleaned + engineered dataframe ready for transaction encoding.
    """
    selected_cols = [
        'host_identity_verified',
        'neighbourhood group',
        'neighbourhood',
        'lat',
        'long',
        'instant_bookable',
        'cancellation_policy',
        'room type',
        'Construction year',
        'price',
        'service fee',
        'minimum nights',
        'number of reviews',
        'reviews per month',
        'review rate number',
        'calculated host listings count',
        'availability 365'
    ]

    df = df_raw[selected_cols].copy()

    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Convert price & service fee to numeric (strip "$" and commas)
    for col in ["price", "service_fee"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .replace(["nan", "None", ""], np.nan),
                errors="coerce"
            )

    # ---------------------------
    # Fill Nulls
    # ---------------------------

    # Fix common misspellings in neighbourhood_group
    df['neighbourhood_group'] = df['neighbourhood_group'].replace({
        'brookln': 'Brooklyn',
        'manhatan': 'Manhattan'
    })

    # Fill missing neighbourhood_group using neighbourhood mode mapping
    mapping = (
        df.dropna(subset=['neighbourhood_group'])
          .groupby('neighbourhood')['neighbourhood_group']
          .agg(lambda x: x.mode().iloc[0])
          .to_dict()
    )
    df['neighbourhood_group'] = df.apply(
        lambda row: mapping.get(row['neighbourhood'], row['neighbourhood_group'])
        if pd.isna(row['neighbourhood_group'])
        else row['neighbourhood_group'],
        axis=1
    )

    # Fill missing neighbourhood using neighbourhood_group + closest (lat,long)
    known = df[df['neighbourhood'].notna()].copy()

    def fill_neighbourhood(row):
        candidates = known[known['neighbourhood_group'] == row['neighbourhood_group']]
        if candidates.empty or pd.isna(row['lat']) or pd.isna(row['long']):
            return row['neighbourhood']
        # Find closest lat/lon (Euclidean distance)
        distances = (candidates['lat'] - row['lat'])**2 + (candidates['long'] - row['long'])**2
        closest_idx = distances.idxmin()
        return candidates.loc[closest_idx, 'neighbourhood']

    df['neighbourhood'] = df.apply(
        lambda row: fill_neighbourhood(row) if pd.isna(row['neighbourhood']) else row['neighbourhood'],
        axis=1
    )

    # Drop rows missing geo coordinates
    df = df.dropna(subset=['lat', 'long'])

    # host_identity_verified: add "unknown"
    df['host_identity_verified'] = df['host_identity_verified'].fillna('unknown')

    # Replace with mode for these categorical cols
    for col in ['instant_bookable', 'cancellation_policy']:
        if col in df.columns and df[col].notna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])


    # Fill with median for numeric cols used later
    median_columns = [
        'construction_year',
        'minimum_nights',
        'number_of_reviews',
        'review_rate_number',
        'calculated_host_listings_count',
        'availability_365',

    ]
    for col in median_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Compute typical service fee ratio (median), then fill missing price/service_fee
    fee_ratio = (df['service_fee'] / df['price']).median()
    if not np.isfinite(fee_ratio) or fee_ratio <= 0:
        fee_ratio = 0.2  # fallback if ratio can't be computed

    mask_fee = df['service_fee'].isna() & df['price'].notna()
    df.loc[mask_fee, 'service_fee'] = df.loc[mask_fee, 'price'] * fee_ratio

    mask_price = df['price'].isna() & df['service_fee'].notna()
    df.loc[mask_price, 'price'] = df.loc[mask_price, 'service_fee'] / fee_ratio

    df['price'] = df['price'].fillna(df['price'].median())
    df['service_fee'] = df['service_fee'].fillna(df['service_fee'].median())

    # Fill missing with 0 where there are no reviews
    mask_zero_reviews = df['reviews_per_month'].isna() & (df['number_of_reviews'] == 0)
    df.loc[mask_zero_reviews, 'reviews_per_month'] = 0
    df['reviews_per_month'] = df['reviews_per_month'].fillna(df['reviews_per_month'].median())

    # Drop duplicates
    df = df.drop_duplicates()

    # ---------------------------
    # Feature Engineering
    # ---------------------------
    df_engineering = df.copy()

    df_engineering["price_level"] = pd.cut(
        df_engineering["price"],
        bins=[0, 50, 100, 200, 500, 1000, np.inf],
        labels=["very_low", "low", "medium", "high", "very_high", "luxury"]
    )

    df_engineering["stay_type"] = pd.cut(
        df_engineering["minimum_nights"],
        bins=[0, 7, 30, np.inf],
        labels=["short_stay", "medium_stay", "long_stay"]
    )

    df_engineering["reviews_popularity"] = pd.cut(
        df_engineering["number_of_reviews"],
        bins=[-1, 10, 50, 200, np.inf],
        labels=["low", "medium", "high", "very_high"]
    )

    df_engineering["review_quality"] = pd.cut(
        df_engineering["review_rate_number"],
        bins=[0, 3, 4, 5],
        labels=["low", "medium", "high"],
        include_lowest=True
    )

    df_engineering["review_activity"] = pd.cut(
        df_engineering["reviews_per_month"],
        bins=[-0.01, 0.5, 2, np.inf],
        labels=["inactive", "moderate", "active"]
    )

    df_engineering["availability_level"] = pd.cut(
        df_engineering["availability_365"],
        bins=[-1, 60, 180, np.inf],
        labels=["rarely_available", "seasonal", "mostly_available"]
    )

    df_engineering["host_experience"] = pd.cut(
        df_engineering["calculated_host_listings_count"],
        bins=[-1, 1, 5, 20, np.inf],
        labels=["single", "small_portfolio", "experienced", "professional"]
    )

    df_engineering["total_min_cost"] = df_engineering["price"] * df_engineering["minimum_nights"]
    df_engineering["total_cost_level"] = pd.cut(
        df_engineering["total_min_cost"],
        bins=[0, 100, 500, 2000, np.inf],
        labels=["budget", "mid_range", "expensive", "luxury_investment"]
    )

    df_engineering["review_density"] = df_engineering["number_of_reviews"] / (df_engineering["availability_365"] + 1)
    df_engineering["review_density_level"] = pd.cut(
        df_engineering["review_density"],
        bins=[-0.01, 0.05, 0.2, np.inf],
        labels=["low", "medium", "high"]
    )

    return df_engineering

def calculate_support_confidence(df, conditions, target_col, target_value):
    """
    conditions: list of (column, value)
    """
    if not conditions:
        return None

    # Filter antecedent
    antecedent_df = df.copy()
    for col, val in conditions:
        antecedent_df = antecedent_df[antecedent_df[col] == val]

    antecedent_count = len(antecedent_df)
    total_count = len(df)

    if antecedent_count == 0:
        return 0, 0, 0

    # Filter consequent
    both_df = antecedent_df[antecedent_df[target_col] == target_value]
    both_count = len(both_df)

    support = both_count / total_count
    confidence = both_count / antecedent_count

    return antecedent_count, support, confidence
    
# =====================================================
# Page 1
# =====================================================
if page == "🔍 ARM Explorer":

    st.header("🔍 Association Rule Mining Explorer")

    # ---------------------------
    # Data Source UI
    # ---------------------------
    st.sidebar.header("🐣 Data Source")
    default_path = "Airbnb_Open_Data.csv"
    csv_path = st.sidebar.text_input("CSV path", value=default_path)

    try:
        df_raw = load_data(csv_path)
    except Exception as e:
        st.error(f"Failed to load CSV from '{csv_path}'. Error: {e}")
        st.stop()

    df = preprocess(df_raw)

    st.session_state.df = df

    st.success("Dataset loaded + preprocessed successfully")
    st.write("Dataset shape (after preprocessing):", df.shape)

    with st.expander("Preview Cleaned + Engineered Data", expanded=False):
        st.dataframe(df.head(30), use_container_width=True)

    # ---------------------------
    # Transaction Encoding
    # ---------------------------
    categorical_cols = [
        "host_identity_verified",
        "neighbourhood_group",
        "instant_bookable",
        "cancellation_policy",
        "room_type",
        "price_level",
        "stay_type",
        "reviews_popularity",
        "review_quality",
        "review_activity",
        "availability_level",
        "host_experience",
        "total_cost_level",
        "review_density_level"
    ]

    # Keep only existing cols (safety)
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    df_assoc = df[categorical_cols].dropna()

    transaction_df = pd.get_dummies(df_assoc)

    st.subheader("📦 Transaction Matrix")
    st.write("Shape:", transaction_df.shape)
    st.dataframe(transaction_df.head(), use_container_width=True)

    # ---------------------------
    # Sidebar Parameters
    # ---------------------------
    st.sidebar.header("⚙️ ARM Parameters")

    algorithm = st.sidebar.selectbox(
        "Algorithm",
        ["Apriori", "FP-Growth"]
    )

    min_support = st.sidebar.slider(
        "Minimum Support",
        0.01, 0.6, 0.3
    )

    min_confidence = st.sidebar.slider(
        "Minimum Confidence",
        0.1, 1.0, 0.6
    )

    # ---------------------------
    # Generate Frequent Itemsets
    # ---------------------------
    if algorithm == "Apriori":
        frequent_itemsets = apriori(
            transaction_df,
            min_support=min_support,
            use_colnames=True
        )
    else:
        frequent_itemsets = fpgrowth(
            transaction_df,
            min_support=min_support,
            use_colnames=True
        )

    if frequent_itemsets.empty:
        st.warning("No frequent itemsets found. Try lowering support.")
        st.stop()

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    if rules.empty:
        st.warning("No association rules generated. Try lowering confidence.")
        st.stop()

    # ---------------------------
    # Rule-Based Filters
    # ---------------------------
    st.sidebar.header("🧸 Rule Filters")

    rules = rules.copy()
    rules["ante_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["cons_len"] = rules["consequents"].apply(lambda x: len(x))

    all_items = transaction_df.columns.tolist()

    # ---------------------------
    # Item Matching
    # ---------------------------
    st.sidebar.subheader("🎀 Item Constraints")

    match_mode = st.sidebar.radio(
        "Item match mode",
        ["Contains ANY selected", "Contains ALL selected", "Exact match (set equals)"],
        index=0
    )

    selected_antecedent = st.sidebar.multiselect(
        "Antecedent items",
        all_items
    )

    selected_consequent = st.sidebar.multiselect(
        "Consequent items",
        all_items
    )

    def match_set(rule_set, selected, mode):
        if not selected:
            return True
        rule_set = set(rule_set)
        selected = set(selected)

        if mode == "Contains ANY selected":
            return any(item in rule_set for item in selected)
        elif mode == "Contains ALL selected":
            return selected.issubset(rule_set)
        else:  # Exact match
            return rule_set == selected

    # Apply item filtering
    rules = rules[
        rules.apply(
            lambda r: match_set(r["antecedents"], selected_antecedent, match_mode)
                    and match_set(r["consequents"], selected_consequent, match_mode),
            axis=1
        )
    ]

    # ---------------------------
    # Rule Size Filters
    # ---------------------------
    st.sidebar.subheader("📐 Rule Size")

    if rules.empty:
        st.warning("No rules match the selected filters.")
        st.stop()

    ante_min = int(rules["ante_len"].min())
    ante_max = int(rules["ante_len"].max())
    cons_min = int(rules["cons_len"].min())
    cons_max = int(rules["cons_len"].max())

    # ----- Antecedent size -----
    if ante_min == ante_max:
        st.sidebar.markdown(f"**Antecedent size:** {ante_min}")
        ante_len_rng = (ante_min, ante_max)
    else:
        ante_len_rng = st.sidebar.slider(
            "Antecedent size",
            ante_min,
            ante_max,
            (ante_min, ante_max)
        )

    # ----- Consequent size -----
    if cons_min == cons_max:
        st.sidebar.markdown(f"**Consequent size:** {cons_min}")
        cons_len_rng = (cons_min, cons_max)
    else:
        cons_len_rng = st.sidebar.slider(
            "Consequent size",
            cons_min,
            cons_max,
            (cons_min, cons_max)
        )

    rules = rules[
        rules["ante_len"].between(ante_len_rng[0], ante_len_rng[1]) &
        rules["cons_len"].between(cons_len_rng[0], cons_len_rng[1])
    ]

    # ---------------------------
    # Sorting & Top-K
    # ---------------------------
    st.sidebar.subheader("✨ Sorting & Display")

    sort_by = st.sidebar.selectbox("Sort by", ["lift", "confidence", "support"], index=0)
    sort_dir = st.sidebar.radio("Order", ["Descending", "Ascending"], horizontal=True, index=0)
    top_k = st.sidebar.slider("Show top K rules", 10, 100, 20, step=10)

    rules_sorted = rules.sort_values(sort_by, ascending=(sort_dir == "Ascending"))

    total_found = len(rules_sorted)
    rules_topk = rules_sorted.head(top_k)
    shown = len(rules_topk)

    # ---------------------------
    # Empty Check
    # ---------------------------
    if rules.empty:
        st.warning("No rules match the selected filters.")
        st.stop()

    # ---------------------------
    # Sort & Display Rules
    # ---------------------------
    st.subheader("📊 Association Rules")

    rules_display = rules_topk[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    rules_display["antecedents"] = rules_display["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules_display["consequents"] = rules_display["consequents"].apply(lambda x: ", ".join(list(x)))

    st.dataframe(rules_display, use_container_width=True)

    st.info(
        f"Showing top {shown} rules out of {total_found} found "
        f"(sorted by {sort_by}, {sort_dir.lower()}) | "
        f"{algorithm} (support ≥ {min_support}, confidence ≥ {min_confidence})"
    )
   
    # ---------------------------
    # Visualization
    # ---------------------------
    st.subheader(f"📈 {algorithm} Association Rules: Lift vs Support (Bubble = Support)")

    plot_df = rules_topk
    fig, ax = plt.subplots(figsize=(8, 3))
    sc = ax.scatter(
        plot_df["support"],
        plot_df["lift"],
        s=plot_df["support"] * 1000,
        c=plot_df["confidence"],
        edgecolors="black",
        linewidths=0.7,
        alpha=0.7
    )
    fig.colorbar(sc, ax=ax, label="Confidence")
    ax.grid(True)
    ax.set_xlabel("Support")
    ax.set_ylabel("Lift")
    st.pyplot(fig)

# =====================================================
# Page 2
# =====================================================

else:
    if "df" not in st.session_state:
        st.warning("⚠️ Please load the dataset in the ARM Explorer page first.")
        st.stop()

    df = st.session_state.df

    if "conditions" not in st.session_state:
        st.session_state.conditions = []

    selectable_cols = [
        "host_identity_verified",
        "neighbourhood_group",
        "instant_bookable",
        "cancellation_policy",
        "room_type",
        "price_level",
        "stay_type",
        "reviews_popularity",
        "review_quality",
        "review_activity",
        "availability_level",
        "host_experience",
        "total_cost_level",
        "review_density_level"
    ]

    selectable_cols = [c for c in selectable_cols if c in df.columns]

    # Friendly labels
    pretty_names = {
        "neighbourhood_group": "Area",
        "room_type": "Room Type",
        "price_level": "Price Range",
        "stay_type": "Length of Stay",
        "instant_bookable": "Instant Bookable",
    }

    def pretty_col(c: str) -> str:
        return pretty_names.get(c, c.replace("_", " ").title())

    def format_filters(conds):
        if not conds:
            return "No filters selected"
        return " • ".join([f"{pretty_col(c)} = {v}" for c, v in conds])

    # =====================================================
    # SIMPLE DASHBOARD (NON-TECHNICAL)
    # =====================================================
    st.header("📊 Simple Dashboard")
    st.caption("Choose filters, choose what you want to see, then click **Show Results**.")

    # ---------------------------
    # Step 1: Filters
    # ---------------------------
    st.markdown("### Step 1 — Choose filters")

    used_cols = {c for c, _ in st.session_state.conditions}
    available_cols = [c for c in selectable_cols if c not in used_cols]

    a, b, c = st.columns([3, 3, 1])

    with a:
        if not available_cols:
            st.info("All filter types are used. Remove one to add a new filter.")
            selected_col = None
        else:
            selected_col = st.selectbox(
                "Filter type",
                available_cols,
                format_func=pretty_col,
                key="cond_col"
            )

    with b:
        if selected_col is None:
            selected_val = None
            st.selectbox("Value", ["-"], disabled=True)
        else:
            selected_val = st.selectbox(
                "Value",
                sorted(df[selected_col].dropna().unique()),
                key="cond_val"
            )

    with c:
        st.write("")
        add_clicked = st.button("➕ Add Filter", use_container_width=True)

    if add_clicked:
        if selected_col is None or selected_val is None:
            st.warning("Please choose a filter type and value.")
        elif (selected_col, selected_val) in st.session_state.conditions:
            st.warning("This filter is already added.")
        else:
            st.session_state.conditions.append((selected_col, selected_val))
            st.success(f"Added: {pretty_col(selected_col)} = {selected_val}")

    st.markdown("#### Your filters")
    if not st.session_state.conditions:
        st.info("No filters yet. Add at least one filter to see results.")
    else:
        for i, (fc, fv) in enumerate(st.session_state.conditions):
            r1, r2 = st.columns([10, 2])
            with r1:
                st.write(f"✅ **{pretty_col(fc)}:** {fv}")
            with r2:
                if st.button("Remove", key=f"rm_{i}", use_container_width=True):
                    st.session_state.conditions.pop(i)
                    st.rerun()

        col_clear, col_hint = st.columns([2, 8])
        with col_clear:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.conditions = []
                st.rerun()
        with col_hint:
            st.caption("Tip: remove one filter to change it.")

    st.markdown("---")

    # ---------------------------
    # Step 2: What to show
    # ---------------------------
    st.markdown("### Step 2 — Choose what you want to see")

    target_col = st.selectbox(
        "Show results for",
        selectable_cols,
        format_func=pretty_col,
        key="target_col"
    )

    st.markdown("---")

    # ---------------------------
    # Step 3: Results
    # ---------------------------
    st.markdown("### Step 3 — Results")
    st.caption("Confidence = % within your filters. Support = % out of all listings.")

    show_clicked = st.button("🔍 Show Results", use_container_width=True)

    if show_clicked:
        if not st.session_state.conditions:
            st.warning("Please add at least one filter in Step 1.")
        else:
            # Build filtered set
            mask = pd.Series(True, index=df.index)
            for fc, fv in st.session_state.conditions:
                mask &= (df[fc] == fv)

            filtered_df = df[mask]
            filtered_count = len(filtered_df)
            total_count = len(df)

            if filtered_count == 0:
                st.warning("No listings match your filters. Try removing or changing a filter.")
            else:
                # Show ALL possible values (even those with 0%)
                all_values = sorted(df[target_col].dropna().unique())

                rows = []
                for val in all_values:
                    both_count = int((filtered_df[target_col] == val).sum())

                    confidence = (both_count / filtered_count) if filtered_count else 0
                    support = (both_count / total_count) if total_count else 0

                    rows.append({
                        "Result": f"{pretty_col(target_col)} = {val}",
                        "Confidence (%)": round(confidence * 100, 2),
                        "Support (%)": round(support * 100, 2),
                    })
                
                result_df = pd.DataFrame(rows).sort_values("Confidence (%)", ascending=False)
                # Check if ALL results have 0 support
                if (result_df["Support (%)"] == 0).all():
                    st.warning("❌ No results found for the selected filters.")
                    st.info("Try changing or removing some filters to see meaningful results.")
                    st.stop()

                # Show chosen filters briefly (non-technical)
                st.markdown("#### 📌 Your selected filters")
                st.write(format_filters(st.session_state.conditions))

                # Simple table only
                st.markdown("#### 📋 Result table")
                st.dataframe(result_df[["Result", "Confidence (%)", "Support (%)"]], use_container_width=True)

                # Graph with % labels
                st.markdown("#### 📈 Confidence chart (within your filters)")
                chart_df = result_df.sort_values("Confidence (%)", ascending=True)

                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.barh(chart_df["Result"], chart_df["Confidence (%)"])
                ax.set_xlabel("Confidence (%)")
                ax.set_ylabel("")
                ax.grid(True, axis="x")

                # Add % labels on bars
                for bar, val in zip(bars, chart_df["Confidence (%)"]):
                    ax.text(
                        bar.get_width() + 0.3,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.2f}%",
                        va="center"
                    )

                st.pyplot(fig)
                # --- Highlight card: Most Likely Outcome ---
                # --- Highlight card: Filter → Most Likely Result ---
                top = result_df.iloc[0]

                st.markdown(
                    f"""
                    <div style="
                        padding:18px;
                        border-radius:14px;
                        background:#f0f7ff;
                        border:1px solid #cfe0ff;
                        margin-bottom:16px;">
                        <div style="font-size:18px; font-weight:700; margin-bottom:6px;">
                            ⭐ Most Likely Result for Your Filters
                        </div>
                        <div style="font-size:15px; margin-bottom:10px;">
                            Listings that match your filters are most likely to have:
                        </div>
                        <div style="font-size:17px; font-weight:600; margin-bottom:10px;">
                            {top['Result']}
                        </div>
                        <div style="font-size:14px; color:#333;">
                            Likelihood: <b>{top['Confidence (%)']:.2f}%</b><br/>
                            Overall presence: <b>{top['Support (%)']:.2f}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("CDS6314 Data Mining | NYC Airbnb Analysis")
