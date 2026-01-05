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
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Airbnb_Open_Data.csv")

df = load_data()

st.success("Dataset loaded successfully")
st.write("Dataset shape:", df.shape)
st.dataframe(df.head())

# ---------------------------
# Data Cleaning
# ---------------------------
selected_cols = ['host_identity_verified','neighbourhood group', 'neighbourhood', 'lat', 'long', 'instant_bookable', 
                 'cancellation_policy', 'room type', 'Construction year', 'price', 'service fee', 'minimum nights', 
                 'number of reviews', 'reviews per month','review rate number', 
                 'calculated host listings count','availability 365']
df_selected = df[selected_cols].copy()

# Standardize column names
df_selected.columns = df_selected.columns.str.lower().str.replace(" ", "_")

# Convert price & service fee to numeric
for col in ["price", "service_fee"]:
    if col in df_selected.columns:
        df_selected[col] = (
            pd.to_numeric(
                df_selected[col]           # original column
                .astype(str)               # convert everything to string
                .str.replace("$", "", regex=False)  # remove $
                .str.replace(",", "", regex=False)  # remove commas
                .replace(["nan", "None", ""], np.nan), # convert invalid strings to NaN
                errors="coerce"           # convert to numeric
            )
        )

# ---------------------------
# Fill Null Value
# ---------------------------
# fill in neighbourhood_group nil value by refering to neighbour
mapping = df_selected.dropna(subset=['neighbourhood_group']).groupby('neighbourhood')['neighbourhood_group'].agg(lambda x: x.mode()[0]).to_dict()
df_selected['neighbourhood_group'] = df_selected.apply(
    lambda row: mapping.get(row['neighbourhood'], row['neighbourhood_group']) if pd.isna(row['neighbourhood_group']) else row['neighbourhood_group'],
    axis=1
)

# # fill in neighbourhood nil value by refering to neighbourhood_group and lat & long
known = df_selected[df_selected['neighbourhood'].notna()]
unknown = df_selected[df_selected['neighbourhood'].isna()]

def fill_neighbourhood(row):
    # Filter only same neighbourhood_group
    candidates = known[known['neighbourhood_group'] == row['neighbourhood_group']]
    if candidates.empty:
        return row['neighbourhood']  # no info to fill
    
    # Find closest lat/lon (Euclidean distance)
    distances = (candidates['long'] - row['lat'])**2 + (candidates['long'] - row['long'])**2
    closest_idx = distances.idxmin()
    
    return candidates.loc[closest_idx, 'neighbourhood']

# Step 2: Apply to fill missing neighbourhoods
df_selected['neighbourhood'] = df_selected.apply(
    lambda row: fill_neighbourhood(row) if pd.isna(row['neighbourhood']) else row['neighbourhood'],
    axis=1
)

# Fill missing price using service_fee
df_selected['price'] = df_selected.apply(
    lambda row: row['service_fee'] / 0.2 if pd.isna(row['price']) and not pd.isna(row['service_fee']) else row['price'],
    axis=1
)

# Fill missing service_fee using price
df_selected['service_fee'] = df_selected.apply(
    lambda row: np.floor(row['price'] * 0.2) if pd.isna(row['service_fee']) and not pd.isna(row['price']) else row['service_fee'],
    axis=1
)

# Create price range categories
df_selected["price_range"] = pd.cut(
    df_selected["price"],
    bins=[0, 50, 100, 200, 500, 1000, np.inf],
    labels=["0-50", "50-100", "100-200", "200-500", "500-1000", "1000+"]
)

# ---------------------------
# Transaction Encoding
# ---------------------------
transactions = df_selected[[
    "room_type",
    "neighbourhood_group",
    "neighbourhood",
    "price_range",
    "instant_bookable",
    "host_identity_verified"
]]

transaction_df = pd.get_dummies(transactions)

st.subheader("📦 Transaction Matrix")
st.write("Shape:", transaction_df.shape)
st.dataframe(transaction_df.head())

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
    0.01, 0.2, 0.05
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
# Rule-Based Filters (IMPORTANT)
# ---------------------------
st.sidebar.header("🧩 Rule Filters")

all_items = transaction_df.columns.tolist()

selected_antecedent = st.sidebar.multiselect(
    "Filter by Antecedent Item",
    all_items
)

selected_consequent = st.sidebar.multiselect(
    "Filter by Consequent Item",
    all_items
)

# Apply filters
if selected_antecedent:
    rules = rules[
        rules["antecedents"].apply(
            lambda x: any(item in x for item in selected_antecedent)
        )
    ]

if selected_consequent:
    rules = rules[
        rules["consequents"].apply(
            lambda x: any(item in x for item in selected_consequent)
        )
    ]

if rules.empty:
    st.warning("No rules match the selected antecedent/consequent filters.")
    st.stop()

# ---------------------------
# Sort & Display Rules
# ---------------------------
rules = rules.sort_values("lift", ascending=False)
total_rules = rules.shape[0]

st.subheader("📊 Association Rules")

rules_display = rules[
    ["antecedents", "consequents", "support", "confidence", "lift"]
].head(20).copy()

rules_display["antecedents"] = rules_display["antecedents"].apply(
    lambda x: ", ".join(list(x))
)
rules_display["consequents"] = rules_display["consequents"].apply(
    lambda x: ", ".join(list(x))
)

st.dataframe(rules_display, use_container_width=True)

st.info(
    f"Displaying top 20 rules out of {total_rules} discovered "
    f"using {algorithm} (support ≥ {min_support}, confidence ≥ {min_confidence})"
)

# ---------------------------
# Visualization
# ---------------------------
st.subheader(f"📈 {algorithm} Association Rules: Lift vs Support (Bubble = Support)")

fig, ax = plt.subplots(figsize=(8, 3))
sc = ax.scatter(
    rules['support'],
    rules['lift'],
    s=rules['support'] * 1000,
    c=rules['confidence'],
    edgecolors='black',
    linewidths=0.7,
    alpha=0.7
)

fig.colorbar(sc, ax=ax, label='Confidence')
ax.grid(True)

ax.set_xlabel("Support")
ax.set_ylabel("Lift")
st.pyplot(fig)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("CDS6314 Data Mining | Association Rule Mining on NYC Airbnb Open Data")
