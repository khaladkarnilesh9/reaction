import streamlit as st
import torch
import torch.nn.functional as F
import re, spacy, pandas as pd, json
from transformers import BertTokenizerFast, BertForSequenceClassification
from datasets import Dataset
from graphviz import Digraph
from PIL import Image

# -------------------------------
# 1️⃣ Setup labels & model
# -------------------------------
labels = ["Preparation", "Reaction", "Work-up", "Purification", "Analysis", "Other"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

data = {
    "sentence": [
        "Weigh 2 g of NaCl.",
        "Dissolve NaCl in 50 mL of water.",
        "Add ethanol and stir for 2 hours at 60°C.",
        "Cool the reaction mixture to room temperature.",
        "Filter the solid and dry under vacuum.",
        "Purify the product by recrystallization.",
        "Analyze the sample by NMR spectroscopy."
    ],
    "label": [
        "Preparation", "Preparation", "Reaction", "Reaction",
        "Work-up", "Purification", "Analysis"
    ]
}

# Manual encoding for compatibility
data["label_id"] = [label2id[l] for l in data["label"]]
dataset = Dataset.from_dict(data)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["sentence"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.3)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# -------------------------------
# 2️⃣ NLP setup
# -------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy model 'en_core_web_sm' not found. Please run: `python -m spacy download en_core_web_sm`")
    st.stop()

def extract_entities(sentence):
    doc = nlp(sentence)
    entities = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, []).append(ent.text)
    temp = re.findall(r"\d+\s?°C", sentence)
    amt = re.findall(r"\d+(?:\.\d+)?\s?(?:g|mg|mL|L|mol)", sentence)
    time = re.findall(r"\d+\s?(?:h|hour|hours|min|minutes)", sentence)
    if temp: entities["Temperature"] = temp
    if amt: entities["Amount"] = amt
    if time: entities["Duration"] = time
    return entities

substage_keywords = {
    "Preparation": {"weigh": "Weighing", "dissolve": "Dissolution", "mix": "Mixing", "add": "Addition"},
    "Reaction": {"heat": "Heating", "reflux": "Refluxing", "stir": "Stirring", "cool": "Cooling"},
    "Work-up": {"filter": "Filtration", "wash": "Washing", "dry": "Drying", "extract": "Extraction"},
    "Purification": {"recrystall": "Recrystallization", "column": "Chromatography"},
    "Analysis": {"nmr": "NMR", "ir": "IR", "ms": "Mass Spectrometry"}
}

def detect_substage(stage, sentence):
    sentence_lower = sentence.lower()
    if stage in substage_keywords:
        for k, v in substage_keywords[stage].items():
            if k in sentence_lower:
                return v
    return "General"

# -------------------------------
# 3️⃣ Parsing logic
# -------------------------------
def parse_reaction(text):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        return None, None, None

    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).tolist()

    results = []
    for i, (p, s) in enumerate(zip(preds, sentences)):
        stage = id2label[p]
        substage = detect_substage(stage, s)
        ents = extract_entities(s)
        results.append({"Step": i+1, "Stage": stage, "Sub-Stage": substage, "Sentence": s, "Entities": ents})

    df = pd.DataFrame([{
        "Step": r["Step"],
        "Stage": r["Stage"],
        "Sub-Stage": r["Sub-Stage"],
        "Sentence": r["Sentence"],
        "Entities": str(r["Entities"])
    } for r in results])

    dot = Digraph(comment='Reaction Procedure Flow', format='png')
    for r in results:
        label = f"Step {r['Step']}\n{r['Stage']} → {r['Sub-Stage']}\n{r['Sentence']}"
        dot.node(str(r['Step']), label)
    for i in range(len(results) - 1):
        dot.edge(str(results[i]['Step']), str(results[i + 1]['Step']))
    
    graph_path = dot.render('reaction_flow', cleanup=True)
    return df, results, graph_path

# -------------------------------
# 4️⃣ Streamlit UI
# -------------------------------
st.set_page_config(page_title="Chemical Stage Parser", page_icon="🧪", layout="wide")

# 🌈 Theme toggle
theme_choice = st.sidebar.radio("🌗 Theme", ["Dark", "Light"], index=0)
dark_mode = theme_choice == "Dark"

bg_color = "#0f172a" if dark_mode else "#f9fafb"
text_color = "#f9fafb" if dark_mode else "#0f172a"
accent_color = "#38bdf8"

# 🧭 Navbar
st.markdown(f"""
    <style>
    .navbar {{
        position: sticky;
        top: 0;
        z-index: 100;
        background-color: {bg_color};
        padding: 1rem 2rem;
        border-radius: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.3);
    }}
    .navbar h1 {{
        font-size: 1.6rem;
        color: {accent_color};
        margin: 0;
        font-weight: 700;
    }}
    .navbar a {{
        color: {text_color};
        margin-left: 1.5rem;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.3s ease, transform 0.2s ease;
    }}
    .navbar a:hover {{
        color: {accent_color};
        transform: scale(1.1);
    }}
    </style>
    <div class="navbar">
        <h1>🧪 Reaction Parser</h1>
        <div>
            <a href="#">Home</a>
            <a href="#">Upload</a>
            <a href="#">Docs</a>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown(
    f"<h3 style='color:{text_color};margin-top:1rem;'>Chemical Reaction Stage + Sub-Stage Parser with Flow Diagram</h3>",
    unsafe_allow_html=True
)

# -------------------------------
# 5️⃣ Input section
# -------------------------------
st.write("You can either type a reaction procedure or upload a file:")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("✍️ Enter Reaction Procedure", height=200, placeholder="Type or paste your procedure here...")

with col2:
    uploaded_file = st.file_uploader("📂 Upload .txt or .csv file", type=["txt", "csv"])
    if uploaded_file:
        if uploaded_file.name.endswith(".txt"):
            content = uploaded_file.read()
            user_input = content.decode("utf-8") if isinstance(content, bytes) else content
        elif uploaded_file.name.endswith(".csv"):
            df_uploaded = pd.read_csv(uploaded_file)
            user_input = " ".join(df_uploaded.iloc[:, 0].astype(str).tolist())

# -------------------------------
# 6️⃣ Processing
# -------------------------------
if st.button("Analyze Reaction"):
    if user_input.strip():
        with st.spinner("🔍 Parsing your procedure..."):
            df, results, graph_path = parse_reaction(user_input)
        if df is not None:
            st.success("✅ Parsing Complete!")

            st.subheader("📋 Parsed Steps")
            st.dataframe(df, use_container_width=True)

            st.download_button(
                label="⬇️ Download as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="reaction_parsed.csv",
                mime="text/csv"
            )

            st.download_button(
                label="⬇️ Download as JSON",
                data=json.dumps(results, indent=4).encode('utf-8'),
                file_name="reaction_parsed.json",
                mime="application/json"
            )

            st.subheader("🔗 Reaction Flow Diagram")
            image = Image.open(graph_path)
            st.image(image, caption="Reaction Flow Diagram", use_column_width=True)

            st.subheader("🧠 JSON Output")
            st.json(results)

        else:
            st.error("⚠️ No valid sentences found.")
    else:
        st.warning("Please provide a reaction procedure first!")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit, Transformers, and SpaCy.")
