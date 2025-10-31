import streamlit as st
import torch
import torch.nn.functional as F
import re, spacy, pandas as pd, json, subprocess, sys
from transformers import BertTokenizerFast, BertForSequenceClassification
from graphviz import Digraph
from PIL import Image

# ---------------------------------------
# ⚙️ Streamlit Page Config
# ---------------------------------------
st.set_page_config(page_title="Chemical Stage Parser", page_icon="🧪", layout="wide")
st.info("⏳ Initializing app... please wait a few seconds while models load...")

# ---------------------------------------
# ⚡ Cached Resource Loaders
# ---------------------------------------
@st.cache_resource
def load_model():
    """Load and cache a small transformer model."""
    tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")
    model = BertForSequenceClassification.from_pretrained(
        "prajjwal1/bert-tiny",
        num_labels=6,
        id2label={i: l for i, l in enumerate(["Preparation", "Reaction", "Work-up", "Purification", "Analysis", "Other"])},
        label2id={l: i for i, l in enumerate(["Preparation", "Reaction", "Work-up", "Purification", "Analysis", "Other"])}
    )
    return tokenizer, model

@st.cache_resource
def load_spacy():
    """Load SpaCy English model (auto-download if missing)."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.warning("⚙️ Downloading SpaCy English model... (first run only)")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        return spacy.load("en_core_web_sm")

# ✅ Load model + tokenizer + SpaCy now
tokenizer, model = load_model()
nlp = load_spacy()

# ---------------------------------------
# 📚 Setup
# ---------------------------------------
labels = ["Preparation", "Reaction", "Work-up", "Purification", "Analysis", "Other"]
id2label = {i: l for i, l in enumerate(labels)}

substage_keywords = {
    "Preparation": {"weigh": "Weighing", "dissolve": "Dissolution", "mix": "Mixing", "add": "Addition"},
    "Reaction": {"heat": "Heating", "reflux": "Refluxing", "stir": "Stirring", "cool": "Cooling"},
    "Work-up": {"filter": "Filtration", "wash": "Washing", "dry": "Drying", "extract": "Extraction"},
    "Purification": {"recrystall": "Recrystallization", "column": "Chromatography"},
    "Analysis": {"nmr": "NMR", "ir": "IR", "ms": "Mass Spectrometry"}
}

# ---------------------------------------
# 🧠 NLP Helpers
# ---------------------------------------
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

def detect_substage(stage, sentence):
    sentence_lower = sentence.lower()
    if stage in substage_keywords:
        for k, v in substage_keywords[stage].items():
            if k in sentence_lower:
                return v
    return "General"

# ---------------------------------------
# 🔍 Reaction Parser
# ---------------------------------------
def parse_reaction(text):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        return None, None, None

    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).tolist()
        probs = F.softmax(outputs.logits, dim=1).tolist()

    results = []
    for i, (p, s) in enumerate(zip(preds, sentences)):
        stage = id2label[p]
        substage = detect_substage(stage, s)
        ents = extract_entities(s)
        conf = round(max(probs[i]) * 100, 2)
        results.append({
            "Step": i + 1,
            "Stage": stage,
            "Sub-Stage": substage,
            "Confidence (%)": conf,
            "Sentence": s,
            "Entities": ents
        })

    df = pd.DataFrame([{
        "Step": r["Step"],
        "Stage": r["Stage"],
        "Sub-Stage": r["Sub-Stage"],
        "Confidence (%)": r["Confidence (%)"],
        "Sentence": r["Sentence"],
        "Entities": str(r["Entities"])
    } for r in results])

    dot = Digraph(comment='Reaction Flow', format='svg')
    for r in results:
        label = f"Step {r['Step']}: {r['Stage']} → {r['Sub-Stage']}\\nConf: {r['Confidence (%)']}%"
        dot.node(str(r['Step']), label)
    for i in range(len(results) - 1):
        dot.edge(str(results[i]['Step']), str(results[i + 1]['Step']))
    graph_path = dot.render('reaction_flow', cleanup=True)
    return df, results, graph_path

# ---------------------------------------
# 🌈 UI
# ---------------------------------------
theme_choice = st.sidebar.radio("🌗 Theme", ["Dark", "Light"], index=0)
dark_mode = theme_choice == "Dark"

bg = "#0f172a" if dark_mode else "#f9fafb"
fg = "#f9fafb" if dark_mode else "#0f172a"
accent = "#38bdf8"

st.markdown(
    f"<h3 style='color:{fg};margin-top:1rem;'>🧪 Chemical Reaction Stage + Sub-Stage Parser</h3>",
    unsafe_allow_html=True
)

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

# ---------------------------------------
# 🧮 Run Parser
# ---------------------------------------
if st.button("Analyze Reaction"):
    if user_input.strip():
        with st.spinner("🔍 Parsing your procedure..."):
            df, results, graph_path = parse_reaction(user_input)
        if df is not None:
            st.success("✅ Parsing Complete!")

            st.subheader("📋 Parsed Steps")
            st.dataframe(df, use_container_width=True)

            st.download_button(
                "⬇️ Download CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="reaction_parsed.csv",
                mime="text/csv"
            )

            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(results, indent=4).encode('utf-8'),
                file_name="reaction_parsed.json",
                mime="application/json"
            )

            st.subheader("🔗 Reaction Flow Diagram")
            st.image(graph_path, caption="Reaction Flow Diagram", use_column_width=True)

            st.subheader("🧠 JSON Output")
            st.json(results)
        else:
            st.error("⚠️ No valid sentences found.")
    else:
        st.warning("Please provide a reaction procedure first!")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit, TinyBERT, and SpaCy.")
