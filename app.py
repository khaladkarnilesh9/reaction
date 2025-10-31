import streamlit as st
import pandas as pd
import re
from PIL import Image, ImageDraw, ImageFont
import io

# -------------------------------
# 1️⃣ Stage keywords
# -------------------------------
stage_keywords = {
    "Preparation": ["weigh", "dissolve", "mix", "add"],
    "Reaction": ["heat", "reflux", "stir", "cool", "react"],
    "Work-up": ["filter", "wash", "dry", "extract"],
    "Purification": ["recrystall", "column", "purify"],
    "Analysis": ["nmr", "ir", "analy", "test"],
}

substage_keywords = {
    "Preparation": {"weigh": "Weighing", "dissolve": "Dissolution", "add": "Addition"},
    "Reaction": {"stir": "Stirring", "heat": "Heating", "cool": "Cooling"},
    "Work-up": {"filter": "Filtration", "dry": "Drying"},
    "Purification": {"recrystall": "Recrystallization"},
    "Analysis": {"nmr": "NMR", "ir": "IR"},
}

# -------------------------------
# 2️⃣ Entity extraction (regex)
# -------------------------------
def extract_entities(sentence):
    entities = {}
    temp = re.findall(r"\d+\s?°C", sentence)
    amt = re.findall(r"\d+(?:\.\d+)?\s?(?:g|mg|mL|L|mol)", sentence)
    time = re.findall(r"\d+\s?(?:h|hour|hours|min|minutes)", sentence)
    if temp: entities["Temperature"] = temp
    if amt: entities["Amount"] = amt
    if time: entities["Duration"] = time
    return entities

# -------------------------------
# 3️⃣ Rule-based classification
# -------------------------------
def detect_stage(sentence):
    sentence_lower = sentence.lower()
    for stage, keys in stage_keywords.items():
        if any(k in sentence_lower for k in keys):
            return stage
    return "Other"

def detect_substage(stage, sentence):
    sentence_lower = sentence.lower()
    if stage in substage_keywords:
        for k, v in substage_keywords[stage].items():
            if k in sentence_lower:
                return v
    return "General"

# -------------------------------
# 4️⃣ Main parsing logic
# -------------------------------
def parse_reaction(text):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    results = []

    for i, s in enumerate(sentences, 1):
        stage = detect_stage(s)
        substage = detect_substage(stage, s)
        ents = extract_entities(s)
        results.append({
            "Step": i,
            "Stage": stage,
            "Sub-Stage": substage,
            "Sentence": s,
            "Entities": str(ents)
        })

    df = pd.DataFrame(results)
    return df

# -------------------------------
# 5️⃣ Flow diagram (image-based)
# -------------------------------
def create_flow_image(df):
    width, height = 800, 100 + 80 * len(df)
    img = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)

    y = 50
    for _, row in df.iterrows():
        text = f"Step {row['Step']}: {row['Stage']} → {row['Sub-Stage']}"
        draw.rectangle([(50, y), (750, y + 50)], fill=(220, 240, 255), outline=(100, 100, 100))
        draw.text((60, y + 15), text, fill=(0, 0, 0))
        y += 80

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer

# -------------------------------
# 6️⃣ Streamlit UI
# -------------------------------
st.set_page_config(page_title="Light Reaction Parser", layout="wide")
st.title("🧪 Lightweight Reaction Parser (No Graphviz)")

user_input = st.text_area("✍️ Enter Reaction Procedure", height=200)

if st.button("Analyze"):
    if user_input.strip():
        df = parse_reaction(user_input)
        st.success("✅ Parsed Successfully!")

        st.subheader("📋 Parsed Steps")
        st.dataframe(df, use_container_width=True)

        st.subheader("🔗 Reaction Flow Diagram")
        flow_image = create_flow_image(df)
        st.image(flow_image, caption="Reaction Flow Diagram", use_column_width=True)

        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"),
                           "reaction_parsed.csv", "text/csv")
    else:
        st.warning("Please enter some text first!")

st.caption("⚡ Offline, lightweight, and mobile-data friendly!")
