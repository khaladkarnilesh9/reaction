Absolutely ✅ — here’s a **professional `README.md`** for your Streamlit-based **Chemical Reaction Stage Parser** app.
It’s formatted for GitHub and includes setup instructions, screenshots placeholders, and feature highlights.

---

## 🧪 Chemical Reaction Stage & Sub-Stage Parser

An intelligent **Streamlit web app** that parses synthetic or experimental **chemical procedures**, identifies **stages and sub-stages**, extracts **key entities** (amounts, temperature, time, etc.), and visualizes the **reaction flow** as a diagram.

Built with **Transformers (BERT)**, **SpaCy**, and **Graphviz**, the app brings chemistry text understanding to life in an interactive, elegant interface.

---

### 🚀 Features

| Feature                          | Description                                                      |
| -------------------------------- | ---------------------------------------------------------------- |
| 🧭 **Stylish Navbar**            | Sticky top navigation bar with smooth hover animations           |
| 🌗 **Light/Dark Mode**           | Toggle between professional dark or light themes                 |
| 🧪 **Text & File Input**         | Enter procedures manually or upload `.txt` / `.csv` files        |
| 🧠 **BERT-based Classification** | Classifies each sentence into chemical stages and sub-stages     |
| 🧩 **Entity Extraction**         | Automatically detects temperature, quantities, and durations     |
| 🔗 **Flow Diagram**              | Generates a visual reaction workflow using Graphviz              |
| 💾 **Data Export**               | Download structured results as CSV or JSON                       |
| 📜 **Clean Layout**              | Intuitive interface built entirely with Streamlit and custom CSS |

---

### 🖥️ Demo Preview

*(Add screenshots here after running your app)*

| Parsed Steps                              | Reaction Flow                             |
| ----------------------------------------- | ----------------------------------------- |
| ![Steps Table](docs/screenshot_table.png) | ![Flow Diagram](docs/screenshot_flow.png) |

---

### ⚙️ Installation

#### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/chemical-reaction-parser.git
cd chemical-reaction-parser
```

#### 2️⃣ Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # (Mac/Linux)
venv\Scripts\activate      # (Windows)
```

#### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

#### 4️⃣ Download SpaCy English model

```bash
python -m spacy download en_core_web_sm
```

#### 5️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

---

### 📁 Project Structure

```
chemical-reaction-parser/
│
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # Documentation (this file)
├── reaction_flow.png      # Auto-generated flow diagram (runtime)
└── docs/                  # (Optional) Store screenshots here
```

---

### 🧠 How It Works

1. **Sentence Segmentation**
   The input procedure is split into logical sentences.

2. **Stage Classification**
   Each sentence is classified into one of:
   `Preparation`, `Reaction`, `Work-up`, `Purification`, `Analysis`, or `Other`.

3. **Sub-Stage Detection**
   The app detects sub-stages based on keyword mappings (e.g., *Weighing*, *Filtration*, *Cooling*).

4. **Entity Extraction**
   SpaCy & regex identify temperatures, quantities, durations, and chemical entities.

5. **Visualization**
   Graphviz builds a connected flowchart of reaction steps, rendered as a PNG diagram.

---

### 📊 Example Input

```
Weigh 2 g of NaCl.
Dissolve NaCl in 50 mL of water.
Add ethanol and stir for 2 hours at 60°C.
Filter the solid and dry under vacuum.
Analyze the product by NMR spectroscopy.
```

### 📋 Example Output

| Step | Stage       | Sub-Stage   | Entities                                           |
| ---- | ----------- | ----------- | -------------------------------------------------- |
| 1    | Preparation | Weighing    | {"Amount": ["2 g"]}                                |
| 2    | Preparation | Dissolution | {"Amount": ["50 mL"]}                              |
| 3    | Reaction    | Stirring    | {"Temperature": ["60°C"], "Duration": ["2 hours"]} |
| 4    | Work-up     | Filtration  | {}                                                 |
| 5    | Analysis    | NMR         | {}                                                 |

---

### 🧭 App Sections

* **Home:** Enter or upload reaction procedures
* **Upload:** Upload `.txt` or `.csv` files
* **Docs:** Placeholder for user instructions or API docs

---

### 🧰 Technologies Used

* [Streamlit](https://streamlit.io/) – interactive web app framework
* [Transformers (Hugging Face)](https://huggingface.co/) – BERT-based text classification
* [SpaCy](https://spacy.io/) – entity extraction
* [Graphviz](https://graphviz.org/) – flow diagram generation
* [Pandas](https://pandas.pydata.org/) – data manipulation

---

### 🧑‍🔬 Future Enhancements

* 🧾 Add **session history** (save & reload previous analyses)
* 🧬 Integrate with **PubChem API** for compound recognition
* 📈 Add analytics dashboard for reaction step distribution

---

### 🩵 Acknowledgements

Developed with ❤️ by [Your Name or Lab Name]
Powered by **BERT**, **SpaCy**, and **Streamlit**

---

### 📄 License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---

Would you like me to include a ready-made `LICENSE` (MIT) and a `.gitignore` optimized for Python + Streamlit projects as well?
That would make your repository fully GitHub-ready.
