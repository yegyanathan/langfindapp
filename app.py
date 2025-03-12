import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from gcld3 import NNetLanguageIdentifier

MIN_BYTES = 0
MAX_BYTES = 1000
PROPORTION_THRESHOLD = 0.10

detector = NNetLanguageIdentifier(min_num_bytes=MIN_BYTES, max_num_bytes=MAX_BYTES)


lang_dict = {
    "af": ("Afrikaans", "Latin"),
    "am": ("Amharic", "Ethiopic"),
    "ar": ("Arabic", "Arabic"),
    "bg": ("Bulgarian", "Cyrillic"),
    "bg-Latn": ("Bulgarian", "Latin"),
    "bn": ("Bangla", "Bangla"),
    "bs": ("Bosnian", "Latin"),
    "ca": ("Catalan", "Latin"),
    "ceb": ("Cebuano", "Latin"),
    "co": ("Corsican", "Latin"),
    "cs": ("Czech", "Latin"),
    "cy": ("Welsh", "Latin"),
    "da": ("Danish", "Latin"),
    "de": ("German", "Latin"),
    "el": ("Greek", "Greek"),
    "el-Latn": ("Greek", "Latin"),
    "en": ("English", "Latin"),
    "eo": ("Esperanto", "Latin"),
    "es": ("Spanish", "Latin"),
    "et": ("Estonian", "Latin"),
    "eu": ("Basque", "Latin"),
    "fa": ("Persian", "Arabic"),
    "fi": ("Finnish", "Latin"),
    "fil": ("Filipino", "Latin"),
    "fr": ("French", "Latin"),
    "fy": ("Western Frisian", "Latin"),
    "ga": ("Irish", "Latin"),
    "gd": ("Scottish Gaelic", "Latin"),
    "gl": ("Galician", "Latin"),
    "gu": ("Gujarati", "Gujarati"),
    "ha": ("Hausa", "Latin"),
    "haw": ("Hawaiian", "Latin"),
    "hi": ("Hindi", "Devanagari"),
    "hi-Latn": ("Hindi", "Latin"),
    "hmn": ("Hmong", "Latin"),
    "hr": ("Croatian", "Latin"),
    "ht": ("Haitian Creole", "Latin"),
    "hu": ("Hungarian", "Latin"),
    "hy": ("Armenian", "Armenian"),
    "id": ("Indonesian", "Latin"),
    "ig": ("Igbo", "Latin"),
    "is": ("Icelandic", "Latin"),
    "it": ("Italian", "Latin"),
    "iw": ("Hebrew", "Hebrew"),
    "ja": ("Japanese", "Japanese"),
    "ja-Latn": ("Japanese", "Latin"),
    "jv": ("Javanese", "Latin"),
    "ka": ("Georgian", "Georgian"),
    "kk": ("Kazakh", "Cyrillic"),
    "km": ("Khmer", "Khmer"),
    "kn": ("Kannada", "Kannada"),
    "ko": ("Korean", "Korean"),
    "ku": ("Kurdish", "Latin"),
    "ky": ("Kyrgyz", "Cyrillic"),
    "la": ("Latin", "Latin"),
    "lb": ("Luxembourgish", "Latin"),
    "lo": ("Lao", "Lao"),
    "lt": ("Lithuanian", "Latin"),
    "lv": ("Latvian", "Latin"),
    "mg": ("Malagasy", "Latin"),
    "mi": ("Maori", "Latin"),
    "mk": ("Macedonian", "Cyrillic"),
    "ml": ("Malayalam", "Malayalam"),
    "mn": ("Mongolian", "Cyrillic"),
    "mr": ("Marathi", "Devanagari"),
    "ms": ("Malay", "Latin"),
    "mt": ("Maltese", "Latin"),
    "my": ("Burmese", "Myanmar"),
    "ne": ("Nepali", "Devanagari"),
    "nl": ("Dutch", "Latin"),
    "no": ("Norwegian", "Latin"),
    "ny": ("Nyanja", "Latin"),
    "pa": ("Punjabi", "Gurmukhi"),
    "pl": ("Polish", "Latin"),
    "ps": ("Pashto", "Arabic"),
    "pt": ("Portuguese", "Latin"),
    "ro": ("Romanian", "Latin"),
    "ru": ("Russian", "Cyrillic"),
    "ru-Latn": ("Russian", "English"),
    "sd": ("Sindhi", "Arabic"),
    "si": ("Sinhala", "Sinhala"),
    "sk": ("Slovak", "Latin"),
    "sl": ("Slovenian", "Latin"),
    "sm": ("Samoan", "Latin"),
    "sn": ("Shona", "Latin"),
    "so": ("Somali", "Latin"),
    "sq": ("Albanian", "Latin"),
    "sr": ("Serbian", "Cyrillic"),
    "st": ("Southern Sotho", "Latin"),
    "su": ("Sundanese", "Latin"),
    "sv": ("Swedish", "Latin"),
    "sw": ("Swahili", "Latin"),
    "ta": ("Tamil", "Tamil"),
    "te": ("Telugu", "Telugu"),
    "tg": ("Tajik", "Cyrillic"),
    "th": ("Thai", "Thai"),
    "tr": ("Turkish", "Latin"),
    "uk": ("Ukrainian", "Cyrillic"),
    "ur": ("Urdu", "Arabic"),
    "uz": ("Uzbek", "Latin"),
    "vi": ("Vietnamese", "Latin"),
    "xh": ("Xhosa", "Latin"),
    "yi": ("Yiddish", "Hebrew"),
    "yo": ("Yoruba", "Latin"),
    "zh": ("Chinese", "Han"),
    "zh-Latn": ("Chinese", "Latin"),
    "zu": ("Zulu", "Latin"),
    "und": ("Undefined", "Undefined")
}

st.title("📝 Language Detection App")
st.write("Upload a `.txt` file to detect languages.")

# Reliability Filter
reliable_only = st.checkbox("Show only reliable detections", value=True)

# File Upload
uploaded_file = st.file_uploader("Choose a text file", type=["txt"])

if uploaded_file:
    with st.spinner("Detecting languages..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        assert uploaded_file.name.lower().endswith(".txt"), "Only .txt files are allowed"

        text = uploaded_file.getvalue().decode("utf-8")
        assert text, "Uploaded file is empty"

        results = detector.FindTopNMostFreqLangs(text=text, num_langs=1000)
        results = [
            {
                "language": result.language,
                "probability": result.probability,
                "is_reliable": result.is_reliable,
                "proportion": result.proportion
            }
            for result in results if result.proportion >= PROPORTION_THRESHOLD
        ]
        
        # Apply reliability filter if selected
        if reliable_only:
            results = [res for res in results if res.get("is_reliable", False)]

        if not results:
            st.warning("No significant languages detected.")
        else:
            st.subheader("Languages")
            fig, ax = plt.subplots(figsize=(6, 1))
            languages = [f'{lang_dict[res["language"]][0]} {res["proportion"]:.1%}'  for res in results]
            proportions = [res["proportion"] for res in results]
            colors = plt.cm.Paired(np.linspace(0, 1, len(languages)))
            
            left = np.cumsum([0] + proportions[:-1])
            for i, (lang, width) in enumerate(zip(languages, proportions)):
                ax.barh(0, width, left=left[i], color=colors[i], label=lang)
            
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
            st.pyplot(fig)



