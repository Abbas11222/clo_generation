"""
app.py  –  Streamlit UI for the CLO Generator + Skill Set Classifier
─────────────────────────────────────────────────────────────────────
Tab 1 : Generate CLOs   — Phi-3-mini (local text-generation pipeline)
Tab 2 : Classify Skills — facebook/bart-large-mnli (local zero-shot NLI)

The user types skills manually, one per line.
The model classifies each skill into:
  • Category     : Technical | Conceptual | Practical
  • Bloom's Level: Remember | Understand | Apply | Analyze | Evaluate | Create

Imports from:
  config.py          → MODEL_ID, CLASSIFIER_MODEL_ID, BLOOMS_VERBS, GENERATION_DEFAULTS
  extractor.py       → extract_from_docx / extract_from_pdf
  model_loader.py    → load_tokenizer, load_model, build_pipeline
  prompt_builder.py  → build_messages
  clo_generation.py  → generate_clos
  skill_classifier.py→ load_classifier, classify_skills, format_classified_skills_text
"""

import time
import threading
import streamlit as st

from config            import (MODEL_ID, CLASSIFIER_MODEL_ID, BLOOMS_VERBS,
                                GENERATION_DEFAULTS, DEFAULT_N_CLOS)
from extractor         import extract_from_docx, extract_from_pdf
from model_loader      import load_tokenizer, load_model, build_pipeline
from prompt_builder    import build_messages
from clo_generation    import generate_clos
from skill_classifier  import (load_classifier, extract_skill_set,
                                format_skill_set_text,
                                BLOOMS_COLORS, CATEGORY_COLORS)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CLO & Skill Classifier", page_icon="🎓", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background:#0f172a; }
  label, .stSelectbox label { color:#94a3b8 !important; }
  .stTextArea textarea { background:#1e293b; color:#e2e8f0; border-color:#334155; }

  .main-header {
    background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
    padding:2rem 2.5rem; border-radius:16px; margin-bottom:2rem; text-align:center;
  }
  .main-header h1 { color:#e2e8f0; font-size:2.2rem; margin:0; }
  .main-header p  { color:#94a3b8; margin:.4rem 0 0; font-size:1rem; }

  .card {
    background:#1e293b; border:1px solid #334155;
    border-radius:12px; padding:1.5rem; margin-bottom:1.5rem;
  }
  .card h3 { color:#7dd3fc; margin-top:0; }

  .extracted-box {
    background:#0f172a; border:1px solid #0ea5e9; border-radius:10px;
    padding:1.2rem; font-family:monospace; font-size:.85rem; color:#cbd5e1;
    white-space:pre-wrap; max-height:300px; overflow-y:auto;
  }
  .clo-box {
    background:#0f172a; border:1px solid #22c55e; border-radius:10px;
    padding:1.4rem; font-size:.95rem; color:#e2e8f0;
    white-space:pre-wrap; line-height:2;
  }
  .badge {
    display:inline-block; padding:.2rem .7rem; border-radius:99px;
    font-size:.75rem; font-weight:600; margin-right:.4rem;
  }
  .badge-blue  { background:#1d4ed8; color:#bfdbfe; }
  .badge-green { background:#15803d; color:#bbf7d0; }
  .badge-warn  { background:#92400e; color:#fde68a; }

  .stButton > button {
    width:100%; background:linear-gradient(135deg,#0ea5e9,#6366f1);
    color:white; border:none; border-radius:10px;
    padding:.75rem; font-size:1rem; font-weight:600; transition:opacity .2s;
  }
  .stButton > button:hover { opacity:.85; }

  /* ── Skill card ──────────────────────────────────────────────── */
  .skill-card {
    background:#0f172a; border-radius:10px; padding:.8rem 1.1rem;
    margin:.4rem 0; display:flex; align-items:flex-start; gap:.9rem;
  }
  .skill-name { color:#e2e8f0; font-weight:600; font-size:.95rem; }
  .skill-meta { color:#64748b; font-size:.78rem; margin-top:.15rem; }
  .pill {
    display:inline-block; padding:.18rem .65rem; border-radius:99px;
    font-size:.72rem; font-weight:700; white-space:nowrap;
  }

  /* ── Loading animation ───────────────────────────────────────── */
  .loader-wrap {
    background:#0f172a; border:1px solid #334155; border-radius:14px;
    padding:2.5rem 2rem; text-align:center;
  }
  .brain-anim {
    font-size:3.5rem; display:inline-block;
    animation:pulse 1.2s ease-in-out infinite;
  }
  @keyframes pulse {
    0%,100% { transform:scale(1);   opacity:1;   }
    50%      { transform:scale(1.2); opacity:0.6; }
  }
  .loader-title { color:#7dd3fc; font-size:1.2rem; font-weight:700; margin:.8rem 0 .3rem; }
  .loader-sub   { color:#64748b; font-size:.85rem; margin:0 0 1.2rem; }
  .progress-track {
    background:#1e293b; border-radius:99px; height:6px;
    overflow:hidden; width:100%; margin:.5rem 0 1.2rem;
  }
  .progress-bar {
    height:6px; border-radius:99px;
    background:linear-gradient(90deg,#0ea5e9,#6366f1,#22c55e);
    background-size:200% 100%; animation:slide 1.8s linear infinite;
  }
  @keyframes slide {
    0%   { background-position:200% 0; }
    100% { background-position:-200% 0; }
  }
  .step-row {
    display:flex; align-items:center; gap:.6rem;
    padding:.4rem .7rem; border-radius:8px; margin:.25rem 0; font-size:.85rem;
  }
  .step-done    { background:#0d2a1a; color:#4ade80; }
  .step-active  { background:#0c1f3a; color:#7dd3fc;
                  animation:glow .9s ease-in-out infinite alternate; }
  .step-pending { background:#1a1a2e; color:#475569; }
  @keyframes glow {
    from { box-shadow:none; }
    to   { box-shadow:0 0 8px #0ea5e940; }
  }
  .tip-box {
    background:#1e293b; border:1px solid #334155; border-radius:8px;
    padding:.8rem 1rem; margin-top:1.2rem;
    font-size:.8rem; color:#64748b; text-align:left; line-height:1.6;
  }
  .tip-box strong { color:#94a3b8; }
  .elapsed { color:#475569; font-size:.75rem; margin-top:.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🎓 CLO Generator &amp; Skill Classifier</h1>
  <p>Upload course content → generate CLOs · Write skills → local AI classifies them</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ CLO Settings")

    n_clos = st.slider("Number of CLOs", 1, 10, DEFAULT_N_CLOS, 1,
                       help="How many CLOs to generate")
    temperature = st.slider("Temperature", 0.05, 1.0,
                             GENERATION_DEFAULTS["temperature"], 0.05,
                             help="Lower = more precise. Higher = more creative.")
    max_tokens  = st.slider("Max output tokens", 100, 600,
                             GENERATION_DEFAULTS["max_new_tokens"], 50,
                             help="More tokens = longer CLOs.")

    st.markdown("---")
    st.markdown("**Bloom's Verbs for CLOs**")
    all_verbs = BLOOMS_VERBS + ["Evaluate", "Demonstrate", "Construct",
                                 "Compare", "Classify", "Develop"]
    selected_verbs = st.multiselect("verbs", options=all_verbs,
                                    default=BLOOMS_VERBS,
                                    label_visibility="collapsed")
    if not selected_verbs:
        st.warning("Select at least one verb.")
        selected_verbs = BLOOMS_VERBS

    st.markdown("---")
    st.markdown("## 🧠 Skill Classifier")
    st.caption(f"Model: `{CLASSIFIER_MODEL_ID}`")
    st.caption("Runs 100% locally — no internet needed at inference.")
    st.markdown("**Categories:** Technical · Conceptual · Practical")
    st.markdown("**Bloom's levels:** Remember → Create")

    st.markdown("---")
    st.caption(f"CLO model: `{MODEL_ID}`")


# ── Cached pipelines ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_clo_pipeline(model_id: str):
    tok   = load_tokenizer(model_id)
    model = load_model(model_id)
    return build_pipeline(model, tok)


@st.cache_resource(show_spinner=False)
def get_classifier(model_id: str):
    return load_classifier(model_id)



# ── Loading animation (CLO) ───────────────────────────────────────────────────
TIPS = [
    "💡 Bloom's verbs make CLOs measurable and assessable.",
    "📐 Good CLOs are specific — one concept, one action, one outcome.",
    "🔬 Lower temperature → more focused CLOs. Higher → more variety.",
    "🧠 BART-large-mnli uses NLI to classify skills — no fine-tuning needed.",
    "✅ Each CLO should be testable — if you can't assess it, rewrite it.",
    "🎯 Aim for verbs at different Bloom's levels for a balanced course.",
]

STEPS = [
    ("🔧", "Loading CLO model into memory"),
    ("🔍", "Analysing course content"),
    ("✍️", "Generating CLOs"),
    ("📋", "Formatting output"),
]


def render_loader(placeholder, n: int, step: int, elapsed: float, tip: str):
    rows = ""
    for i, (icon, label) in enumerate(STEPS):
        lbl = label if i != 2 else f"Generating {n} CLOs"
        if i < step:
            css, bullet = "step-done",    "✅"
        elif i == step:
            css, bullet = "step-active",  "⏳"
        else:
            css, bullet = "step-pending", "○"
        rows += (f'<div class="step-row {css}">'
                 f'<span style="width:1.4rem;text-align:center">{bullet}</span>'
                 f'{icon} {lbl}</div>')

    placeholder.markdown(f"""
<div class="loader-wrap">
  <div class="brain-anim">🧠</div>
  <div class="loader-title">AI is thinking …</div>
  <div class="loader-sub">Generating {n} Course Learning Outcomes</div>
  <div class="progress-track"><div class="progress-bar"></div></div>
  {rows}
  <div class="elapsed">⏱ {elapsed:.0f}s elapsed</div>
  <div class="tip-box"><strong>Did you know?</strong><br>{tip}</div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TWO-COLUMN LAYOUT
#   col_left  → upload document + paste/edit content
#   col_right → tabbed: [🎯 CLO Generator] | [🧠 Skill Set]
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 1], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# LEFT — Upload & Content
# ─────────────────────────────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="card"><h3>📁 Upload Course Document</h3>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose a .docx or .pdf file", type=["docx", "pdf"],
                                 label_visibility="collapsed")
    extracted_text = ""

    if uploaded:
        file_bytes = uploaded.read()
        ext        = uploaded.name.rsplit(".", 1)[-1].lower()

        with st.spinner("🔍 Extracting course content …"):
            try:
                if ext == "docx":
                    extracted_text, method = extract_from_docx(file_bytes)
                else:
                    extracted_text, method = extract_from_pdf(file_bytes)
            except Exception as e:
                st.error(f"Extraction error: {e}")
                method = "error"

        if extracted_text:
            badge = ("badge-blue"  if "non-table" in method else
                     "badge-green" if "table"     in method else "badge-warn")
            st.markdown(
                f'<span class="badge {badge}">{method}</span>'
                f'<span style="color:#94a3b8;font-size:.82rem"> '
                f'{len(extracted_text.split())} words</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="extracted-box">{extracted_text}</div>',
                        unsafe_allow_html=True)
        else:
            st.warning("⚠️ No course outline heading found. Paste content below.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>✏️ Course Content</h3>', unsafe_allow_html=True)
    st.caption("Extracted text appears here automatically — or paste/edit manually.")
    manual_text = st.text_area(
        "course_content_edit",
        value=extracted_text,
        height=260,
        label_visibility="collapsed",
        placeholder="Paste or edit your course outline here …",
    )
    st.markdown("</div>", unsafe_allow_html=True)

final_content = (manual_text or extracted_text).strip()

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT — Tabbed: CLO Generator | Skill Set
# ─────────────────────────────────────────────────────────────────────────────
with col_right:

    tab_clo, tab_skills = st.tabs(["🎯  CLO Generator", "🧠  Skill Set"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — CLO Generator
    # ══════════════════════════════════════════════════════════════════════════
    with tab_clo:
        st.markdown('<div class="card"><h3>🎯 Generate CLOs</h3>', unsafe_allow_html=True)

        if final_content:
            st.markdown(
                f'<div style="color:#64748b;font-size:.78rem;margin-bottom:.6rem">'
                f'Will generate <b style="color:#7dd3fc">{n_clos} CLOs</b> · '
                f'temp <b style="color:#7dd3fc">{temperature}</b> · '
                f'max tokens <b style="color:#7dd3fc">{max_tokens}</b> · '
                f'verbs: <b style="color:#7dd3fc">{", ".join(selected_verbs)}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Upload a document or paste content on the left to get started.")

        generate_clicked = st.button("✨ Generate CLOs",
                                      disabled=not bool(final_content),
                                      key="btn_clo")

        if generate_clicked and final_content:
            loader        = st.empty()
            tip           = TIPS[int(time.time()) % len(TIPS)]
            result_holder = {"value": None, "error": None}
            start         = time.time()

            render_loader(loader, n_clos, step=0, elapsed=0, tip=tip)
            try:
                pipe = get_clo_pipeline(MODEL_ID)
            except Exception as e:
                loader.empty()
                st.error(f"Model loading failed: {e}")
                st.stop()

            messages = build_messages(course_content=final_content,
                                      no_of_clos=n_clos,
                                      blooms_verbs=selected_verbs)
            gen_args = {**GENERATION_DEFAULTS,
                        "temperature"   : temperature,
                        "max_new_tokens": max_tokens}

            render_loader(loader, n_clos, step=1, elapsed=time.time()-start, tip=tip)
            time.sleep(0.5)

            def _run_clo():
                try:
                    result_holder["value"] = generate_clos(pipe, messages, gen_args)
                except Exception as e:
                    result_holder["error"] = str(e)

            thread = threading.Thread(target=_run_clo, daemon=True)
            thread.start()
            while thread.is_alive():
                render_loader(loader, n_clos, step=2,
                              elapsed=time.time()-start,
                              tip=TIPS[int(time.time()) % len(TIPS)])
                time.sleep(1)
            thread.join()

            if result_holder["error"]:
                loader.empty()
                st.error(f"Generation error: {result_holder['error']}")
                st.stop()

            render_loader(loader, n_clos, step=3, elapsed=time.time()-start, tip=tip)
            time.sleep(0.4)
            loader.empty()

            st.session_state["last_result"]  = result_holder["value"]
            st.session_state["last_n"]       = n_clos
            st.session_state["last_elapsed"] = time.time() - start

        if "last_result" in st.session_state:
            elapsed_str = f"{st.session_state['last_elapsed']:.1f}s"
            st.markdown(
                f'<div style="color:#4ade80;font-size:.82rem;margin-bottom:.5rem">'
                f'✅ {st.session_state["last_n"]} CLOs generated in {elapsed_str}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="clo-box">{st.session_state["last_result"]}</div>',
                unsafe_allow_html=True,
            )
            st.download_button("⬇️ Download CLOs (.txt)",
                               data=st.session_state["last_result"],
                               file_name="generated_clos.txt", mime="text/plain",
                               key="dl_clo")

        st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Skill Set
    # ══════════════════════════════════════════════════════════════════════════
    with tab_skills:
        st.markdown('<div class="card"><h3>🧠 Expected Skill Set</h3>', unsafe_allow_html=True)

        st.markdown(
            '<div style="color:#64748b;font-size:.78rem;margin-bottom:.9rem">'
            'After completing this course, students will have mastered '
            '<b style="color:#e2e8f0">3–5 broad skill areas</b> — '
            'each tagged with its Bloom\'s taxonomy level.'
            '</div>',
            unsafe_allow_html=True,
        )

        if not final_content:
            st.info("Upload a document or paste content on the left to get started.")

        skill_clicked = st.button("🎯 Extract Skill Set",
                                   disabled=not bool(final_content),
                                   key="btn_skills")

        if skill_clicked and final_content:
            with st.spinner("🔍 Classifying skill areas …"):
                try:
                    clf     = get_classifier(CLASSIFIER_MODEL_ID)
                    results = extract_skill_set(clf, final_content)
                    st.session_state["skill_results"] = results
                    st.session_state["skill_text"]    = format_skill_set_text(results)
                except Exception as e:
                    st.error(f"Skill extraction error: {e}")

        if "skill_results" in st.session_state:
            results = st.session_state["skill_results"]
            st.markdown(
                f'<div style="color:#4ade80;font-size:.82rem;margin-bottom:.8rem">'
                f'✅ {len(results)} skill areas identified</div>',
                unsafe_allow_html=True,
            )

            for r in results:
                cat_clr   = CATEGORY_COLORS.get(r["category"], "#94a3b8")
                bloom_clr = BLOOMS_COLORS.get(r["bloom"], "#94a3b8")
                st.markdown(
                    f'<div style="background:#0f172a;border-left:4px solid {cat_clr};'
                    f'border-radius:10px;padding:1rem 1.2rem;margin:.5rem 0">'
                    f'  <div style="color:#e2e8f0;font-weight:700;font-size:.92rem;margin-bottom:.4rem">'
                    f'    {r["skill"]}'
                    f'  </div>'
                    f'  <div>'
                    f'    <span style="background:{cat_clr}22;color:{cat_clr};'
                    f'    padding:.15rem .6rem;border-radius:99px;font-size:.72rem;font-weight:700">'
                    f'    {r["category"]}</span>'
                    f'    &nbsp;'
                    f'    <span style="background:{bloom_clr}22;color:{bloom_clr};'
                    f'    padding:.15rem .6rem;border-radius:99px;font-size:.72rem;font-weight:700">'
                    f'    Bloom\'s: {r["bloom"]}</span>'
                    f'    &nbsp;'
                    f'    <span style="color:#475569;font-size:.72rem">{r["score"]:.0%} confidence</span>'
                    f'  </div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download Skill Set (.txt)",
                data=st.session_state["skill_text"],
                file_name="skill_set.txt",
                mime="text/plain",
                key="dl_skills",
            )

        st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#334155;margin-top:2rem"/>
<p style="text-align:center;color:#475569;font-size:.8rem">
  SE-412L Deep Learning Lab · UET Mardan · Spring 2026
</p>
""", unsafe_allow_html=True)