"""
_fix_paper.py
Apply all corrections to generate_paper.py:
  1. Replace em dashes in prose with clean alternatives
  2. Fix m̄ combining-macron symbol → plain text
  3. Fix ⟨E⟩ angle brackets → <E>
  4. Fix ⌊⌋ floor brackets → floor()
  5. Rewrite sec_title_abstract with full academic metadata
  6. Add sec_acknowledgments function
  7. Update main() to include acknowledgments
"""
from pathlib import Path

SRC = Path(__file__).parent / "generate_paper.py"
text = SRC.read_text(encoding="utf-8")

# ── 1.  m̄  (m + combining macron, U+006D U+0304) → plain m_r ─────────────
text = text.replace("m̄", "m_r")          # 'r' for 'reduced', i.e. m mod S

# ── 2.  ⟨E⟩  mathematical angle brackets → <E> (HTML will escape these) ────
text = text.replace("⟨E⟩", "mean(E)")

# ── 3.  ⌊ ⌋  floor brackets → floor( ) ─────────────────────────────────────
text = text.replace("⌊", "floor(")
text = text.replace("⌋", ")")

# ── 4.  ∂  partial derivative sign → 'd' in fraction notation ───────────────
text = text.replace("|∂E/∂h|", "|dE/dh|")
text = text.replace("|∂E/∂d|", "|dE/dd|")

# ── 5.  Em dash replacements (context-specific, ordered longest→ shortest) ──

# Abstract: periodicity class — quasi-periodic … periodic-factor —
text = text.replace(
    "the operator's periodicity class — quasi-periodic "
    "(m coprime with S), periodic-resonant, or periodic-factor — inducing",
    "the operator's periodicity class (quasi-periodic when m is coprime with S, "
    "periodic-resonant, or periodic-factor), inducing",
)

# Intro: "students — the field relies" pattern
text = text.replace(
    "For realistic problem sizes — hundreds of exams and thousands "
    "of students — the field relies",
    "For realistic problem sizes (hundreds of exams and thousands "
    "of students), the field relies",
)

# Intro: cyclic shift — induces
text = text.replace(
    "The operator governing the motion — a cyclic shift applied when a particle (exam) "
    "is in conflict with a neighbor — induces dynamics",
    "The operator governing the motion, a cyclic shift applied when a particle (exam) "
    "is in conflict with a neighbor, induces dynamics",
)

# Intro: Hebbian rule — weights … — encodes
text = text.replace(
    "The Hebbian learning rule [11] — weights proportional to the outer product of stored patterns "
    "— encodes timetable configurations as attractors",
    "The Hebbian learning rule [11], with weights proportional to the outer product of stored patterns, "
    "encodes timetable configurations as attractors",
)

# Intro: "a condition called a clash"
text = text.replace(
    "such that no two exams share a slot if they have common enrolled students — a condition called",
    "such that no two exams share a slot if they have common enrolled students, a condition called",
)

# Literature: Toronto benchmark suite — twelve … — which
text = text.replace(
    "introduced the Toronto benchmark suite — twelve real-world "
    "exam scheduling instances from North American universities — which has since become",
    "introduced the Toronto benchmark suite (twelve real-world "
    "exam scheduling instances from North American universities), which has since become",
)

# Literature: "graph coloring — and by extension timetabling —"
text = text.replace(
    "placed graph coloring — and by extension timetabling — "
    "firmly in the class of canonical NP-hard problems.",
    "placed graph coloring, and by extension timetabling, "
    "firmly in the class of canonical NP-hard problems.",
)

# Literature: "storage capacity of a Hopfield network — the maximum number …  — was"
text = text.replace(
    "The storage capacity of a Hopfield network — the maximum number of patterns "
    "that can be reliably recalled — was analytically bounded by",
    "The storage capacity of a Hopfield network, defined as the maximum number of patterns "
    "that can be reliably recalled, was analytically bounded by",
)

# Problem def: "paper-minimum timeslots — the values S"
text = text.replace(
    "We call these <b>paper-minimum timeslots</b> — the values S for which the",
    "We call these <b>paper-minimum timeslots</b>: the values S for which the",
)

# Evaluation metrics list items with —
text = text.replace(
    "(i) <b>final clash count</b> — the clash count at the last event; ",
    "(i) <b>final clash count</b>: the clash count at the last event; ",
)
text = text.replace(
    "(ii) <b>zero-clash rate</b> — the fraction of runs that achieve C(σ) = 0; ",
    "(ii) <b>zero-clash rate</b>: the fraction of runs that achieve C(σ) = 0; ",
)
text = text.replace(
    "(iii) <b>first-zero event</b> — the event index at which zero clashes are first reached; ",
    "(iii) <b>first-zero event</b>: the event index at which zero clashes are first reached; ",
)
text = text.replace(
    "(iv) <b>first-minimum event</b> — the event index at which the run's minimum is first reached; ",
    "(iv) <b>first-minimum event</b>: the event index at which the run's minimum is first reached; ",
)
text = text.replace(
    "(v) <b>mean rolling Pearson r</b> — the average over defined windows of the",
    "(v) <b>mean rolling Pearson r</b>: the average over defined windows of the",
)

# Ordering strategies list items
text = text.replace(
    "(i) <b>natural</b> — exams processed in index order 1, …, N; ",
    "(i) <b>natural</b>: exams processed in index order 1, …, N; ",
)
text = text.replace(
    "(ii) <b>random</b> — a fresh random permutation each step; ",
    "(ii) <b>random</b>: a fresh random permutation each step; ",
)
text = text.replace(
    "(iii) <b>largest-weighted-degree</b> — exams sorted by Σ_{j~k} w_{kj} in decreasing order, ",
    "(iii) <b>largest-weighted-degree</b>: exams sorted by sum_j w_{kj} in decreasing order, ",
)
text = text.replace(
    "(iv) <b>saturation-like</b> — exams sorted by 2×degree + weighted_degree. ",
    "(iv) <b>saturation-like</b>: exams sorted by 2 x degree + weighted_degree. ",
)

# Section 7 content comments line
text = text.replace(
    "# Content builders — one function per section",
    "# Content builders - one function per section",
)

# rolling correlation "uncorrelated — evidence"
text = text.replace(
    "the two trajectories' local fluctuations are statistically uncorrelated — "
    "evidence that the methods are exploring different regions of the energy landscape.",
    "the two trajectories' local fluctuations are statistically uncorrelated, "
    "which is evidence that the methods are exploring different regions of the energy landscape.",
)

# directory listing dashes
text = text.replace(
    "models/           — Instance, Course, Autoassociator dataclasses<br/>",
    "models/            Instance, Course, Autoassociator dataclasses<br/>",
)
text = text.replace(
    "scripts/          — experiment drivers and visualization scripts<br/>",
    "scripts/           experiment drivers and visualization scripts<br/>",
)
text = text.replace(
    "data/             — Toronto benchmark .crs / .stu files<br/>",
    "data/              Toronto benchmark .crs / .stu files<br/>",
)
text = text.replace(
    "outputs/          — all generated CSV, JSON, .npy, PNG, HTML results<br/>",
    "outputs/           all generated CSV, JSON, .npy, PNG, HTML results<br/>",
)
text = text.replace(
    "sensitivity_figs/ — sensitivity sweep outputs<br/>",
    "sensitivity_figs/  sensitivity sweep outputs<br/>",
)
text = text.replace(
    'figs/             — main experiment visualizations"',
    'figs/              main experiment visualizations"',
)

# Sensitivity results: "different final timetable structures — the discrete-state analog"
text = text.replace(
    "different final timetable structures — the discrete-state analog of chaotic sensitivity.",
    "different final timetable structures, which is the discrete-state analog of chaotic sensitivity.",
)

# Results overview: "all instances — suggesting that"
text = text.replace(
    "the latest first-minimum event of all instances — suggesting that this large, ",
    "the latest first-minimum event of all instances, suggesting that this large, ",
)

# ear-f-83: "local energy landscape — when"
text = text.replace(
    "in opposite directions in the local energy landscape — "
    "when dynamic-only's clash count rises, hybrid's tends to fall.",
    "in opposite directions in the local energy landscape: "
    "when dynamic-only's clash count rises, hybrid's tends to fall.",
)

# hec-s-92: "Both methods achieve min_clash = 2 — neither"
text = text.replace(
    "Both methods achieve min_clash = 2 — neither reaches zero.",
    "Both methods achieve min_clash = 2; neither reaches zero.",
)

# kfu-s-93: "at event t = 3 — essentially"
text = text.replace(
    "reaches its minimum at event t = 3 — "
    "essentially immediately after initialization.",
    "reaches its minimum at event t = 3, "
    "essentially immediately after initialization.",
)

# lse-f-91: "vs hybrid 14 — one of only two"
text = text.replace(
    "Dynamic achieves 12 clashes vs hybrid 14 — one of only two instances where",
    "Dynamic achieves 12 clashes vs hybrid 14, making this one of only two instances where",
)

# rye-s-93: "(32.9% of budget remaining) — the only instance"
text = text.replace(
    "(32.9% of budget remaining) — the only instance where meaningful improvement",
    "(32.9% of budget remaining), the only instance where meaningful improvement",
)

# sta-f-83: "descent paths — there is little"
text = text.replace(
    "follow almost identical descent paths — there is little divergence possible",
    "follow almost identical descent paths, with little divergence possible",
)

# Discussion: "behavior — mode-dependent … limit-cycle attractors — maps"
text = text.replace(
    "behavior — mode-dependent orbit structure, ergodic vs. integrable dynamics, "
    "limit-cycle attractors — maps naturally onto concepts",
    "behavior (mode-dependent orbit structure, ergodic or integrable dynamics, "
    "and limit-cycle attractors) maps naturally onto concepts",
)

# Discussion: "limit cycles — which repeats … in hec-s-92 — reflects"
text = text.replace(
    "The autocorrelation structure of these limit cycles — "
    "which repeats approximately every 18 events in hec-s-92 — reflects",
    "The autocorrelation structure of these limit cycles, "
    "which repeats approximately every 18 events in hec-s-92, reflects",
)

# Discussion: "first few hundred events — when both"
text = text.replace(
    "in the first few hundred events — when both methods start from the same ",
    "in the first few hundred events, when both methods start from the same ",
)

# Discussion: "unaffected — their standard and minimum"
text = text.replace(
    "Some instances (sta-f-83, ute-s-92) are unaffected — "
    "their standard and minimum slot counts are equal,",
    "Some instances (sta-f-83, ute-s-92) are unaffected, as their standard and minimum "
    "slot counts are equal,",
)

# Future work: "chaotic regimes — as a function … and m — would"
text = text.replace(
    '"integrable and chaotic regimes — as a function of N, S, conflict density, "\n'
    '         "and m — would provide principled guidance for parameter selection.")',
    '"integrable and chaotic regimes, as a function of N, S, conflict density, "\n'
    '         "and m, would provide principled guidance for parameter selection.")',
)

# Future work: "a tuning phase — even a short grid search — before"
text = text.replace(
    '"suggests that a tuning phase — even a short grid search — "\n'
    '         "before the main run would provide substantial improvement over "',
    '"suggests that a tuning phase (even a short grid search) "\n'
    '         "before the main run would provide substantial improvement over "',
)

# caption: "f'{inst} — extended dynamics'" → colon
text = text.replace(
    "f\"{inst} — extended dynamics\"",
    "f\"{inst}: extended dynamics\"",
)
text = text.replace(
    "f\"{inst} — rolling r\"",
    "f\"{inst}: rolling r\"",
)

# header separator comment
text = text.replace(
    "# ─────────────────────────────────────────────────────────────────────────────",
    "# ---------------------------------------------------------------------------",
)

# Section 7 header comment
text = text.replace(
    "# Section 7: Results – the big one",
    "# Section 7: Results",
)

# Sensitivity header in table: "⟨E⟩" - already handled above but headers_sens
text = text.replace(
    '["Instance", "N", "S (slots)", "Verdict", "S-score", "⟨E⟩", "d_H/N"]',
    '["Instance", "N", "S (slots)", "Verdict", "S-score", "Mean E", "d_H/N"]',
)
text = text.replace(
    '"S-score = normalized sensitivity index; ⟨E⟩ = mean clash over (m,d) grid; "',
    '"S-score = normalized sensitivity index; Mean E = mean clash over (m,d) grid; "',
)
text = text.replace(
    "achieve ⟨E⟩ = 0: the energy landscape",
    "achieve mean clash = 0: the energy landscape",
)

# sensitivity formula: "S = max(|∂E/∂h|, |∂E/∂d|) / ⟨E⟩,"
text = text.replace(
    "S = max(|∂E/∂h|, |∂E/∂d|) / ⟨E⟩, ",
    "S = max(|dE/dh|, |dE/dd|) / mean(E), ",
)

# mode table: m_r entries (done above via m̄ → m_r)
# clean up any residual combining chars
import unicodedata
# Remove all combining characters except in f-strings with {
cleaned = []
for ch in text:
    cat = unicodedata.category(ch)
    if cat.startswith('M'):   # combining marks
        continue
    cleaned.append(ch)
text = "".join(cleaned)

# ── 6.  Rewrite sec_title_abstract ──────────────────────────────────────────
OLD_TITLE = '''def sec_title_abstract(styles) -> list:
    s = styles
    story = []
    story.append(sp(1.5 * cm))
    story.append(P(
        "Hybrid Propagating-Particle and Hopfield Network Dynamics<br/>"
        "for Exam Timetabling:<br/>"
        "Cyclic Behavior, Convergence Analysis,<br/>"
        "and Benchmark Evaluation",
        s["title"],
    ))
    story.append(sp(4))
    story.append(P("Arpine Tadevosyan", s["author"]))
    story.append(P("Senior Capstone Project in Computer Science", s["subtitle"]))
    story.append(P("Department of Computer Science and Information Technologies", s["subtitle"]))
    story.append(P("American University of Armenia", s["subtitle"]))
    story.append(sp(6))
    story.append(hr())
    story.append(sp(8))'''

NEW_TITLE = '''def sec_title_abstract(styles) -> list:
    s = styles
    story = []
    story.append(sp(1.8 * cm))
    story.append(P(
        "Hybrid Propagating-Particle and Hopfield Network Dynamics<br/>"
        "for Exam Timetabling:<br/>"
        "Cyclic Behavior, Convergence Analysis,<br/>"
        "and Benchmark Evaluation",
        s["title"],
    ))
    story.append(sp(14))
    story.append(hr())
    story.append(sp(10))

    # Academic metadata block
    meta_label = ParagraphStyle(
        "MetaLabel",
        fontSize=10, leading=14, fontName="Times-Bold",
        alignment=TA_LEFT, textColor=colors.black,
    )
    meta_value = ParagraphStyle(
        "MetaValue",
        fontSize=10, leading=14, fontName="Times-Roman",
        alignment=TA_LEFT, textColor=colors.black,
    )
    meta_rows = [
        ("Student:", "Arpine Tadevosyan"),
        ("Program:", "Bachelor of Science in Computer Science (BS in CS)"),
        ("Institution:", "American University of Armenia (AUA)"),
        ("Department:", "Computer Science and Information Technologies"),
        ("Academic Year:", "2025-2026"),
        ("Capstone Chair:", "Hayk Nersisyan"),
        ("Supervisors:", "Suren Khachatryan"),
        ("", "Aleksandr Hayrapetyan"),
        ("Date:", "May 2026"),
    ]
    meta_table_data = [
        [P(lbl, meta_label), P(val, meta_value)]
        for lbl, val in meta_rows
    ]
    meta_table = Table(
        meta_table_data,
        colWidths=[3.8 * cm, TEXT_W - 3.8 * cm],
    )
    meta_table.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("LINEABOVE",    (0, 0), (-1, 0),  0.3, colors.HexColor("#888888")),
        ("LINEBELOW",    (0, -1),(-1, -1), 0.3, colors.HexColor("#888888")),
    ]))
    story.append(meta_table)
    story.append(sp(10))
    story.append(hr())
    story.append(sp(10))'''

assert OLD_TITLE in text, "OLD_TITLE block not found — check whitespace"
text = text.replace(OLD_TITLE, NEW_TITLE)

# ── 7.  Add sec_acknowledgments before sec_references ───────────────────────
ACK_FN = '''
def sec_acknowledgments(styles) -> list:
    s = styles
    story = []
    story.append(H1("Acknowledgments", s))
    story.append(B(
        "The author wishes to express sincere gratitude to the individuals and institution "
        "whose guidance and support made this work possible.",
        s,
    ))
    story.append(B(
        "Special thanks are due to Suren Khachatryan and Aleksandr Hayrapetyan, "
        "whose technical expertise, thoughtful feedback, and consistent encouragement "
        "shaped every stage of this project. "
        "Their insights into the algorithmic and mathematical foundations of the work "
        "were invaluable, and their willingness to engage with questions at any stage "
        "of the research process is deeply appreciated.",
        s,
    ))
    story.append(B(
        "The author is also grateful to Hayk Nersisyan for serving as Capstone Chair "
        "and for providing academic oversight throughout the capstone process.",
        s,
    ))
    story.append(B(
        "Finally, the author thanks the American University of Armenia for providing "
        "the academic environment, computational resources, and intellectual community "
        "that made this capstone project possible. "
        "The Department of Computer Science and Information Technologies has been a "
        "supportive and stimulating home for this research.",
        s,
    ))
    return story


'''

# Insert before def sec_references
assert "def sec_references(styles)" in text, "sec_references not found"
text = text.replace("def sec_references(styles)", ACK_FN + "def sec_references(styles)")

# ── 8.  Update main() to include acknowledgments ────────────────────────────
text = text.replace(
    "    story += sec_conclusion(styles)\n"
    "    story += sec_future_work(styles)\n"
    "    story.append(PageBreak())\n"
    "    story += sec_references(styles)",

    "    story += sec_conclusion(styles)\n"
    "    story += sec_future_work(styles)\n"
    "    story.append(PageBreak())\n"
    "    story += sec_acknowledgments(styles)\n"
    "    story.append(PageBreak())\n"
    "    story += sec_references(styles)",
)

# ── 9.  Update section cross-references in the intro ────────────────────────
text = text.replace(
    '"Sections 8 through 11 provide discussion, conclusion, future work, and references."',
    '"Sections 8 through 12 provide discussion, conclusion, future work, acknowledgments, and references."',
)

# Write updated file
SRC.write_text(text, encoding="utf-8")
print("generate_paper.py patched successfully.")
