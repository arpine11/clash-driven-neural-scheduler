"""
generate_paper.py
-----------------
Generates the full capstone academic paper as a PDF using ReportLab Platypus.
Output: outputs/reports/capstone_paper.pdf
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    KeepTogether,
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.platypus import ListFlowable, ListItem
from reportlab.platypus import CondPageBreak
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figs"
DATA_VIS = ROOT / "data_visuals"
SENS_DIR = ROOT / "sensitivity_figs"
MIN_TS = ROOT / "outputs" / "min_timeslot_experiment"
EXT_DYN = MIN_TS / "extended_dynamics"
OUT_DIR = ROOT / "outputs" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUT_DIR / "capstone_paper.pdf"

PAGE_W, PAGE_H = A4
L_MARGIN = R_MARGIN = 3.0 * cm
T_MARGIN = 2.8 * cm
B_MARGIN = 2.5 * cm
TEXT_W = PAGE_W - L_MARGIN - R_MARGIN

# ---------------------------------------------------------------------------
# Style sheet
# ---------------------------------------------------------------------------

def build_styles():
    ss = getSampleStyleSheet()

    title = ParagraphStyle(
        "PaperTitle",
        parent=ss["Title"],
        fontSize=18,
        leading=22,
        spaceAfter=6,
        fontName="Times-Bold",
        alignment=TA_CENTER,
        textColor=colors.black,
    )
    subtitle = ParagraphStyle(
        "PaperSubtitle",
        fontSize=11,
        leading=14,
        spaceAfter=4,
        fontName="Times-Italic",
        alignment=TA_CENTER,
        textColor=colors.black,
    )
    author = ParagraphStyle(
        "Author",
        fontSize=11,
        leading=14,
        spaceAfter=2,
        fontName="Times-Roman",
        alignment=TA_CENTER,
    )
    abstract_head = ParagraphStyle(
        "AbstractHead",
        fontSize=10,
        leading=13,
        spaceAfter=4,
        fontName="Times-Bold",
        alignment=TA_CENTER,
    )
    abstract_body = ParagraphStyle(
        "AbstractBody",
        fontSize=10,
        leading=14,
        leftIndent=1.5 * cm,
        rightIndent=1.5 * cm,
        spaceAfter=10,
        fontName="Times-Roman",
        alignment=TA_JUSTIFY,
    )
    h1 = ParagraphStyle(
        "H1",
        fontSize=12,
        leading=16,
        spaceBefore=16,
        spaceAfter=6,
        fontName="Times-Bold",
        alignment=TA_LEFT,
        keepWithNext=1,
    )
    h2 = ParagraphStyle(
        "H2",
        fontSize=11,
        leading=14,
        spaceBefore=10,
        spaceAfter=4,
        fontName="Times-Bold",
        alignment=TA_LEFT,
        keepWithNext=1,
    )
    h3 = ParagraphStyle(
        "H3",
        fontSize=10,
        leading=13,
        spaceBefore=8,
        spaceAfter=3,
        fontName="Times-BoldItalic",
        alignment=TA_LEFT,
    )
    body = ParagraphStyle(
        "Body",
        fontSize=10,
        leading=14,
        spaceAfter=6,
        fontName="Times-Roman",
        alignment=TA_JUSTIFY,
    )
    caption = ParagraphStyle(
        "Caption",
        fontSize=8.5,
        leading=11,
        spaceAfter=8,
        spaceBefore=2,
        fontName="Times-Italic",
        alignment=TA_CENTER,
    )
    table_head = ParagraphStyle(
        "TableHead",
        fontSize=8.5,
        leading=10,
        fontName="Times-Bold",
        alignment=TA_CENTER,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        fontSize=8.5,
        leading=10,
        fontName="Times-Roman",
        alignment=TA_CENTER,
    )
    code_style = ParagraphStyle(
        "Code",
        fontSize=8,
        leading=11,
        fontName="Courier",
        leftIndent=0.8 * cm,
        spaceAfter=4,
        spaceBefore=4,
    )
    ref_style = ParagraphStyle(
        "Ref",
        fontSize=9,
        leading=12,
        fontName="Times-Roman",
        spaceAfter=4,
    )
    return {
        "title": title, "subtitle": subtitle, "author": author,
        "abstract_head": abstract_head, "abstract_body": abstract_body,
        "h1": h1, "h2": h2, "h3": h3, "body": body, "caption": caption,
        "table_head": table_head, "table_cell": table_cell,
        "code": code_style, "ref": ref_style,
    }


# ---------------------------------------------------------------------------
# Image helper
# ---------------------------------------------------------------------------

def img(path: str | Path, width: float = None, max_height: float = None) -> Image | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        pil = PILImage.open(p)
        iw, ih = pil.size
    except Exception:
        return None
    if width is None:
        width = TEXT_W
    aspect = ih / iw
    height = width * aspect
    if max_height and height > max_height:
        height = max_height
        width = height / aspect
    im = Image(str(p), width=width, height=height)
    return im


def img_pair(p1: Path, p2: Path, styles, caption1: str, caption2: str, w: float = None):
    """Return a table with two side-by-side images."""
    if w is None:
        w = (TEXT_W - 0.4 * cm) / 2
    i1 = img(p1, width=w)
    i2 = img(p2, width=w)
    if i1 is None or i2 is None:
        return None
    rows = [
        [i1, i2],
        [Paragraph(caption1, styles["caption"]), Paragraph(caption2, styles["caption"])],
    ]
    t = Table(rows, colWidths=[w, w])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
    ]))
    return t


def grid_images(paths_captions: list[tuple[Path, str]], styles, cols: int = 3):
    """Return a grid of images with captions below each."""
    w = (TEXT_W - (cols - 1) * 0.3 * cm) / cols
    rows = []
    row_imgs = []
    row_caps = []
    for i, (p, cap) in enumerate(paths_captions):
        im = img(p, width=w, max_height=8 * cm)
        row_imgs.append(im if im else Paragraph("(figure not found)", styles["caption"]))
        row_caps.append(Paragraph(cap, styles["caption"]))
        if (i + 1) % cols == 0 or i + 1 == len(paths_captions):
            while len(row_imgs) < cols:
                row_imgs.append("")
                row_caps.append("")
            rows.append(row_imgs[:])
            rows.append(row_caps[:])
            row_imgs.clear()
            row_caps.clear()
    t = Table(rows, colWidths=[w] * cols)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


# ---------------------------------------------------------------------------
# Paragraph helpers
# ---------------------------------------------------------------------------

def P(text: str, style) -> Paragraph:
    return Paragraph(text, style)


def H1(text: str, styles) -> Paragraph:
    return P(text, styles["h1"])


def H2(text: str, styles) -> Paragraph:
    return P(text, styles["h2"])


def H3(text: str, styles) -> Paragraph:
    return P(text, styles["h3"])


def B(text: str, styles) -> Paragraph:
    return P(text, styles["body"])


def sp(h: float = 6) -> Spacer:
    return Spacer(1, h)


def hr() -> HRFlowable:
    return HRFlowable(width="100%", thickness=0.5, color=colors.black, spaceAfter=4, spaceBefore=4)


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def make_table(headers: list, rows: list[list], styles, col_widths=None) -> Table:
    th = styles["table_head"]
    tc = styles["table_cell"]
    data = [[P(h, th) for h in headers]]
    for row in rows:
        data.append([P(str(c), tc) for c in row])
    t = Table(data, colWidths=col_widths, splitByRow=0)
    ts = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDDDDD")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ])
    t.setStyle(ts)
    return t


# ---------------------------------------------------------------------------
# Page templates
# ---------------------------------------------------------------------------

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    page_num = canvas.getPageNumber()
    canvas.drawCentredString(PAGE_W / 2, 1.5 * cm, str(page_num))
    canvas.restoreState()


def first_page(canvas, doc):
    canvas.saveState()
    # Institution name at top (above frame)
    canvas.setFont("Times-Roman", 13)
    y_inst = PAGE_H - 1.5 * cm
    canvas.drawCentredString(PAGE_W / 2, y_inst, "AMERICAN  UNIVERSITY  OF  ARMENIA")
    canvas.setLineWidth(0.6)
    canvas.line(L_MARGIN, y_inst - 0.55 * cm, PAGE_W - R_MARGIN, y_inst - 0.55 * cm)
    # Submission statement at bottom (below frame)
    canvas.setFont("Times-Italic", 10)
    canvas.drawCentredString(PAGE_W / 2, 4.0 * cm,
        "A thesis submitted in fulfillment of the requirements")
    canvas.drawCentredString(PAGE_W / 2, 3.4 * cm,
        "for the degree of Bachelor of Science in Computer Science")
    canvas.setFont("Times-Roman", 11)
    canvas.drawCentredString(PAGE_W / 2, 2.6 * cm, "Yerevan, 2026")
    canvas.restoreState()


def later_pages(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Italic", 8.5)
    canvas.drawString(L_MARGIN, PAGE_H - T_MARGIN + 0.8 * cm,
                      "Clash-Driven Propagation with Hopfield Network Repair for Exam Timetabling Optimization")
    canvas.setFont("Times-Roman", 9)
    canvas.drawCentredString(PAGE_W / 2, B_MARGIN - 0.8 * cm, str(canvas.getPageNumber()))
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Content builders - one function per section
# ---------------------------------------------------------------------------

def sec_title_page(styles) -> list:
    """KMV-style title page: institution+bottom drawn on canvas; middle = title + authors."""
    s = styles
    story = []

    # Push content below the institution line drawn on canvas (~1.5+0.55 cm from top)
    story.append(sp(3.8 * cm))

    # Main title
    story.append(P(
        "Clash-Driven Propagation with Hopfield Network Repair<br/>"
        "for Exam Timetabling Optimization",
        s["title"],
    ))
    story.append(sp(1.5 * cm))
    story.append(hr())
    story.append(sp(1.0 * cm))

    # Two-column Author | Supervisors block (KMV style)
    tp_cell = ParagraphStyle(
        "TpCell", fontSize=11, fontName="Times-Roman",
        leading=17, alignment=TA_LEFT,
    )
    left_text = (
        "<b>Author:</b><br/>"
        "Arpine Tadevosyan"
    )
    right_text = (
        "<b>Supervisors:</b><br/>"
        "Suren Khachatryan<br/>"
        "Aleksandr Hayrapetyan<br/>"
        "<i>American University of Armenia</i>"
    )
    half_w = TEXT_W / 2
    tp_table = Table(
        [[P(left_text, tp_cell), P(right_text, tp_cell)]],
        colWidths=[half_w, half_w],
    )
    tp_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
    ]))
    story.append(tp_table)
    story.append(PageBreak())
    return story


def sec_abstract(styles) -> list:
    """Abstract + keywords on their own page (page 1 of body)."""
    s = styles
    story = []
    story.append(H1("Abstract", s))
    story.append(B(
        "Exam timetabling is a classic hard combinatorial optimization problem. "
        "This paper presents a hybrid algorithm that combines a physics-inspired "
        "propagating-particle model with a Hopfield associative memory network for timetable repair. "
        "In the dynamic model, each exam moves on a discrete cycle of S timeslots. "
        "A shift operator E advances a clashing exam by one slot at a time, up to m times per step. "
        "The parameter m controls the type of motion: when m is coprime with S, "
        "the system can reach any slot (quasi-periodic mode); otherwise it gets restricted to a smaller orbit. "
        "The hybrid method alternates dynamic segments with Hopfield recall steps. "
        "A Hebbian weight matrix is trained on timetable snapshots, and during repair, "
        "clashing exams are moved toward the nearest stored pattern. "
        "A rollback safeguard prevents the repair from making the schedule worse. "
        "I tested both methods on all 12 Toronto benchmark instances under both standard and "
        "tight (paper-minimum) timeslot counts, with runs extended to 10,000 events. "
        "Under standard timeslots, the dynamic-only method reaches zero clashes on 10 of 12 instances. "
        "Under tight timeslots, the hybrid wins on 5 of 12 instances, dynamic-only wins on 2, "
        "and 5 ties occur, including 2 where both methods find a complete zero-clash solution. "
        "Rolling Pearson correlation analysis shows that after the shared initial descent, "
        "the two methods explore different parts of the search space, with mean rolling r ≈ ±0.02. "
        "Sensitivity analysis finds 7 instances in a chaotic regime where small parameter changes "
        "produce large differences in the final result.",
        s,
    ))
    story.append(B(
        "<b>Keywords:</b> exam timetabling, propagating particles, Hopfield network, "
        "discrete dynamical systems, cyclic shift operator, graph coloring, Toronto benchmark, "
        "nonlinear dynamics, convergence analysis, rolling correlation",
        s,
    ))
    story.append(PageBreak())
    return story


def sec_intro(styles) -> list:
    s = styles
    story = []
    story.append(H1("1. Introduction", s))
    story.append(B(
        "University exam timetabling is one of the classic hard problems in combinatorial optimization. "
        "The task is to assign each of N exams to one of S available timeslots so that no two exams "
        "with shared students end up in the same slot. Two exams conflict if at least one student "
        "is enrolled in both. Finding a valid zero-clash schedule is equivalent to properly coloring "
        "the conflict graph, a problem that is NP-complete in general [4], [15]. "
        "In practice, additional soft constraints like spreading exams evenly make it even harder. "
        "Since universities run this process every semester, having a reliable automated tool matters [1], [6].",
        s,
    ))
    story.append(B(
        "Exact methods like integer programming only work for small instances. "
        "For realistic sizes, with hundreds of exams and thousands of students, "
        "researchers rely on heuristic approaches: simulated annealing, tabu search, "
        "evolutionary algorithms, and graph-coloring heuristics [7], [8]. "
        "These methods often give good results, but they tell us little about "
        "why the search space is hard or how conflicts actually get resolved.",
        s,
    ))
    story.append(B(
        "A different perspective comes from nonlinear physics. "
        "If we treat each exam as a particle moving on a discrete cycle of S slots, "
        "timetabling becomes a problem of driving a high-clash starting state toward a low-clash attractor. "
        "The shift parameter m controls the character of the motion: "
        "when gcd(m, S) = 1, a single exam can reach any slot over time (quasi-periodic mode); "
        "when m divides S, it gets trapped in a smaller orbit. "
        "These differences have real effects on how well the algorithm escapes local minima.",
        s,
    ))
    story.append(B(
        "Hopfield networks [10] offer another useful idea. "
        "They work as associative memories: trained on timetable snapshots using Hebbian learning [11], "
        "the network can push a conflict-heavy schedule toward a stored low-clash configuration. "
        "Instead of fixing one conflict at a time, the recall step moves all clashing exams at once. "
        "This can jump over local minima that purely local methods cannot escape.",
        s,
    ))
    story.append(B(
        "This paper describes a hybrid algorithm that alternates dynamic propagation steps with "
        "Hopfield recall and repair. I started from an initial prototype and built on it through "
        "a series of experiments: mapping the effect of mode m, studying sensitivity to driving "
        "parameters, testing under tight timeslot constraints, and analyzing 10,000-event trajectories "
        "with rolling correlation. All experiments use the Toronto benchmark [1], "
        "the standard test suite for exam timetabling.",
        s,
    ))
    story.append(B(
        "The rest of the paper is organized as follows. "
        "Section 2 reviews the relevant literature. "
        "Section 3 formally defines the problem and the metrics I used. "
        "Section 4 explains the full methodology. "
        "Section 5 describes the Python implementation. "
        "Section 6 covers the experimental setup. "
        "Section 7 presents results, starting with dataset-level plots and then going instance by instance. "
        "Sections 8 through 12 cover discussion, conclusion, future work, acknowledgments, and references.",
        s,
    ))
    return story


def sec_literature(styles) -> list:
    s = styles
    story = []
    story.append(H1("2. Literature Review", s))

    story.append(H2("2.1  Exam Timetabling: Problem and Benchmarks", s))
    story.append(B(
        "Carter, Laporte, and Lee [1] introduced the Toronto benchmark suite: twelve real-world "
        "exam scheduling instances from North American universities that became the standard "
        "comparison set for timetabling algorithms. Each instance comes with the exam count, "
        "student count, conflict graph density, and the minimum feasible timeslot count. "
        "The same group also published an early algorithm survey [2] that formalized the "
        "graph-coloring connection: two exams share a conflict edge if any student is in both, "
        "and a valid timetable is a proper graph coloring.",
        s,
    ))
    story.append(B(
        "Even, Itai, and Shamir [3] proved that variants of timetabling are NP-complete. "
        "Garey and Johnson [4] placed graph coloring firmly in the NP-hard class. "
        "Welsh and Powell [16] gave one of the earliest greedy coloring heuristics: "
        "sort vertices by decreasing degree and assign the smallest available color. "
        "It is fast but usually uses more timeslots than the minimum possible.",
        s,
    ))
    story.append(B(
        "Qu et al. [7] survey search methods for exam timetabling, "
        "including constraint-based methods, local search, and population-based approaches. "
        "Burke and Petrovic [8] review more recent work with a focus on soft constraints. "
        "McCollum et al. [9] describe the ITC-2007 competition, which extended the Toronto "
        "benchmarks with soft-constraint scores.",
        s,
    ))

    story.append(H2("2.2  Propagating-Particle and Periodic Dynamics", s))
    story.append(B(
        "The propagating-particle model treats each exam as something that shifts its slot "
        "when it is in conflict. This can be formalized as an "
        "operator E(m, d) on the N-dimensional discrete torus Z_S^N: applying E moves exam k "
        "by +1 slot (mod S) if it is in conflict, up to m times. "
        "Quasi-periodic modes (gcd(m, S) = 1) work best because they let every "
        "exam eventually reach any slot, avoiding the trapping that happens in resonant modes.",
        s,
    ))
    story.append(B(
        "The connection to nonlinear physics is an important part of the framework. "
        "The Chirikov standard map [13] is a well-known map on the torus that transitions "
        "from regular to chaotic behavior as a kick parameter grows. "
        "The propagating-particle system is a dissipative, high-dimensional analog: "
        "total clashes play the role of energy that decreases on average, "
        "while the cyclic state space creates different orbit structures depending on m. "
        "Strogatz [14] provides the nonlinear dynamics background I drew on here.",
        s,
    ))

    story.append(H2("2.3  Hopfield Networks and Associative Memory", s))
    story.append(B(
        "Hopfield [10] introduced a class of recurrent networks that act as associative memories. "
        "The energy E = -½ x^T W x decreases monotonically under asynchronous updates, "
        "so the network always converges to a fixed point. "
        "Patterns are stored using the Hebbian learning rule W = sum_mu xi^mu (xi^mu)^T [11]. "
        "Hopfield and Tank [12] later extended this to optimization problems "
        "by formulating constraints as energy minimization.",
        s,
    ))
    story.append(B(
        "The storage capacity was analyzed by Amit, Gutfreund, and Sompolinsky [5]: "
        "a network of N units can reliably recall at most about 0.138 x N patterns "
        "before spurious attractors start to dominate. "
        "Kriesel [6] gives a clear practical overview of Hopfield networks and their limits. "
        "In this project, I use a capacity limit of 0.14 x N, consistent with this bound.",
        s,
    ))

    story.append(H2("2.4  Hybrid and Physics-Inspired Timetabling Approaches", s))
    story.append(B(
        "Combining a local search with a global repair step is a common idea in timetabling. "
        "The motivation for pairing the dynamic model with Hopfield recall is that they work "
        "at different scales. The dynamic operator makes small sequential adjustments, one exam "
        "and one slot at a time. The Hopfield recall makes a simultaneous global jump toward a "
        "stored low-clash pattern. Together, they provide local refinement with occasional global "
        "escapes when the local search gets stuck.",
        s,
    ))
    return story


def sec_problem_def(styles) -> list:
    s = styles
    story = []
    story.append(H1("3. Problem Definition", s))

    eq_style = ParagraphStyle(
        "EqBlock", parent=s["body"],
        leftIndent=2.0 * cm, fontName="Times-Italic",
        spaceAfter=4, spaceBefore=4,
    )
    story.append(H2("3.1  Formal Problem Statement", s))
    story.append(B(
        "Let the set of exams and the set of timeslots be defined as:",
        s,
    ))
    story.append(P("E = { e_1, e_2, ..., e_N }  —  set of N exams", eq_style))
    story.append(P("T = { 0, 1, ..., S−1 }  —  set of S timeslots", eq_style))
    story.append(B(
        "A student enrollment matrix A records which students are enrolled in which exams. "
        "The conflict graph G = (E, C) has an edge (e_i, e_j) ∈ C if any student is enrolled "
        "in both e_i and e_j. A timetable is a function:",
        s,
    ))
    story.append(P("σ : E → T", eq_style))
    story.append(B(
        "that assigns each exam to a timeslot.",
        s,
    ))
    story.append(B(
        "A <b>clash</b> occurs when two conflicting exams are assigned the same slot. "
        "The total clash count is:",
        s,
    ))
    story.append(P(
        "C(σ) = 2 × |{ (i, j) ∈ C  :  σ(e_i) = σ(e_j) }|",
        eq_style,
    ))
    story.append(B(
        "The factor of 2 counts each conflicting pair from both sides. "
        "A timetable is <b>feasible</b> if C(σ) = 0. "
        "The goal is to minimize clashes; soft constraints like exam spread are not the focus here.",
        s,
    ))
    story.append(B(
        "The <b>chromatic number</b> chi(G) is the minimum number of timeslots needed for a "
        "feasible timetable. An instance with S = chi(G) is as tight as it can get. "
        "I call these <b>paper-minimum timeslots</b>: the S values for which Carter et al. [1] "
        "demonstrate that feasible solutions exist on the Toronto instances.",
        s,
    ))

    story.append(H2("3.2  Toronto Benchmark Instances", s))
    story.append(B(
        "I used all 12 instances from the Toronto benchmark suite. "
        "Table 1 summarizes the key properties of each instance.",
        s,
    ))

    headers_t1 = ["Instance", "Exams (N)", "Students", "Std. Slots", "Min. Slots",
                  "Edges", "Density (%)", "Avg. Degree"]
    rows_t1 = [
        ["car-f-92", "543", "18,419", "32", "28", "20,305", "13.8", "74.8"],
        ["car-s-91", "682", "16,925", "35", "28", "29,814", "12.8", "87.4"],
        ["ear-f-83", "190", "1,125",  "24", "22", "4,793",  "26.7", "50.5"],
        ["hec-s-92", "81",  "2,823",  "18", "17", "1,363",  "42.1", "33.7"],
        ["kfu-s-93", "461", "5,349",  "20", "19", "5,893",  "5.6",  "25.6"],
        ["lse-f-91", "381", "2,726",  "18", "17", "4,531",  "6.3",  "23.8"],
        ["rye-s-93", "486", "11,483", "23", "21", "8,872",  "7.5",  "36.5"],
        ["sta-f-83", "139", "611",    "13", "13", "1,381",  "14.4", "19.9"],
        ["tre-s-92", "261", "4,360",  "23", "20", "6,131",  "18.1", "47.0"],
        ["uta-s-92", "622", "21,266", "35", "30", "24,249", "12.6", "78.0"],
        ["ute-s-92", "184", "2,749",  "10", "10", "1,430",  "8.5",  "15.5"],
        ["yor-f-83", "181", "941",    "21", "19", "4,706",  "28.9", "52.0"],
    ]
    cw_t1 = [2.2*cm, 1.5*cm, 1.5*cm, 1.5*cm, 1.5*cm, 1.5*cm, 1.8*cm, 2.0*cm]
    story.append(make_table(headers_t1, rows_t1, s, col_widths=cw_t1))
    story.append(P("Table 1. Toronto benchmark instance properties. "
                   "Std. Slots = standard timeslot count; "
                   "Min. Slots = paper-minimum timeslots; "
                   "Density = fraction of possible conflict edges present.",
                   s["caption"]))

    # Dataset overview figure — full width, no height cap so it renders properly
    fig_path = DATA_VIS / "00_all_instances_summary.png"
    im = img(fig_path, width=TEXT_W, max_height=14 * cm)
    if im:
        story.append(PageBreak())
        story.append(im)
        story.append(P("Figure 1. Overview of Toronto benchmark instances: exams, students, "
                       "and clash graph properties across the 12 instances used in this study.",
                       s["caption"]))

    story.append(H2("3.3  Evaluation Metrics", s))
    story.append(B(
        "The main metric is the <b>minimum clash count</b> over a run: "
        "min_clash = min over all events of C(sigma_t). "
        "I also track: "
        "(i) <b>final clash count</b> at the last event; "
        "(ii) <b>zero-clash rate</b>: how often a run reaches C(sigma) = 0; "
        "(iii) <b>first-zero event</b>: when zero clashes is first achieved; "
        "(iv) <b>first-minimum event</b>: when the run's minimum is first reached; "
        "(v) <b>mean rolling Pearson r</b>: the average Pearson correlation between the "
        "dynamic-only and hybrid clash trajectories in a sliding window of W = 20 events.",
        s,
    ))
    return story


def sec_methodology(styles) -> list:
    s = styles
    story = []
    story.append(H1("4. Methodology", s))

    story.append(H2("4.1  State Space and the E Operator", s))
    story.append(B(
        "The algorithm works on the discrete N-dimensional torus Z_S^N. "
        "A state is a slot assignment vector sigma with N values in {0, ..., S-1}. "
        "The <b>E operator</b> with shift parameter m processes each exam k in a set order. "
        "If exam k is in conflict with any neighbor in its current slot, "
        "the slot advances by +1 (mod S). This repeats up to m times, "
        "stopping early if the conflict is resolved. "
        "One full pass over all N exams is one <b>dynamic step</b>, or one <b>event</b>.",
        s,
    ))
    story.append(B(
        "The parameter m determines the type of motion. "
        "Writing m_r = m mod S, the mode is classified as:",
        s,
    ))
    mode_data = [
        ["Mode Class", "Condition", "Behavior"],
        ["Quasi-periodic", "gcd(m_r, S) = 1", "Single-particle map is an ergodic rotation of Z_S; every slot reachable"],
        ["Periodic-factor", "S divisible by m_r", "Orbit restricted to S/gcd(m_r,S) slots; creates trapped subspaces"],
        ["Periodic-resonant", "gcd(m_r, S) = g > 1, S not divisible by m_r", "Mixed behavior with resonance subspaces"],
        ["Identity", "m_r = 0", "No net shift; trivial fixed-point"],
    ]
    t_mode = Table(
        [[P(r[0], s["table_head"]), P(r[1], s["table_head"]), P(r[2], s["table_head"])] for r in mode_data[:1]] +
        [[P(r[0], s["table_cell"]), P(r[1], s["table_cell"]), P(r[2], s["table_cell"])] for r in mode_data[1:]],
        colWidths=[3.2*cm, 3.8*cm, 8.0*cm],
    )
    t_mode.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDDDDD")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
    ]))
    story.append(t_mode)
    story.append(P("Table 2. Mode classification of the E operator.", s["caption"]))

    story.append(H2("4.2  Ordering Strategies", s))
    story.append(B(
        "The order in which exams are processed each step affects how the trajectory unfolds. "
        "I implemented four orderings: "
        "(i) <b>natural</b>: exams processed from index 1 to N; "
        "(ii) <b>random</b>: a new random permutation each step; "
        "(iii) <b>largest-weighted-degree</b>: exams sorted by number of shared-student conflicts, "
        "highest first; "
        "(iv) <b>saturation-like</b>: exams sorted by 2 x degree + weighted_degree. "
        "The weighted-degree ordering puts the most conflicted exams first, "
        "similar to the DSATUR graph-coloring heuristic.",
        s,
    ))

    story.append(H2("4.3  Dynamic-Only Method", s))
    story.append(B(
        "The dynamic-only method runs T sequential dynamic steps starting from all exams in slot 0. "
        "The clash count is recorded after each step, forming the <b>clash trajectory</b>. "
        "The run stops early if clashes reach zero. "
        "I track min_clash (the lowest clash count reached) and the step at which it first occurred.",
        s,
    ))

    story.append(H2("4.4  Hopfield Autoassociator Component", s))
    story.append(B(
        "The Hopfield component is a fully connected network of N units with weight matrix W, "
        "initialized to zero. Training uses the Hebbian rule: for each timeslot s, "
        "I encode the current assignment as a bipolar pattern "
        "(+1 if an exam is in slot s, -1 otherwise) and update W by adding the outer product. "
        "I store at most 0.14 x N patterns; beyond this limit, training is skipped to avoid "
        "spurious attractors [5].",
        s,
    ))
    story.append(B(
        "<b>Pre-training warm-up:</b> Before the main run, the network is pre-trained on "
        "patterns from short warm-up runs using a few candidate modes. "
        "This gives the network useful starting material before real optimization begins.",
        s,
    ))
    story.append(B(
        "<b>Recall and repair:</b> For each clashing exam, I compute how well each slot matches "
        "the stored patterns and move the exam to the best-matching option, "
        "as long as the move does not increase that exam's own clash count. "
        "If the total clash count after all moves is higher than before, "
        "the entire repair step is rolled back.",
        s,
    ))

    story.append(H2("4.5  Hybrid Schedule", s))
    story.append(B(
        "The hybrid method follows a repeating cycle pattern (L_1, L_2, ..., L_P): "
        "run L_1 dynamic steps, call Hopfield repair, run L_2 steps, call repair, and so on. "
        "The number of cycles is chosen so the total event count reaches about 10,000. "
        "After each dynamic segment, the network is also updated with the current timetable.",
        s,
    ))
    story.append(B(
        "I tested six patterns: [10, 100], [25, 100], [50, 100], [10, 50, 100], "
        "[10, 25, 50, 100], [100]. Short segments give the repair step an early chance to help; "
        "longer segments let the dynamics run before repair is called. "
        "For the minimum-timeslot extended experiments, I used the best-performing "
        "pattern per instance from the standard-timeslot results.",
        s,
    ))

    story.append(H2("4.6  Minimum-Timeslot Experiment", s))
    story.append(B(
        "To see how the algorithm behaves near the graph-coloring lower bound, "
        "I re-ran each instance with its paper-minimum timeslot count. "
        "These tighter constraints make the problem much harder: valid colorings are "
        "a smaller fraction of the state space, and the landscape has more local minima. "
        "I compared dynamic-only and hybrid under the same event budget.",
        s,
    ))

    story.append(H2("4.7  Extended 10,000-Event Experiment", s))
    story.append(B(
        "The initial minimum-timeslot experiment used 350-1,000 events. "
        "To study long-run convergence, I extended both methods to 10,000 events. "
        "Dynamic-only runs 10,000 steps. For the hybrid, the cycle count is scaled so the "
        "total (dynamic steps + Hopfield calls) reaches about 10,000. "
        "The full clash trajectory is saved as a numpy array for later analysis.",
        s,
    ))

    story.append(H2("4.8  Rolling Correlation Analysis", s))
    story.append(B(
        "To compare how the two methods' trajectories relate over time, I compute the "
        "rolling Pearson r in a sliding window of width W = min(20, max(5, n // 4)), "
        "where n is the common trajectory length.",
        s,
    ))
    story.append(P(
        "<i>r</i>(t) = Pearson( dyn[t − W + 1 : t + 1],  hyb[t − W + 1 : t + 1] )",
        ParagraphStyle("eq", parent=s["body"], leftIndent=2*cm, fontName="Times-Italic",
                       fontSize=10, spaceAfter=6, spaceBefore=4),
    ))
    story.append(B(
        "Windows where either sequence is constant return NaN and appear as gaps in the plot. "
        "The mean rolling r is computed over all valid windows. "
        "A near-zero mean r means the two methods' local fluctuations are uncorrelated: "
        "after the shared initial descent, they are exploring different parts of the landscape.",
        s,
    ))
    return story


def sec_implementation(styles) -> list:
    s = styles
    story = []
    story.append(H1("5. Implementation", s))

    story.append(H2("5.1  Project Structure", s))
    story.append(B(
        "The project is written in Python 3.11 and organized as follows:",
        s,
    ))
    story.append(Paragraph(
        "models/            Instance, Course, Autoassociator dataclasses<br/>"
        "scripts/           experiment drivers and visualization scripts<br/>"
        "data/              Toronto benchmark .crs / .stu files<br/>"
        "outputs/           all generated CSV, JSON, .npy, PNG, HTML results<br/>"
        "sensitivity_figs/  sensitivity sweep outputs<br/>"
        "figs/              main experiment visualizations",
        ParagraphStyle("code_indent", parent=s["body"],
                       fontName="Courier", fontSize=9, leading=13,
                       leftIndent=1.2*cm, spaceAfter=6),
    ))
    story.append(B(
        "The core algorithm module contains the instance loader, the E operator, "
        "the Hopfield network, and the hybrid scheduler. "
        "All functions are deterministic given a fixed seed, which makes results reproducible. "
        "Parallel runs use Python's ProcessPoolExecutor "
        "(4 workers for extended runs, 14 for the main experiment).",
        s,
    ))

    story.append(H2("5.2  Key Implementation Details", s))
    story.append(B(
        "<b>Instance loading:</b> The benchmark input files contain exam enrollment sizes "
        "and per-student course lists. The loader builds the conflict graph as a numpy edge list "
        "and computes weighted degrees (shared-student counts per exam pair), "
        "used for the largest-weighted-degree ordering.",
        s,
    ))
    story.append(B(
        "<b>Clash computation:</b> Total clashes = 2 × (number of conflict edges where both "
        "endpoints share the same slot), vectorized with numpy boolean indexing. "
        "This runs in O(|C|) time per call.",
        s,
    ))
    story.append(B(
        "<b>E operator:</b> The step function iterates over all N exams in the chosen order. "
        "For each exam, it checks if any neighbor shares the slot and shifts up to m times, "
        "stopping early if the conflict is resolved.",
        s,
    ))
    story.append(B(
        "<b>Hopfield repair:</b> The N × N weight matrix W is stored as float32. "
        "Net activations are computed for all exam-slot pairs using matrix-vector products. "
        "The repair loop only processes exams currently in conflict, "
        "which keeps it fast when clashes are already low.",
        s,
    ))
    story.append(B(
        "<b>Rollback safeguard:</b> Before each Hopfield call, I save a copy of the slot vector. "
        "If total clashes after repair is higher than before, I restore the copy. "
        "This means the Hopfield step can never make the schedule worse.",
        s,
    ))
    story.append(B(
        "<b>Extended run scaling:</b> The cycle count is computed as "
        "max(3, (target − 1) // (sum(pattern) + len(pattern))). "
        "For pattern [50, 100] and target = 10,000, this gives about 66 cycles "
        "and roughly 9,966 total events.",
        s,
    ))

    story.append(H2("5.3  Reproducibility", s))
    story.append(B(
        "All experiments fix seeds: Python's random.Random(seed) for exam ordering "
        "and numpy.random.seed(seed) for any stochastic operations. "
        "Three seeds (0, 1, 2) are used for the main experiment; "
        "seed 0 for the extended-dynamics runs. "
        "The run configuration is serialized at startup. "
        "Clash history arrays are stored in binary format so results can be "
        "re-analyzed without re-running the full experiment.",
        s,
    ))
    return story


def sec_experimental_setup(styles) -> list:
    s = styles
    story = []
    story.append(H1("6. Experimental Setup", s))

    story.append(H2("6.1  Datasets", s))
    story.append(B(
        "I used all 12 standard Toronto benchmark instances (Table 1). "
        "They range from 81 to 682 exams and 611 to 21,266 students, "
        "with conflict densities from 5.6% to 42.1%. "
        "hec-s-92 has the highest conflict density; ute-s-92 has the fewest paper-minimum timeslots. "
        "The 13th Toronto instance (pur-s-93, 2,419 exams) was left out of the minimum-timeslot "
        "experiment because it is much more expensive to run.",
        s,
    ))

    story.append(H2("6.2  Main Experiment (Standard Timeslots)", s))
    story.append(B(
        "The main experiment runs 2,899 combinations of instance, mode, duration, pattern, "
        "ordering, and seed, using 14 parallel workers. "
        "Dynamic-only runs use durations of 500, 1,000, and 3,000 steps "
        "(400 for the three large instances). "
        "Hybrid runs use 5 cycles of each pattern with 5 Hopfield calls per cycle. "
        "Orderings tested: natural, largest-weighted-degree, saturation-like. "
        "Seeds 0, 1, 2 per configuration. Mode m swept from 1 to 2S.",
        s,
    ))

    story.append(H2("6.3  Minimum-Timeslot Extended Experiment", s))
    story.append(B(
        "I re-ran each instance under its paper-minimum timeslot count. "
        "For each instance and method, I picked the best mode m and ordering "
        "from the standard experiment by minimum clash. "
        "This transfer is reasonable because the structural properties of each "
        "instance's conflict graph — density, degree distribution, and chromatic "
        "structure — do not change when the timeslot count is reduced. "
        "The mode that allows the most ergodic exploration under standard S "
        "remains quasi-periodic and thus the best-coverage choice at minimum S as well. "
        "Both methods were extended to 10,000 total events.",
        s,
    ))

    story.append(H2("6.4  Evaluation Protocol", s))
    story.append(B(
        "For each run, I recorded: minimum clash, final clash, event index of first minimum, "
        "event index of first zero (if reached), number of Hopfield calls, "
        "and the full clash trajectory. "
        "The main comparison metric is <b>minimum clash under paper-minimum timeslots</b> "
        "at the 10,000-event budget. "
        "Rolling Pearson r uses W = 20 for trajectories with 80 or more events, "
        "and W = max(5, n//4) for shorter ones.",
        s,
    ))
    return story


# ---------------------------------------------------------------------------
# Section 7: Results
# ---------------------------------------------------------------------------

def sec_results_overview(styles) -> list:
    s = styles
    story = []
    story.append(H1("7. Results and Analysis", s))
    story.append(H2("7.1  Dataset-Level Overview", s))
    story.append(B(
        "Before looking at results, it helps to get a sense of the dataset structure. "
        "Figures 2a and 2b show how the instances differ in size and difficulty.",
        s,
    ))

    # Instance size/density plots — equal width side by side
    p1 = DATA_VIS / "00_size_vs_density.png"
    p2 = DATA_VIS / "00_clash_difficulty.png"
    half = (TEXT_W - 0.4 * cm) / 2
    pair = img_pair(p1, p2, s,
                    "(a) Exam count vs. conflict graph density.",
                    "(b) Clash difficulty index by instance.")
    if pair:
        story.append(pair)
        story.append(P(
            "Figure 2. Dataset structural properties for all 12 Toronto benchmark instances: "
            "(a) number of exams vs. conflict graph density; "
            "(b) clash difficulty index as a function of N and density.",
            s["caption"],
        ))

    story.append(H2("7.2  Standard-Timeslot Experiment Results", s))
    story.append(B(
        "Under standard timeslots, the dynamic-only method reaches zero clashes on 10 of 12 instances. "
        "car-f-92 and lse-f-91 are the two misses, with residuals of 12 and 2 clashes respectively. "
        "A greedy graph-coloring baseline also solves most instances easily in this setting, "
        "which confirms the problem is not very hard under standard timeslots. "
        "The Hopfield-only method (no dynamic component) performs very poorly, with thousands of clashes. "
        "This shows that Hopfield recall alone cannot reduce clashes from the all-zero starting state.",
        s,
    ))

    # Main comparison figure
    im1 = img(FIG_DIR / "dyn_vs_hybrid.png", width=TEXT_W * 0.85, max_height=8*cm)
    if im1:
        story.append(im1)
        story.append(P("Figure 3. Minimum clashes achieved by dynamic-only vs hybrid method "
                       "across all 12 instances under standard timeslots. "
                       "Lower is better; zero clashes (feasible timetable) is the goal.",
                       s["caption"]))

    im2 = img(FIG_DIR / "paper_max_vs_best.png", width=TEXT_W * 0.85, max_height=7*cm)
    if im2:
        story.append(im2)
        story.append(P("Figure 4. Ratio of best achieved minimum clashes to paper maximum clashes. "
                       "Values near 1.0 indicate near-elimination of all conflicts.",
                       s["caption"]))

    story.append(H3("Mode Type Analysis", s))
    story.append(B(
        "Figure 5 shows minimum clashes grouped by mode type. "
        "Quasi-periodic modes (gcd(m, S) = 1) consistently give lower clashes than "
        "periodic-resonant or periodic-factor modes. "
        "This matches the theoretical expectation: when an exam can reach any slot, "
        "the system is better at escaping local minima.",
        s,
    ))
    im3 = img(FIG_DIR / "mode_type_comparison.png", width=TEXT_W * 0.85, max_height=7*cm)
    if im3:
        story.append(im3)
        story.append(P("Figure 5. Minimum clash distributions by mode-type category "
                       "(quasi-periodic, periodic-factor, periodic-resonant). "
                       "Quasi-periodic modes achieve lower median clash counts.",
                       s["caption"]))

    im4 = img(FIG_DIR / "best_mode_heatmap.png", width=TEXT_W * 0.85, max_height=7*cm)
    if im4:
        story.append(im4)
        story.append(P("Figure 6. Best mode m (color) and best minimum clashes (annotation) "
                       "per instance for the dynamic-only experiment under standard timeslots.",
                       s["caption"]))

    story.append(H3("Hopfield Effect", s))
    im5 = img(FIG_DIR / "hopfield_effect.png", width=TEXT_W * 0.85, max_height=7*cm)
    if im5:
        story.append(im5)
        story.append(P("Figure 7. Effect of individual Hopfield repair calls: "
                       "distribution of clash-count change (Δ clashes) across all repair events. "
                       "The majority of calls are beneficial or neutral; "
                       "the rollback safeguard prevents negative outcomes.",
                       s["caption"]))

    story.append(H2("7.3  Sensitivity Analysis", s))
    story.append(B(
        "To understand whether the algorithm is settling into fixed attractors or behaving chaotically, "
        "I ran a sensitivity sweep over the (mode m, dynamic duration d) grid for each instance. "
        "I measured normalized sensitivity S = max(|dE/dh|, |dE/dd|) / mean(E), "
        "where h is the Hopfield call frequency. An instance is labeled SENSITIVE if S > 0.05.",
        s,
    ))

    im6 = img(SENS_DIR / "comparison_normalized.png", width=TEXT_W * 0.85, max_height=7*cm)
    if im6:
        story.append(im6)
        story.append(P("Figure 8. Normalized sensitivity S across all 13 Toronto instances. "
                       "Red bars (S > 0.05) indicate the SENSITIVE/chaotic regime; "
                       "green bars indicate the STABLE/frozen regime. "
                       "Seven instances are sensitive; six are stable "
                       "(most of the latter achieve zero clashes trivially).",
                       s["caption"]))

    # Sensitivity verdict table
    story.append(sp(4))
    headers_sens = ["Instance", "N", "S (slots)", "Verdict", "S-score", "mean(E)", "d_H/N"]
    rows_sens = [
        ["sta-f-83",  "139", "13", "STABLE",    "0.000", "0.00",   "0.016"],
        ["ute-s-92",  "184", "10", "STABLE",    "0.000", "0.00",   "0.338"],
        ["car-s-91",  "682", "35", "STABLE",    "0.047", "297.17", "0.861"],
        ["uta-s-92",  "622", "35", "STABLE",    "0.012", "169.50", "0.872"],
        ["car-f-92",  "543", "32", "STABLE",    "0.036", "179.73", "0.890"],
        ["hec-s-92",  "81",  "18", "SENSITIVE", "1.000", "2.00",   "0.709"],
        ["ear-f-83",  "190", "24", "SENSITIVE", "0.328", "36.53",  "0.857"],
        ["lse-f-91",  "381", "18", "SENSITIVE", "0.556", "21.60",  "0.752"],
        ["kfu-s-93",  "461", "20", "SENSITIVE", "0.294", "11.33",  "0.635"],
        ["yor-f-83",  "181", "21", "SENSITIVE", "0.544", "33.07",  "0.918"],
        ["rye-s-93",  "486", "23", "SENSITIVE", "1.180", "11.87",  "0.872"],
        ["tre-s-92",  "261", "23", "SENSITIVE", "0.641", "40.53",  "0.872"],
    ]
    cw_sens = [2.4*cm, 1.2*cm, 1.5*cm, 2.2*cm, 1.6*cm, 1.6*cm, 1.6*cm]
    story.append(make_table(headers_sens, rows_sens, s, col_widths=cw_sens))
    story.append(P("Table 3. Sensitivity analysis results. "
                   "S-score = normalized sensitivity index; mean(E) = mean clash over (m,d) grid; "
                   "d_H/N = normalized Hamming distance of final assignments across runs "
                   "(near 1 = ergodic exploration).",
                   s["caption"]))

    story.append(B(
        "The stable instances where S = 0 (sta-f-83, ute-s-92) have mean(E) = 0: "
        "no matter what parameters are used, the system always finds zero clashes. "
        "The sensitive instances show non-monotone energy landscapes and Hamming "
        "distances near 0.8-0.9 between final assignments from different (m, d) settings. "
        "This means different parameter choices lead to very different final timetables, "
        "which is essentially chaotic sensitivity in a discrete system.",
        s,
    ))

    story.append(H2("7.4  Minimum-Timeslot Extended Results", s))
    story.append(B(
        "Under paper-minimum timeslots with a 10,000-event budget, the problem is noticeably harder. "
        "Table 4 shows the results for all 12 instances.",
        s,
    ))

    headers_min = ["Instance", "Std. Slots", "Min. Slots", "Dyn Min Clash",
                   "Hyb Min Clash", "Δ (Dyn−Hyb)", "Winner"]
    rows_min = [
        ["car-f-92", "32", "28", "120", "124", "−4",  "Dynamic"],
        ["car-s-91", "35", "28", "328", "186", "+142","Hybrid"],
        ["ear-f-83", "24", "22", "16",  "8",   "+8",  "Hybrid"],
        ["hec-s-92", "18", "17", "2",   "2",   "0",   "Tie"],
        ["kfu-s-93", "20", "19", "16",  "16",  "0",   "Tie"],
        ["lse-f-91", "18", "17", "12",  "14",  "−2",  "Dynamic"],
        ["rye-s-93", "23", "21", "18",  "18",  "0",   "Tie"],
        ["sta-f-83", "13", "13", "0",   "0",   "0",   "Tie (both 0)"],
        ["tre-s-92", "23", "20", "42",  "22",  "+20", "Hybrid"],
        ["uta-s-92", "35", "30", "92",  "52",  "+40", "Hybrid"],
        ["ute-s-92", "10", "10", "0",   "0",   "0",   "Tie (both 0)"],
        ["yor-f-83", "21", "19", "50",  "40",  "+10", "Hybrid"],
    ]
    cw_min = [2.4*cm, 1.8*cm, 1.8*cm, 2.2*cm, 2.2*cm, 2.0*cm, 2.6*cm]
    story.append(make_table(headers_min, rows_min, s, col_widths=cw_min))
    story.append(P("Table 4. Extended 10,000-event results under paper-minimum timeslots. "
                   "Δ = (dynamic_min − hybrid_min); positive = hybrid wins; negative = dynamic wins. "
                   "sta-f-83 and ute-s-92 both reach zero clashes even at minimum slots.",
                   s["caption"]))

    im7 = img(MIN_TS / "dynamic_vs_hybrid_min_slots.png", width=TEXT_W * 0.85, max_height=8*cm)
    if im7:
        story.append(im7)
        story.append(P("Figure 9. Grouped bar chart comparing minimum clashes for dynamic-only "
                       "and hybrid methods across all 12 instances under paper-minimum timeslots "
                       "(10,000-event budget).",
                       s["caption"]))

    im8 = img(MIN_TS / "hybrid_improvement_delta.png", width=TEXT_W * 0.85, max_height=8*cm)
    if im8:
        story.append(im8)
        story.append(P("Figure 10. Per-instance improvement delta (dynamic_min_clash − hybrid_min_clash) "
                       "under paper-minimum timeslots. "
                       "Positive bars indicate hybrid advantage; negative bars indicate dynamic advantage.",
                       s["caption"]))

    story.append(B(
        "<b>Score: hybrid wins 5, dynamic-only wins 2, ties 5</b> (including 2 where both reach zero). "
        "The biggest hybrid wins are on car-s-91 (delta = +142), uta-s-92 (delta = +40), "
        "and tre-s-92 (delta = +20). "
        "The two dynamic-only wins on car-f-92 and lse-f-91 are small (delta = -4 and -2). "
        "The Hopfield repair helps most on the harder, larger instances "
        "where the dynamic walk tends to get stuck.",
        s,
    ))

    story.append(H3("Convergence Plateau Analysis", s))
    story.append(B(
        "One of the more interesting findings is how quickly both methods reach their minimum. "
        "Almost every instance hits its best clash count within the first 5% of the 10,000-event "
        "budget (Table 5). "
        "For example, the hybrid on car-f-92 finds its minimum at event 112 out of 9,969 (1.1%). "
        "Dynamic-only on tre-s-92 finds it at event 5 (0.1%). "
        "The only exception is rye-s-93 hybrid, which keeps improving until event 6,684 (67%). "
        "In all other cases, the remaining budget produces no improvement: "
        "the system is stuck oscillating around the same level, not still descending.",
        s,
    ))

    headers_conv = ["Instance", "Exp.", "First Min Event", "n Events", "% Remaining", "Min Clash"]
    rows_conv = [
        ["car-f-92", "dynamic", "8,088", "10,001", "19.1%", "120"],
        ["car-f-92", "hybrid",  "112",   "9,969",  "98.9%", "124"],
        ["car-s-91", "dynamic", "473",   "10,001", "95.3%", "328"],
        ["car-s-91", "hybrid",  "202",   "10,000", "98.0%", "186"],
        ["hec-s-92", "dynamic", "777",   "10,001", "92.2%", "2"],
        ["hec-s-92", "hybrid",  "234",   "10,000", "97.7%", "2"],
        ["lse-f-91", "dynamic", "2,679", "10,001", "73.2%", "12"],
        ["rye-s-93", "hybrid",  "6,684", "9,969",  "32.9%", "18"],
        ["uta-s-92", "dynamic", "5",     "10,001", "100.0%","92"],
        ["uta-s-92", "hybrid",  "123",   "9,969",  "98.8%", "52"],
    ]
    cw_conv = [2.4*cm, 2.0*cm, 2.4*cm, 2.0*cm, 2.2*cm, 2.0*cm]
    story.append(make_table(headers_conv, rows_conv, s, col_widths=cw_conv))
    story.append(P("Table 5. Selected convergence plateau data: event index of first minimum, "
                   "total events, and percentage of budget remaining unused. "
                   "Most instances plateau within 5% of the budget.",
                   s["caption"]))

    return story


def sec_results_per_dataset(styles) -> list:
    s = styles
    story = []
    story.append(H2("7.5  Per-Dataset Extended Dynamics and Rolling Correlation Analysis", s))
    story.append(B(
        "This section goes through each instance individually, using the 10,000-event clash "
        "trajectory plots and the rolling Pearson r plots (window W = 20). "
        "All 12 extended dynamics plots are in Figure 11; "
        "all 12 rolling correlation plots in Figure 12.",
        s,
    ))

    # Grid of all 12 extended dynamics plots
    instances = ["car-f-92", "car-s-91", "ear-f-83", "hec-s-92",
                 "kfu-s-93", "lse-f-91", "rye-s-93", "sta-f-83",
                 "tre-s-92", "uta-s-92", "ute-s-92", "yor-f-83"]

    ext_dyn_items = [
        (EXT_DYN / "plots" / f"{inst}_extended.png",
         f"{inst}: extended dynamics")
        for inst in instances
    ]
    g1 = grid_images(ext_dyn_items, s, cols=2)
    story.append(g1)
    story.append(P("Figure 11. Extended 10,000-event clash dynamics for all 12 instances "
                   "(top: blue = dynamic-only, orange = hybrid). "
                   "Vertical dotted lines mark first-zero-clash event.",
                   s["caption"]))

    # Grid of all 12 rolling plots — 2 columns for larger, more readable images
    roll_items = [
        (EXT_DYN / "rolling_plots" / f"{inst}_extended_rolling.png",
         f"{inst}: rolling r")
        for inst in instances
    ]
    g2 = grid_images(roll_items, s, cols=2)
    story.append(g2)
    story.append(P("Figure 12. Rolling Pearson r (W = 20) between dynamic-only and hybrid "
                   "clash trajectories for all 12 instances. "
                   "Near-zero r values after the initial descent confirm trajectory divergence.",
                   s["caption"]))

    # Overview rolling correlation
    im_ov = img(EXT_DYN / "rolling_plots" / "extended_rolling_overview.png",
                width=TEXT_W * 0.95, max_height=7*cm)
    if im_ov:
        story.append(im_ov)
        story.append(P("Figure 13. Mean rolling Pearson r per instance across all defined windows. "
                       "Green = moderately or strongly correlated; "
                       "red = anti-correlated or diverging. "
                       "Most instances cluster near zero, indicating independent trajectories.",
                       s["caption"]))

    # Per-dataset text analysis
    story.append(H3("car-f-92 (543 exams, S_min = 28)", s))
    story.append(B(
        "car-f-92 is the second largest instance. Dynamic-only achieves 120 clashes vs hybrid's 124, "
        "so dynamic wins here by a small margin. "
        "The dynamic run is interesting because it does not find its minimum until event 8,088, "
        "later than any other instance. It seems to need a long exploration phase before settling. "
        "Mean rolling r = -0.009, essentially zero. After the initial rapid drop, both trajectories "
        "oscillate independently. The Hopfield repair appears to interrupt the dynamic's descent here.",
        s,
    ))

    story.append(H3("car-s-91 (682 exams, S_min = 28)", s))
    story.append(B(
        "car-s-91 is the largest instance and shows the biggest hybrid advantage: "
        "186 clashes vs 328 for dynamic-only (delta = +142). "
        "Both methods plateau early, at event 202 for hybrid and 473 for dynamic, "
        "leaving over 98% of the budget unused. "
        "Mean rolling r = +0.019 (weakly correlated). "
        "The two methods explore similar regions of state space, "
        "but the Hopfield repair helps find a much better configuration early in the run.",
        s,
    ))

    story.append(H3("ear-f-83 (190 exams, S_min = 22)", s))
    story.append(B(
        "ear-f-83 is moderately dense with sensitivity score S = 0.328. "
        "Hybrid achieves 8 clashes vs dynamic's 16 (delta = +8). "
        "Mean rolling r = -0.246, the strongest anti-correlation in the dataset. "
        "When the dynamic trajectory rises, the hybrid tends to fall, and vice versa. "
        "The Hopfield repair is effective here because it pulls in a genuinely different direction "
        "from the dynamic operator.",
        s,
    ))

    story.append(H3("hec-s-92 (81 exams, S_min = 17)", s))
    story.append(B(
        "hec-s-92 is the densest instance (42% conflict density) and has sensitivity S = 1.0. "
        "Both methods reach 2 clashes but neither finds zero. "
        "S_min = 17 equals the chromatic number, so zero-clash colorings exist but are "
        "extremely rare in the 17^81 state space. "
        "After event 777, the dynamic run enters a repeating oscillation between 2 and about 30 clashes. "
        "Every time it fixes the last 2 conflicts, new ones appear elsewhere. "
        "Mean rolling r = +0.003: both methods are stuck in the same attractor region "
        "but fluctuating independently.",
        s,
    ))

    story.append(H3("kfu-s-93 (461 exams, S_min = 19)", s))
    story.append(B(
        "kfu-s-93 is large but sparse (5.6% conflict density). "
        "Both methods end at 16 clashes; mean rolling r = +0.009. "
        "Unusually, the dynamic run hits its minimum at event t = 3, basically immediately. "
        "The entire 10,000-event budget produces no improvement at all. "
        "The system falls into a deep local minimum right from the start and cannot escape. "
        "The Hopfield repair does not help here either.",
        s,
    ))

    story.append(H3("lse-f-91 (381 exams, S_min = 17)", s))
    story.append(B(
        "lse-f-91 is a sensitive instance (S = 0.556) with a low-density conflict graph. "
        "Dynamic achieves 12 clashes vs hybrid's 14, one of only two instances where dynamic wins. "
        "The dynamic run reaches its minimum at event 2,679, the second latest after car-f-92. "
        "Mean rolling r = +0.013. "
        "The Hopfield repair calls seem to interrupt a productive descent on this instance, "
        "leading to slightly worse results.",
        s,
    ))

    story.append(H3("rye-s-93 (486 exams, S_min = 21)", s))
    story.append(B(
        "rye-s-93 is the most unusual result: the hybrid keeps improving until event 6,684, "
        "the only instance where meaningful progress continues well past the 5% mark. "
        "Both methods end at 18 clashes (tie). Mean rolling r = -0.023. "
        "Under standard timeslots, dynamic-only solves this in 836 steps. "
        "At minimum timeslots it is qualitatively harder, "
        "and the extended budget genuinely helps the hybrid here.",
        s,
    ))

    story.append(H3("sta-f-83 (139 exams, S_min = 13)", s))
    story.append(B(
        "sta-f-83 is one of two instances where both methods find zero clashes at minimum timeslots. "
        "The dynamic run finishes in 9 events; hybrid in 15. "
        "Mean rolling r = +0.776, but this is based on only 9 common events with a window of just 5. "
        "The high correlation simply reflects that both paths are nearly identical "
        "when the run is this short.",
        s,
    ))

    story.append(H3("tre-s-92 (261 exams, S_min = 20)", s))
    story.append(B(
        "tre-s-92 shows a large hybrid advantage: 22 vs 42 clashes (delta = +20). "
        "Both methods plateau very quickly: dynamic at event 5, hybrid at event 51. "
        "The warm-up patterns in the Hopfield network capture useful templates for this instance, "
        "and the repair step exploits them early. "
        "Mean rolling r = +0.008. After the plateau, both trajectories oscillate with "
        "uncorrelated fluctuations.",
        s,
    ))

    story.append(H3("uta-s-92 (622 exams, S_min = 30)", s))
    story.append(B(
        "uta-s-92 shows the second largest hybrid advantage: 52 vs 92 clashes (delta = +40). "
        "Both methods reach their minimum very early (dynamic at t = 5, hybrid at t = 123). "
        "Mean rolling r = -0.006. "
        "This is a large, sparse instance where the Hopfield's global pattern-matching "
        "substantially outperforms the purely local shift. "
        "After the plateau, both methods oscillate in statistically independent regions.",
        s,
    ))

    story.append(H3("ute-s-92 (184 exams, S_min = 10)", s))
    story.append(B(
        "ute-s-92 is the other instance where both methods reach zero clashes at minimum timeslots. "
        "S_min = 10 is already the minimum here, so standard and minimum experiments are the same. "
        "Dynamic reaches zero at event 71 out of 72; hybrid at event 106 out of 107. "
        "Mean rolling r = +0.097. "
        "Together with sta-f-83, this confirms that two Toronto instances are genuinely solvable "
        "at their chromatic number within a small event budget.",
        s,
    ))

    story.append(H3("yor-f-83 (181 exams, S_min = 19)", s))
    story.append(B(
        "yor-f-83 has high conflict density (28.9%) and sensitivity S = 0.544. "
        "Hybrid achieves 40 clashes vs dynamic's 50 (delta = +10). "
        "Mean rolling r = -0.008. Both methods plateau early (dynamic at t = 391, hybrid at t = 434). "
        "The anti-correlation of the trajectories suggests the two methods are landing in "
        "different but similarly valued local minima.",
        s,
    ))

    return story


def sec_discussion(styles) -> list:
    s = styles
    story = []
    story.append(H1("8. Discussion", s))

    story.append(H2("8.1  Where the Hybrid Approach Helps", s))
    story.append(B(
        "The hybrid gives the biggest gains on large, moderately hard instances: "
        "car-s-91 (682 exams, delta = 142), uta-s-92 (622 exams, delta = 40), "
        "and tre-s-92 (261 exams, delta = 20). "
        "These instances share a few things in common: they are large enough that the dynamic-only "
        "operator can get stuck in shallow local minima; their conflict densities are moderate "
        "(12-18%), which gives the Hopfield network useful structure to learn; and they are not "
        "right at the chromatic boundary, so there is enough room for the repair step to help.",
        s,
    ))

    story.append(H2("8.2  Where Clashes Remain", s))
    story.append(B(
        "Ten instances still have unresolved clashes at minimum timeslots after 10,000 events. "
        "The common problem is that S_min is at or near the chromatic number chi(G), "
        "so valid colorings are exponentially rare. "
        "The local E operator cannot coordinate the global reassignment "
        "needed to put all exams into a proper coloring when every slot must be an independent set.",
        s,
    ))
    story.append(B(
        "hec-s-92 is the clearest example: chi(G) = 17, graph density 42%, average degree 33.7. "
        "The system locks into an oscillation between 2 and 30 clashes by event 777. "
        "Every time it fixes the last 2 conflicts, new ones appear elsewhere. "
        "Only a complete combinatorial search like backtracking could reliably find "
        "a valid coloring here.",
        s,
    ))

    story.append(H2("8.3  Effect of Minimum Timeslot Constraints", s))
    story.append(B(
        "Tightening S from standard to paper-minimum makes things much harder, "
        "but not equally for all instances. "
        "sta-f-83 and ute-s-92 are barely affected: they were already at their minimum. "
        "car-s-91 and uta-s-92 show a large jump in difficulty: dynamic-only, which found zero clashes "
        "under standard timeslots, gets stuck at 328 and 92 clashes respectively. "
        "The hybrid reduces these to 186 and 52. "
        "This suggests the Hopfield repair's global moves become more valuable "
        "when the constraint density is higher.",
        s,
    ))

    story.append(H2("8.4  Cyclic Dynamics and Nonlinear Behavior", s))
    story.append(B(
        "The propagating-particle system behaves like a discrete nonlinear dynamical system. "
        "The mode parameter m controls the orbit structure on the cyclic state space. "
        "When m is coprime with S, the orbit visits all S slots. "
        "When m divides S, it is trapped in a smaller subgroup. "
        "This parallels the Chirikov map's transition from regular to chaotic behavior [13]. "
        "The sensitivity analysis confirms that seven instances are in a chaotic regime "
        "and five are in a frozen regime with a single dominant attractor.",
        s,
    ))
    story.append(B(
        "The limit-cycle behavior in hec-s-92 and similar instances means the system has "
        "reached a stationary distribution over a set of near-optimal states. "
        "The cycle period is roughly 18 events in hec-s-92, reflecting the periodic "
        "structure imposed by the mode parameter and the cyclic slot space.",
        s,
    ))

    story.append(H2("8.5  Rolling Correlation Analysis: Trajectory Divergence", s))
    story.append(B(
        "The near-zero mean rolling r across almost all instances has an important implication. "
        "If you compute the global Pearson r between the two full trajectories, "
        "you typically get 0.98-1.0, which looks like strong similarity. "
        "But this is misleading: the high global r comes from the shared steep descent "
        "in the first few hundred events. The rolling r strips that away and shows what happens after. "
        "Once both methods reach their plateaus, they fluctuate independently in different "
        "parts of the energy landscape.",
        s,
    ))

    headers_roll = ["Instance", "Min Slots", "W", "Mean r", "Dyn Min", "Hyb Min", "Interpretation"]
    rows_roll = [
        ["car-f-92", "28", "20", "−0.009", "120", "124", "anti-correlated; dynamic better"],
        ["car-s-91", "28", "20", "+0.019", "328", "186", "weakly correlated; hybrid better"],
        ["ear-f-83", "22", "20", "−0.246", "16",  "8",   "anti-correlated; hybrid better"],
        ["hec-s-92", "17", "20", "+0.003", "2",   "2",   "weakly correlated; tie"],
        ["kfu-s-93", "19", "20", "+0.009", "16",  "16",  "weakly correlated; tie"],
        ["lse-f-91", "17", "20", "+0.013", "12",  "14",  "weakly correlated; dynamic better"],
        ["rye-s-93", "21", "20", "−0.023", "18",  "18",  "anti-correlated; tie"],
        ["sta-f-83", "13", "5",  "+0.776", "0",   "0",   "strongly correlated (n=9)"],
        ["tre-s-92", "20", "20", "+0.008", "42",  "22",  "weakly correlated; hybrid better"],
        ["uta-s-92", "30", "20", "−0.006", "92",  "52",  "anti-correlated; hybrid better"],
        ["ute-s-92", "10", "18", "+0.097", "0",   "0",   "weakly correlated; tie (both 0)"],
        ["yor-f-83", "19", "20", "−0.008", "50",  "40",  "anti-correlated; hybrid better"],
    ]
    cw_roll = [2.2*cm, 1.6*cm, 0.8*cm, 1.4*cm, 1.5*cm, 1.5*cm, 5.5*cm]
    story.append(make_table(headers_roll, rows_roll, s, col_widths=cw_roll))
    story.append(P("Table 6. Rolling correlation summary (10,000-event extended runs). "
                   "W = sliding window width; Mean r = average over all valid windows. "
                   "sta-f-83 has only 9 events, so its mean r is not representative.",
                   s["caption"]))

    story.append(H2("8.6  Limitations", s))
    story.append(B(
        "The extended runs use a single seed (seed 0). Variability across seeds was not "
        "characterized for the extended experiment, so the convergence results are "
        "per-seed trajectories rather than averages.",
        s,
    ))
    story.append(B(
        "The best mode m and ordering from the standard experiment were reused for "
        "extended runs. A jointly optimal configuration might differ under "
        "minimum-timeslot constraints, but re-running the full parameter sweep at "
        "10,000 events per combination would be computationally prohibitive.",
        s,
    ))
    story.append(B(
        "The Hopfield capacity was fixed at 0.14N patterns throughout. How varying "
        "this threshold affects convergence at minimum timeslots was not studied.",
        s,
    ))
    story.append(B(
        "The evaluation only considers hard constraints: zero clashes. Soft constraints "
        "such as exam spread, student rest time, or room capacity are not modeled.",
        s,
    ))
    story.append(B(
        "The rolling Pearson r window was chosen adaptively but tested at a single scale. "
        "Near-zero mean r indicates independence at that scale, but does not rule out "
        "correlation at larger or smaller windows.",
        s,
    ))
    story.append(B(
        "Neither method uses restarts or explicit diversification. The results reflect "
        "single-trajectory behavior starting from the same initial state.",
        s,
    ))

    return story


def sec_conclusion(styles) -> list:
    s = styles
    story = []
    story.append(H1("9. Conclusion", s))
    story.append(B(
        "This paper presented a hybrid exam timetabling algorithm that combines a "
        "propagating-particle dynamic model with Hopfield network repair. "
        "I tested it on all 12 Toronto benchmark instances under both standard "
        "and paper-minimum timeslot constraints.",
        s,
    ))
    story.append(B(
        "Under standard timeslots, the dynamic-only method is highly effective, solving "
        "10 of 12 instances to zero clashes. The Hopfield repair step provides the most "
        "benefit in combination with the dynamic component on the harder instances.",
        s,
    ))
    story.append(B(
        "Under paper-minimum timeslots, the hybrid wins on 5 of 12 instances, with the "
        "largest reductions in minimum clash on car-s-91, uta-s-92, and tre-s-92. "
        "Dynamic-only wins on 2 instances and the remaining 5 are ties.",
        s,
    ))
    story.append(B(
        "Extending to 10,000 events provides limited benefit. Nine of the ten "
        "non-trivial instances plateau within the first 5% of the event budget, "
        "and neither method finds new zero-clash solutions at minimum timeslots "
        "that were not already found in shorter runs.",
        s,
    ))
    story.append(B(
        "Rolling Pearson correlation shows that after the initial shared descent, "
        "the two methods explore statistically independent parts of the search "
        "landscape, with a mean rolling r of approximately ±0.02 across all instances.",
        s,
    ))
    story.append(B(
        "Sensitivity analysis identifies 7 instances in a chaotic regime where small "
        "changes to the mode or drive parameter produce large changes in the outcome. "
        "Five instances are in a frozen regime and are solved trivially by either method.",
        s,
    ))
    story.append(B(
        "The rollback safeguard proves essential: without it, Hopfield calls that "
        "worsen the schedule would destabilize the descent. The capacity limit at "
        "0.14N is sufficient to prevent spurious attractors from dominating recall.",
        s,
    ))
    story.append(B(
        "The propagating-particle framework provides a physically interpretable view "
        "of timetabling, with mode-dependent orbit structures, ergodic or trapped "
        "dynamics depending on gcd(m, S), and limit-cycle attractors at low-clash states.",
        s,
    ))

    story.append(B(
        "The main take-away is that the hybrid is most useful as an escape mechanism "
        "for large, moderately hard instances where the dynamic walk gets stuck. "
        "On the hardest instances at the chromatic number, neither method finds a feasible "
        "solution in 10,000 events, and a fundamentally different strategy would be needed.",
        s,
    ))
    return story


def sec_future_work(styles) -> list:
    s = styles
    story = []
    story.append(H1("10. Future Work", s))
    story.append(B(
        "Several directions could improve on what was done here:",
        s,
    ))

    fw_items = [
        ("<b>Per-timeslot Hopfield networks.</b> Instead of one global network for the full "
         "timetable, a separate network per timeslot would focus on the independent-set "
         "constraint at each slot. This could improve recall precision and make better use "
         "of the capacity limit."),
        ("<b>Adaptive Hopfield call frequency.</b> The hybrid pattern is currently fixed. "
         "An adaptive version that calls Hopfield more often when improvement slows "
         "and less often during active descent would better balance the two mechanisms."),
        ("<b>Conflict-targeted moves.</b> The E operator currently processes all exams. "
         "Restricting it to only exams currently in conflict would reduce wasted steps "
         "and focus effort where it is needed."),
        ("<b>Restart strategies.</b> When the algorithm plateaus early, "
         "restarting with a random perturbation or a different mode m could help escape "
         "limit-cycle attractors. Multi-start with best-of-N is a standard technique."),
        ("<b>Adaptive stopping.</b> Since most instances plateau within 5% of the budget, "
         "an adaptive criterion based on the rolling minimum could stop early "
         "when no progress is being made, while still allowing longer runs "
         "on instances like rye-s-93 that benefit from them."),
        ("<b>Better Hopfield capacity handling.</b> "
         "Using orthogonal pattern storage (e.g., pseudo-inverse learning rules) "
         "would let more patterns be stored without cross-talk. "
         "This could help on large instances where the capacity fills up quickly."),
        ("<b>Theoretical study of periodicity and chaos.</b> "
         "The sensitivity results give empirical evidence for chaotic behavior. "
         "A theoretical analysis of when the system transitions from regular to chaotic dynamics, "
         "as a function of N, S, density, and m, would give principled guidance for "
         "parameter selection."),
        ("<b>Soft constraint integration.</b> "
         "Adding soft constraints like exam spread and student load balance "
         "would make the algorithm applicable to the ITC-2007 benchmark "
         "and real university deployments."),
        ("<b>Dataset-specific tuning.</b> "
         "Since seven instances are sensitive to driving parameters, "
         "even a short tuning phase before the main run could give meaningful improvement "
         "over fixed parameter settings."),
    ]

    for item in fw_items:
        story.append(P(f"• {item}",
                       ParagraphStyle("bull3", parent=s["body"],
                                      leftIndent=0.8*cm, spaceAfter=6)))
    return story



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


def sec_references(styles) -> list:
    s = styles
    story = []
    story.append(H1("References", s))
    refs = [
        "[1] Carter, M. W., Laporte, G., & Lee, S. Y. (1996). Examination timetabling: "
        "Algorithmic strategies and applications. <i>Journal of the Operational Research Society</i>, "
        "47(3), 373–383.",

        "[2] Carter, M. W. (1986). A survey of practical applications of examination timetabling "
        "algorithms. <i>Operations Research</i>, 34(2), 193–202.",

        "[3] Even, S., Itai, A., & Shamir, A. (1976). On the complexity of timetable and "
        "multicommodity flow problems. <i>SIAM Journal on Computing</i>, 5(4), 691–703.",

        "[4] Garey, M. R., & Johnson, D. S. (1979). <i>Computers and Intractability: A Guide to "
        "the Theory of NP-Completeness.</i> W. H. Freeman, San Francisco.",

        "[5] Amit, D. J., Gutfreund, H., & Sompolinsky, H. (1985). Storing infinite numbers of "
        "patterns in a spin-glass model of neural networks. "
        "<i>Physical Review Letters</i>, 55(14), 1530–1533.",

        "[6] Kriesel, D. (2007). <i>A Brief Introduction to Neural Networks.</i> "
        "Available at: http://www.dkriesel.com.",

        "[7] Qu, R., Burke, E. K., McCollum, B., Merlot, L. T. G., & Lee, S. Y. (2009). "
        "A survey of search methodologies and automated system development for "
        "examination timetabling. "
        "<i>Journal of Scheduling</i>, 12(1), 55–89.",

        "[8] Burke, E. K., & Petrovic, S. (2002). Recent research directions in automated "
        "timetabling. <i>European Journal of Operational Research</i>, 140(2), 266–280.",

        "[9] McCollum, B., McMullan, P., Burke, E. K., Pearson, A. J., & Qu, R. (2012). "
        "A new model for automated examination timetabling. "
        "<i>Annals of Operations Research</i>, 194(1), 291–315.",

        "[10] Hopfield, J. J. (1982). Neural networks and physical systems with emergent "
        "collective computational abilities. "
        "<i>Proceedings of the National Academy of Sciences</i>, 79(8), 2554–2558.",

        "[11] Hebb, D. O. (1949). <i>The Organization of Behavior: A Neuropsychological Theory.</i> "
        "Wiley, New York.",

        "[12] Hopfield, J. J., & Tank, D. W. (1985). Neural computation of decisions in "
        "optimization problems. <i>Biological Cybernetics</i>, 52(3), 141–152.",

        "[13] Chirikov, B. V. (1979). A universal instability of many-dimensional oscillator "
        "systems. <i>Physics Reports</i>, 52(5), 263–379.",

        "[14] Strogatz, S. H. (2001). <i>Nonlinear Dynamics and Chaos: With Applications to "
        "Physics, Biology, Chemistry, and Engineering.</i> Westview Press, Cambridge, MA.",

        "[15] Jensen, T. R., & Toft, B. (1995). <i>Graph Coloring Problems.</i> "
        "Wiley-Interscience, New York.",

        "[16] Welsh, D. J. A., & Powell, M. B. (1967). An upper bound for the chromatic number "
        "of a graph and its application to timetabling problems. "
        "<i>The Computer Journal</i>, 10(1), 85–86.",

        "[17] Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., Gruber, L., "
        "Holzleitner, M., Pavlovic, M., Sandve, G. K., Greiff, V., Kreil, D., Kopp, M., "
        "Klambauer, G., Brandstetter, J., & Hochreiter, S. (2021). "
        "Hopfield networks is all you need. "
        "<i>International Conference on Learning Representations (ICLR 2021).</i>",

        "[18] Cappart, Q., Chételat, D., Khalil, E. B., Lodi, A., Morris, C., & "
        "Velickovic, P. (2023). Combinatorial optimization and reasoning with graph "
        "neural networks. "
        "<i>Journal of Machine Learning Research</i>, 24(130), 1–61.",
    ]
    for ref in refs:
        story.append(P(ref, s["ref"]))
        story.append(sp(2))
    return story


# ---------------------------------------------------------------------------
# Main: assemble and build
# ---------------------------------------------------------------------------

def main():
    styles = build_styles()

    story = []
    story += sec_title_page(styles)   # page 1: KMV-style title (no abstract)
    story += sec_abstract(styles)     # page 2: abstract + keywords
    story += sec_intro(styles)
    story.append(PageBreak())
    story += sec_literature(styles)
    story.append(PageBreak())
    story += sec_problem_def(styles)
    story.append(PageBreak())
    story += sec_methodology(styles)
    story.append(PageBreak())
    story += sec_implementation(styles)
    story += sec_experimental_setup(styles)
    story.append(PageBreak())
    story += sec_results_overview(styles)
    story.append(PageBreak())
    story += sec_results_per_dataset(styles)
    story.append(PageBreak())
    story += sec_discussion(styles)
    story.append(PageBreak())
    story += sec_conclusion(styles)
    story += sec_future_work(styles)
    story.append(PageBreak())
    story += sec_acknowledgments(styles)
    story.append(PageBreak())
    story += sec_references(styles)

    doc = BaseDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=L_MARGIN,
        rightMargin=R_MARGIN,
        topMargin=T_MARGIN,
        bottomMargin=B_MARGIN,
        title="Clash-Driven Propagation with Hopfield Network Repair for Exam Timetabling Optimization",
        author="Arpine Tadevosyan",
        subject="Capstone Paper, Computer Science",
    )

    # Title frame with raised bottom to leave room for canvas-drawn submission text
    title_b_margin = 5.0 * cm
    title_frame = Frame(L_MARGIN, title_b_margin, TEXT_W,
                        PAGE_H - T_MARGIN - title_b_margin,
                        id="title_frame")
    body_frame  = Frame(L_MARGIN, B_MARGIN, TEXT_W, PAGE_H - T_MARGIN - B_MARGIN - 0.8*cm,
                        id="body_frame")

    title_template = PageTemplate(id="TitlePage", frames=[title_frame], onPage=first_page)
    body_template  = PageTemplate(id="BodyPages", frames=[body_frame],  onPage=later_pages)

    doc.addPageTemplates([title_template, body_template])

    # Switch to body template after title page
    story.insert(0, NextPageTemplate("BodyPages"))

    print(f"Building PDF: {PDF_PATH}")
    doc.build(story)
    print(f"Done. PDF written to: {PDF_PATH}")


if __name__ == "__main__":
    main()
