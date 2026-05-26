"""Prompt templates and judge rubrics for the workshop notebooks.

Notebook 1 (text QA) uses QUESTION_PROMPT, ANSWER_PROMPT, JUDGE_PROMPT,
and the JUDGE_* score descriptions/options.

Notebook 2 (visual QA) uses VQA_VISUAL_FOCUS_PROMPT, VQA_QUESTION_PROMPT,
VQA_ANSWER_PROMPT, VQA_JUDGE_PROMPT, and VQA_FAILURE_REVIEW_PROMPT.
"""

# ─── Text QA prompts (Notebook 1) ───────────────────────────────────────────

QUESTION_PROMPT = """\
You are an expert at writing reading-comprehension questions grounded in a specific text.

**Article excerpt:**
{{ article }}

**Question difficulty:** {{ question_difficulty }}
**Question type:** {{ question_type }}

**Your task:** Write ONE question about the article excerpt that matches the difficulty and type.

**Difficulty guidance:**
- **easy:** answerable directly from a single sentence in the excerpt.
- **medium:** requires synthesising information from two or more sentences.
- **hard:** requires inference, comparison, or careful reading; a casual reader might miss it.

**Question type guidance:**
- **fact_lookup:** asks for one explicit detail from one sentence.
- **definition:** asks the meaning of a term or concept used in the excerpt.
- **relationship:** asks how two stated ideas or entities connect.
- **evidence_summary:** asks the reader to combine two or more details into a concise answer.
- **inference:** the answer is implied but not directly stated.
- **comparison:** compares or contrasts two entities or concepts mentioned in the excerpt.

**Rules:**
1. The question MUST be answerable using ONLY the excerpt above.
2. Do not include the answer in the question.
3. Do not reference "the article" or "the excerpt" -- ask the question directly.
4. Keep it concise (1-2 sentences).

**Output:** Return ONLY the question text, nothing else.
""".strip()


ANSWER_PROMPT = """\
You are answering a reading-comprehension question grounded in a specific excerpt.

**Article excerpt:**
{{ article }}

**Question:** {{ question }}

**Instructions:**
1. Use ONLY the information in the excerpt.
2. If the excerpt does not contain enough information to answer, respond with "Insufficient information".
3. Keep the answer concise and ground the justification in a quote or paraphrase from the excerpt.
""".strip()


JUDGE_PROMPT = """\
You are an expert evaluator scoring a reading-comprehension question-answer pair.

**Article excerpt:**
{{ article }}

**Question:** {{ question }}

**Answer:** {{ answer.text }}

Score this pair on the rubrics below. Be strict -- a question that leaks the
answer or an answer not supported by the excerpt is a bad training example.
""".strip()


JUDGE_FAITHFULNESS_DESCRIPTION = (
    "Is the answer fully supported by the article excerpt? "
    "An answer that introduces facts not present in the excerpt, contradicts it, "
    "or is too vague to verify should score low."
)

JUDGE_FAITHFULNESS_OPTIONS = {
    "1": "Contradicts the excerpt or makes up unsupported facts.",
    "2": "Mostly unsupported -- introduces material not present in the excerpt.",
    "3": "Partially supported -- some claims trace to the excerpt, others do not.",
    "4": "Well supported -- all claims trace to the excerpt with minor wording differences.",
    "5": "Exactly supported -- every claim is directly and completely verifiable in the excerpt.",
}

JUDGE_COMPLETENESS_DESCRIPTION = (
    "Does the answer fully address the question? "
    "A complete answer covers the requested entities, values, dates, conditions, "
    "or relationships with appropriate depth. An incomplete answer omits a relevant "
    "detail or is too vague to be useful."
)

JUDGE_COMPLETENESS_OPTIONS = {
    "1": "Does not address the question.",
    "2": "Addresses the question only partially; key information missing.",
    "3": "Mostly addresses the question but omits a relevant detail.",
    "4": "Fully addresses the question with all relevant details.",
    "5": "Fully addresses the question and includes useful context from the excerpt.",
}


# ─── Visual QA prompts (Notebook 2) ─────────────────────────────────────────

VQA_VISUAL_FOCUS_PROMPT = """\
You are classifying which visual QA focus areas are supported by a business document page.

Use the image first, then use the seed metadata as orientation. Do not invent a
table, KPI card, annotation, or chart if it is not visible.

**Available visual cues from seed metadata:**
- primary_visual: {{ primary_visual }}
- secondary_visual: {{ secondary_visual }}
- layout_style: {{ layout_style }}
- document_condition: {{ document_condition }}

**Focus area labels:**
- chart_reading: charts, axes, legends, trends, outliers, or plotted comparisons.
- table_lookup: tables, risk registers, checklists, matrices, rows, columns, owners, due dates, or statuses.
- kpi_status: KPI cards, traffic-light status, target values, thresholds, or deltas.
- annotation_and_callout: highlights, handwritten notes, stamps, sticky notes, footnotes, or callout boxes.
- cross_panel_reasoning: two or more sections that can be combined into one answer.
- layout_and_structure: spatial arrangement, panel order, sidebar placement, or where one element appears relative to another.

Pick the single best `focus_area` for a high-quality QA pair. Also list every
clearly supported label in `available_focus_areas`.
""".strip()


VQA_FAILURE_REVIEW_PROMPT = """\
You are reviewing a visual QA pair that failed the multimodal judge.
Examine the business document image and write a concise debugging note.

**Question type:** {{ question_type }}
**Question difficulty:** {{ question_difficulty }}
**Focus area:** {{ visual_focus.focus_area }}
**Question:** {{ question.text }}
**Answer:** {{ answer }}
**Judge reason:** {{ judge.reason }}

**Write:**
- What visible evidence matters for this question
- Why the QA pair likely failed
- One concrete fix for the prompt, focus area, or answer

Keep it under 120 words. Do not include generic advice.
""".strip()


VQA_QUESTION_PROMPT = """\
You are an expert at writing visual reasoning questions grounded in a business document image.

You will see the document image. A prior VLM step has classified which visual
focus area this row should target.

**Question type:** {{ question_type }}
**Question difficulty:** {{ question_difficulty }}
**Focus area:** {{ visual_focus.focus_area }}
**Available focus areas:** {{ visual_focus.available_focus_areas }}
**Focus rationale:** {{ visual_focus.rationale }}

**Difficulty guidance:**
- easy: Ask for one directly visible value, label, status, or location.
- medium: Ask for a comparison, trend, or relationship visible within one panel.
- hard: Ask for reasoning across two visible sections or a concise synthesis of multiple cues.

**Focus area guidance:**
- chart_reading: Ask about a chart value, axis label, legend category, trend, outlier, or comparison.
- table_lookup: Ask about a specific row/column value, subtotal, owner, due date, or status in a table.
- kpi_status: Ask about a KPI card, traffic-light status, target, delta, or threshold.
- annotation_and_callout: Ask about a highlight, handwritten note, stamp, sticky note, footnote, or callout box.
- cross_panel_reasoning: Ask a question that requires combining information from two or more visible sections.
- layout_and_structure: Ask about spatial arrangement, panel order, sidebar placement, or where a chart/table appears relative to another element.

**Your task:** Write ONE question about the document that:
1. Requires looking at the image (not just reading transcribed text).
2. Targets the focus area selected above.
3. Has a single, clearly correct answer visible on the document.
4. Matches the question type. For multiple-choice, list 4 options A-D on separate lines.
5. Matches the requested difficulty.

**Hard rules:**
- Do not mention "the image" or "the document" in the question -- ask directly as if the reader is looking at the page.
- Do not include the answer in the question.
- Avoid trivia like font size or color. Focus on data, layout, structure, or relationships visible on the page.
- Use `available_focus_areas` as supporting context, but write the question for the selected focus area.
- Never return an empty response.

**Output:** Fill the structured object:
- text: the question text (and options for multiple choice)
- rationale: one sentence naming the visual element the question targets
""".strip()


VQA_ANSWER_PROMPT = """\
You are answering a question about a business document using ONLY what is visible in the image.

**Question type:** {{ question_type }}
**Question difficulty:** {{ question_difficulty }}
**Focus area:** {{ visual_focus.focus_area }}
**Question:** {{ question.text }}

**Instructions:**
1. Answer based on what is visible in the document image. Do not guess.
2. For numerical answers, include units, dates, percentages, labels, or currency symbols exactly as shown.
3. For multiple-choice questions, output the full chosen option (e.g., "B. $232.95").
4. For yes/no questions, answer exactly "Yes" or "No".
5. For list questions, output a JSON array of strings.
6. If the requested information is not present on the page, respond exactly with "Not present on this page".
7. Do not write meta-commentary like "Based on the image" or "Looking at the document".
""".strip()


VQA_JUDGE_PROMPT = """\
You are evaluating a (question, answer) pair generated from a business document image.
Decide whether the pair is suitable as training data for a vision-language model.

**Question type:** {{ question_type }}
**Question difficulty:** {{ question_difficulty }}
**Focus area:** {{ visual_focus.focus_area }}
**Question:** {{ question.text }}
**Answer:** {{ answer }}

Examine the document image and verify the answer yourself. A pair passes ONLY if
ALL of these are true:
1. The answer is factually correct given the visible content (±5% tolerance for numbers).
2. The question requires examining the page image, not just reading plain text.
3. The answer is NOT "Not present on this page" or equivalent.

Be strict -- bad pairs poison training data.
""".strip()
