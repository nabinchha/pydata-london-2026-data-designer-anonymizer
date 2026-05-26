"""Prompt templates and judge rubrics for the workshop notebooks.

Notebook 1 (text QA) uses QUESTION_PROMPT, ANSWER_PROMPT, JUDGE_PROMPT,
and the JUDGE_* score descriptions/options.

Notebook 2 (visual QA) uses VQA_QUESTION_PROMPT, VQA_ANSWER_PROMPT,
VQA_JUDGE_PROMPT, and VQA_FAILURE_REVIEW_PROMPT.
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
- **factual:** a specific fact stated in the excerpt.
- **definition:** asks the meaning of a term used in the excerpt.
- **inference:** the answer is implied but not directly stated.
- **comparison:** compares two entities or concepts mentioned in the excerpt.

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
1. Answer using ONLY the information in the excerpt.
2. If the excerpt does not contain enough information to answer, respond exactly with "Insufficient information".
3. Provide the answer first, then a one-sentence justification quoting or paraphrasing the supporting evidence.
""".strip()


JUDGE_PROMPT = """\
You are an expert evaluator scoring a reading-comprehension question-answer pair.

**Article excerpt:**
{{ article }}

**Question:** {{ question }}

**Answer:** {{ answer }}

Score this pair on the rubrics below. Be strict -- a question that leaks the
answer or an answer not supported by the excerpt is a bad training example.
""".strip()


JUDGE_FAITHFULNESS_DESCRIPTION = (
    "Is the answer fully supported by the article excerpt? "
    "An answer that introduces facts not present in the excerpt, or contradicts it, "
    "should score low. An answer that quotes or paraphrases evidence from the excerpt "
    "scores high."
)

JUDGE_FAITHFULNESS_OPTIONS = {
    "1": "Contradicts the excerpt or makes up facts not present.",
    "2": "Mostly unsupported -- introduces material not in the excerpt.",
    "3": "Partially supported -- some claims tracked to the excerpt, others not.",
    "4": "Well supported -- all claims trace back to the excerpt with minor wording differences.",
    "5": "Exactly supported -- every claim is directly verifiable in the excerpt.",
}

JUDGE_COMPLETENESS_DESCRIPTION = (
    "Does the answer fully address the question? "
    "A complete answer covers all aspects of the question with appropriate depth. "
    "An incomplete answer omits a relevant entity, date, or condition."
)

JUDGE_COMPLETENESS_OPTIONS = {
    "1": "Does not address the question.",
    "2": "Addresses the question only partially; key information missing.",
    "3": "Addresses the question but omits a relevant detail.",
    "4": "Fully addresses the question with all relevant details.",
    "5": "Fully addresses the question and includes useful context.",
}


# ─── Visual QA prompts (Notebook 2) ─────────────────────────────────────────

VQA_FAILURE_REVIEW_PROMPT = """\
You are reviewing a visual QA pair that failed the multimodal judge.
Examine the business document image and write a concise debugging note.

**Question type:** {{ question_type }}
**Effective focus area:** {{ question.effective_focus_area }}
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

You will see the document image. The intended question category is below.

**Question type:** {{ question_type }}
**Requested focus area:** {{ focus_area }}

**Available visual cues from seed metadata:**
- primary_visual: {{ primary_visual }}
- secondary_visual: {{ secondary_visual }}
- layout_style: {{ layout_style }}
- annotation_layer: {{ annotation_layer }}
- numeric_context: {{ numeric_context }}

**Effective focus-area selection:**
Choose `effective_focus_area` from the labels below. Prefer the requested focus
area when the visual cues indicate that target is present; otherwise fall back
to the closest valid label.

- Use `table_lookup` only when the visual cues mention a table, risk register,
  checklist, matrix, or action table.
- Use `kpi_status` only when the visual cues mention KPI cards, status chips,
  traffic-light status, target values, thresholds, or deltas.
- Use `annotation_and_callout` only when the annotation layer is not
  "no manual annotations" or the visual cues mention callouts, highlights,
  stamps, sticky notes, or handwritten notes.
- `chart_reading`, `cross_panel_reasoning`, and `layout_and_structure` are safe
  fallbacks for any row with charts, panels, or page structure.

Set the structured `effective_focus_area` field to the exact label you used.

**Focus area guidance:**
- chart_reading: Ask about a chart value, axis label, legend category, trend, outlier, or comparison.
- table_lookup: Ask about a specific row/column value, subtotal, owner, due date, or status in a table.
- kpi_status: Ask about a KPI card, traffic-light status, target, delta, or threshold.
- annotation_and_callout: Ask about a highlight, handwritten note, stamp, sticky note, footnote, or callout box.
- cross_panel_reasoning: Ask a question that requires combining information from two or more visible sections.
- layout_and_structure: Ask about spatial arrangement, panel order, sidebar placement, or where a chart/table appears relative to another element.

**Your task:** Write ONE question about the document that:
1. Requires looking at the image (not just reading transcribed text).
2. Targets the effective focus area selected above.
3. Has a single, clearly correct answer visible on the document.
4. Matches the question type. For multiple-choice, list 4 options A-D on separate lines.

**Hard rules:**
- Do not mention "the image" or "the document" in the question -- ask directly as if the reader is looking at the page.
- Do not include the answer in the question.
- Avoid trivia like font size or color. Focus on data, layout, structure, or relationships visible on the page.
- If the focus area is not applicable to this particular page, pick the closest relevant area and write a question about that instead. Never return an empty response.

**Output:** Fill the structured object:
- text: the question text (and options for multiple choice)
- effective_focus_area: the exact effective_focus_area label selected above
- rationale: one sentence naming the visual element the question targets
""".strip()


VQA_ANSWER_PROMPT = """\
You are answering a question about a business document using ONLY what is visible in the image.

**Question type:** {{ question_type }}
**Effective focus area:** {{ question.effective_focus_area }}
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
**Effective focus area:** {{ question.effective_focus_area }}
**Question:** {{ question.text }}
**Answer:** {{ answer }}

Examine the document image and verify the answer yourself. A pair passes ONLY if
ALL of these are true:
1. The answer is factually correct given the visible content (±5% tolerance for numbers).
2. The question requires examining the page image, not just reading plain text.
3. The answer is NOT "Not present on this page" or equivalent.

Be strict -- bad pairs poison training data.
""".strip()
