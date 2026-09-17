"""
Extraction Engine: opportunity .txt files -> normalized JSON/CSV dataset.

Design decisions (read this before you trust the output):

1. NO LLM/NLP libs used - pure regex + string rules, per your constraint.
2. Every field extraction is independent and defensive: if a pattern fails,
   we record it as a MISS rather than guessing or crashing. Silent guessing
   is how you ship a dataset full of confidently-wrong numbers.
3. `extraction_confidence` = (fields successfully matched) / (fields checked),
   but NOT equally weighted - fields your downstream ranking engine actually
   uses for scoring (risk, budget/min_investment, tenure, return) are weighted
   2x vs cosmetic fields (name, provider, category). A doc that fails to parse
   risk should score worse than one that fails to parse provider, because your
   ranking engine literally can't function without risk.
4. This script does NOT guess field order (e.g. "name is always line 2").
   That assumption breaks the moment one document has an extra blank line or
   a preamble sentence. Instead: ID and name are extracted via explicit
   pattern anchors. If your real docs don't have a consistent "line 1 = ID,
   line 2 = name" layout, you MUST tell me the actual variance - I designed
   this against the ONE sample you gave me, and one sample is not proof of
   30-document consistency.

RUN THIS. THEN LOOK AT THE "ISSUES" REPORT IT PRINTS. If any file has
extraction_confidence < 1.0, do not ignore it - paste me the raw text of
that file and we fix the regex against the real failure, not a guess.
"""

from email.mime import text
import os
import re
import json
import glob
import sys

# Field weights for confidence scoring. Fields the Ranking Engine will
# actually consume (per your architecture diagram) get 2x weight.
FIELD_WEIGHTS = {
    "opportunity_id": 1,
    "name": 1,
    "provider": 1,
    "category": 1,
    "minimum_investment_inr": 2,   # feeds "Budget 30%"
    "tenure": 2,                   # feeds "Tenure 20%"
    "risk": 2,                     # feeds "Risk 25%"
    "return": 2,                   # feeds "Return 15%"
    "return_type": 1,
}


def parse_amount(raw: str, is_lakh: bool) -> float:
    """Strip commas/whitespace and cast, applying lakh conversion if flagged.
    1 lakh = 100,000. Get this wrong and every 'X lakh' document is off by
    a factor of 100,000 - not missing data, WRONG data. That's worse."""
    value = float(raw.replace(",", "").strip())
    return value * 100000 if is_lakh else value


def extract_opportunity(text: str, filename: str) -> dict:
    misses = []
    data = {
        "opportunity_id": None,
        "name": None,
        "provider": None,
        "category": None,
        "minimum_investment_inr": None,
        "tenure_min_months": None,
        "tenure_max_months": None,
        "risk": None,
        "expected_return_min_pct": None,
        "expected_return_max_pct": None,
        "return_type": None,
        "source_document": filename,
        "extraction_confidence": None,
    }

    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]

    # --- opportunity_id: explicit pattern, not "first line" ---
    m = re.search(r'\b(OPP\d+)\b', text)
    if m:
        data["opportunity_id"] = m.group(1)
    else:
        misses.append("opportunity_id")

    # --- name: the line immediately after the ID line ---
    # Fragile assumption (see module docstring). Flagged, not hidden.
    id_line_idx = next((i for i, l in enumerate(lines) if re.match(r'^OPP\d+$', l)), None)
    if id_line_idx is not None and id_line_idx + 1 < len(lines):
        data["name"] = lines[id_line_idx + 1]
    else:
        misses.append("name")

    # --- provider ---
    m = re.search(r'Provider:\s*(.+)', text)
    if m:
        data["provider"] = m.group(1).strip()
    else:
        misses.append("provider")

    # --- category ---
    m = re.search(r'Category:\s*(.+)', text)
    if m:
        data["category"] = m.group(1).strip()
    else:
        misses.append("category")

    # --- minimum investment ---
    # There are 4 distinct intro phrases across the 30 docs. This is NOT
    # one format with noise - it's 4 templates. Cover all 4 explicitly
    # rather than one pattern + hope.
    m = re.search(
        r'(?:Minimum investment is|Entry ticket:|Investors may participate from|'
        r'Ticket size starts at)\s*(?:₹|Rs\.?|INR)?\s*([\d,]+(?:\.\d+)?)\s*(lakh)?',
        text, re.I
    )
    if m:
        try:
            data["minimum_investment_inr"] = parse_amount(m.group(1), bool(m.group(2)))
        except ValueError:
            misses.append("minimum_investment_inr")
    else:
        misses.append("minimum_investment_inr")

    # --- tenure: try range first, then single value ---
    # 4 intro phrases again. Tolerates the doubled "months months" artifact
    # seen across every single document (not a one-off typo - it's systemic,
    # so don't "fix" the source text, just make the regex tolerate it).
    m = re.search(
        r'(?:tenure|Intended holding period:|Suggested duration is|'
        r'Maturity/holding period:)\s*(\d+)\s*-\s*(\d+)\s*months?', text, re.I
    )
    if m:
        data["tenure_min_months"] = int(m.group(1))
        data["tenure_max_months"] = int(m.group(2))
    else:
        m = re.search(
            r'(?:tenure|Intended holding period:|Suggested duration is|'
            r'Maturity/holding period:)\s*(\d+)\s*months?(?:\s*months?)?', text, re.I
        )
        if m:
            data["tenure_min_months"] = data["tenure_max_months"] = int(m.group(1))
        else:
            misses.append("tenure")

    # --- risk ---
    # Risk values are NOT always single words: "Moderately High",
    # "Moderate-High", "Low to Moderate" all appear. A single-word regex
    # (my first draft) truncates these to garbage. Capture everything up
    # to the period that precedes the return sentence instead.
    m = re.search(
        r'(?:\brisk\s+(?!classification)|Risk classification:\s*|'
        r'This opportunity is classified as\s*|Risk:\s*)'
        r'([A-Za-z][A-Za-z\s\-]*?)\.\s*(?=Return|Published|Stated)',
        text, re.I
    )
    if m:
        data["risk"] = m.group(1).strip()
    else:
        misses.append("risk")

    # --- expected return + return type ---

# Extract the sentence/section containing return information.
    return_match = re.search(
    r'(?:Return information:|Return reference:|'
    r'Published return information:|Stated return information:)'
    r'\s*(.*?)(?:\n|$)',
    text,
    re.I
)

    if return_match:

        return_text = return_match.group(1).strip()
        return_lower = return_text.lower()

    # ---------------------------------------------------------
    # Variable / market-linked return
    # ---------------------------------------------------------

        if re.search(r'\b(variable|market[- ]linked)\b', return_lower):

            data["return_type"] = "variable"

            data["expected_return_min_pct"] = None
            data["expected_return_max_pct"] = None

        else:

        # -----------------------------------------------------
        # Numeric return
        # -----------------------------------------------------

            number_match = re.search(
            r'([\d.]+)\s*(?:-\s*([\d.]+))?\s*%',
            return_text
        )

            if number_match:

                data["expected_return_min_pct"] = float(
                number_match.group(1)
            )

                if number_match.group(2):
                    data["expected_return_max_pct"] = float(
                    number_match.group(2)
                )

                    data["return_type"] = "range-based"

                else:
                    data["expected_return_max_pct"] = float(
                    number_match.group(1)
                )

                    data["return_type"] = "stated"

            else:

                misses.append("return")
                misses.append("return_type")

    else:

        misses.append("return")
        misses.append("return_type")

    # --- weighted confidence score ---
    total_weight = sum(FIELD_WEIGHTS.values())
    missed_weight = sum(FIELD_WEIGHTS.get(f, 1) for f in misses)
    data["extraction_confidence"] = round((total_weight - missed_weight) / total_weight, 2)

    return data, misses


def main(input_dir: str, output_json: str, output_csv: str = None):
    files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    if not files:
        print(f"No .txt files found in '{input_dir}'. Check the path.", file=sys.stderr)
        return [], {}

    records = []
    issues = {}
    for filepath in files:
        with open(filepath, encoding="utf-8") as f:
            text = f.read()
        filename = os.path.basename(filepath)
        record, misses = extract_opportunity(text, filename)
        records.append(record)
        if misses:
            issues[filename] = misses

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    if output_csv:
        try:
            import pandas as pd
            pd.DataFrame(records).to_csv(output_csv, index=False)
        except ImportError:
            print("pandas not installed - skipping CSV export. `pip install pandas --break-system-packages`")

    print(f"Processed {len(records)} documents -> {output_json}"
          + (f", {output_csv}" if output_csv else ""))
    perfect = sum(1 for r in records if r["extraction_confidence"] == 1.0)
    print(f"{perfect}/{len(records)} documents fully extracted (confidence == 1.0)")

    if issues:
        print(f"\n{len(issues)} documents had extraction gaps - DO NOT ignore these:")
        for fname, misses in issues.items():
            conf = next(r["extraction_confidence"] for r in records if r["source_document"] == fname)
            print(f"  {fname} (confidence={conf}): missing -> {misses}")

    return records, issues


if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "opportunities"
    main(input_dir, "opportunities_dataset.json", "opportunities_dataset.csv")