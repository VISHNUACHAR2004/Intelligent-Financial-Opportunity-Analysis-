"""
Query Parser
============

Converts a user's natural-language financial requirement
into structured requirements for the retrieval/ranking engine.

Example:

Input:
    "I have ₹2 lakh, medium risk, and want something for around two years."

Output:
    {
        "budget": 200000,
        "risk": "medium",
        "tenure_months": 24,
        "return_preference": None,
        "query_text": "...",
        "parsed_fields": {
            "budget": True,
            "risk": True,
            "tenure": True,
            "return": False
        }
    }

Design principles:
1. Rule-based and deterministic.
2. Never invent missing values.
3. Supports INR, Rs., ₹ and lakh/lakhs.
4. Supports years and months.
5. Supports common risk descriptions.
6. Supports return preferences such as high return,
   low return, specific percentages and ranges.
7. Keeps the original query for TF-IDF/text retrieval.
"""

import re
from typing import Optional, Dict, Any


# ============================================================
# 1. TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize user query text while preserving the original
    meaning.

    Example:
        "  I HAVE ₹2 LAKH!!! "
        ->
        "i have ₹2 lakh"
    """

    text = text.strip().lower()

    # Normalize different dash characters.
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    # Normalize multiple spaces.
    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# 2. NUMBER WORD CONVERSION
# ============================================================

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
}


def word_to_number(word: str) -> Optional[float]:
    """
    Convert a simple number word to a numeric value.

    Examples:
        "two" -> 2
        "twenty" -> 20

    Returns None if the word is not recognized.
    """

    word = word.strip().lower()

    if word in NUMBER_WORDS:
        return float(NUMBER_WORDS[word])

    return None


# ============================================================
# 3. MONEY PARSING
# ============================================================

def parse_money_value(number_text: str, unit: Optional[str] = None) -> Optional[float]:
    """
    Convert an amount into INR.

    Examples:

        "200000"       -> 200000
        "2,00,000"     -> 200000
        "2 lakh"       -> 200000
        "2.5 lakh"     -> 250000
        "5 lakhs"      -> 500000
    """

    try:
        number = float(number_text.replace(",", "").strip())
    except ValueError:
        return None

    if unit:
        unit = unit.lower().strip()

        if unit in ("lakh", "lakhs"):
            number *= 100000

        elif unit in ("crore", "crores"):
            number *= 10000000

        elif unit in ("thousand", "k"):
            number *= 1000

    return number


def extract_budget(query: str) -> Optional[float]:
    """
    Extract the user's available investment budget.

    Supports:

        ₹2 lakh
        Rs. 2 lakh
        Rs 200000
        INR 200000
        2 lakh
        5 lakhs
        ₹2,00,000
        200000

    Also understands phrases such as:

        "I have 2 lakh"
        "budget is 5 lakh"
        "I can invest 3 lakhs"
        "I have ₹2,00,000 available"
    """

    # --------------------------------------------------------
    # Pattern 1:
    # Currency + number + lakh/crore
    #
    # ₹2 lakh
    # Rs 2 lakh
    # INR 2 lakh
    # --------------------------------------------------------

    pattern = re.search(
        r"(?:₹|rs\.?|inr)\s*"
        r"(\d+(?:,\d+)*(?:\.\d+)?)\s*"
        r"(lakh|lakhs|crore|crores)?",
        query,
        re.IGNORECASE
    )

    if pattern:
        value = parse_money_value(
            pattern.group(1),
            pattern.group(2)
        )

        if value is not None:
            return value

    # --------------------------------------------------------
    # Pattern 2:
    # Number + lakh/crore without currency symbol
    #
    # 2 lakh
    # 5 lakhs
    # 1.5 crore
    # --------------------------------------------------------

    pattern = re.search(
        r"\b(\d+(?:\.\d+)?)\s*"
        r"(lakh|lakhs|crore|crores)\b",
        query,
        re.IGNORECASE
    )

    if pattern:
        value = parse_money_value(
            pattern.group(1),
            pattern.group(2)
        )

        if value is not None:
            return value

    # --------------------------------------------------------
    # Pattern 3:
    # Currency + normal numeric amount
    #
    # ₹200000
    # ₹2,00,000
    # Rs. 200000
    # INR 200000
    # --------------------------------------------------------

    pattern = re.search(
        r"(?:₹|rs\.?|inr)\s*"
        r"(\d+(?:,\d+)*(?:\.\d+)?)",
        query,
        re.IGNORECASE
    )

    if pattern:
        value = parse_money_value(pattern.group(1))

        if value is not None:
            return value

    # --------------------------------------------------------
    # Pattern 4:
    # Number followed by INR
    #
    # 200000 INR
    # 2 lakh INR
    # --------------------------------------------------------

    pattern = re.search(
        r"\b(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:inr|rupees?)\b",
        query,
        re.IGNORECASE
    )

    if pattern:
        value = parse_money_value(pattern.group(1))

        if value is not None:
            return value

    return None


# ============================================================
# 4. RISK PARSING
# ============================================================

def extract_risk(query: str) -> Optional[str]:
    """
    Extract risk preference.

    Supported examples:

        low risk
        medium risk
        moderate risk
        high risk
        low to moderate risk
        moderate-high risk
        moderately high risk
        conservative
        aggressive

    Returns a normalized representation.
    """

    # More specific patterns must come BEFORE
    # generic low/medium/high patterns.

    risk_patterns = [
        (
            r"\blow\s*(?:-|to)\s*moderate\b",
            "low to moderate"
        ),
        (
            r"\bmoderate\s*(?:-|to)\s*high\b",
            "moderate to high"
        ),
        (
            r"\bmoderately\s+high\b",
            "moderately high"
        ),
        (
            r"\bmedium\s*(?:-|to)\s*high\b",
            "moderate to high"
        ),
        (
            r"\bmedium\s*(?:-|to)\s*low\b",
            "low to moderate"
        ),
        (
            r"\blow\s+risk\b",
            "low"
        ),
        (
            r"\bmedium\s+risk\b",
            "medium"
        ),
        (
            r"\bmoderate\s+risk\b",
            "moderate"
        ),
        (
            r"\bhigh\s+risk\b",
            "high"
        ),
        (
            r"\bconservative\b",
            "low"
        ),
        (
            r"\baggressive\b",
            "high"
        ),
    ]

    for pattern, normalized_value in risk_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return normalized_value

    return None


# ============================================================
# 5. TENURE PARSING
# ============================================================

def extract_tenure(query: str) -> Optional[int]:
    """
    Extract desired investment duration and normalize to months.

    Examples:

        "2 years"       -> 24
        "two years"     -> 24
        "24 months"     -> 24
        "around 2 years" -> 24
        "about 18 months" -> 18
        "3 yrs"         -> 36
    """

    # --------------------------------------------------------
    # Numeric years
    # --------------------------------------------------------

    pattern = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b",
        query,
        re.IGNORECASE
    )

    if pattern:
        years = float(pattern.group(1))
        return round(years * 12)

    # --------------------------------------------------------
    # Number-word years
    # --------------------------------------------------------

    pattern = re.search(
        r"\b("
        r"one|two|three|four|five|six|seven|eight|nine|ten"
        r")\s*(?:years?|yrs?)\b",
        query,
        re.IGNORECASE
    )

    if pattern:
        years = word_to_number(pattern.group(1))

        if years is not None:
            return round(years * 12)

    # --------------------------------------------------------
    # Numeric months
    # --------------------------------------------------------

    pattern = re.search(
        r"\b(\d+)\s*(?:months?|mos?)\b",
        query,
        re.IGNORECASE
    )

    if pattern:
        return int(pattern.group(1))

    # --------------------------------------------------------
    # Number-word months
    # --------------------------------------------------------

    pattern = re.search(
        r"\b("
        r"one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve"
        r")\s*(?:months?|mos?)\b",
        query,
        re.IGNORECASE
    )

    if pattern:
        months = word_to_number(pattern.group(1))

        if months is not None:
            return int(months)

    return None


# ============================================================
# 6. RETURN PREFERENCE
# ============================================================

def extract_return_preference(query: str) -> Optional[Dict[str, Any]]:
    """
    Extract return-related preferences.

    Examples:

        "high return"
        "better returns"
        "maximum return"
        "at least 10%"
        "return above 12%"
        "around 10%"
        "between 8 and 12%"

    Returns a dictionary.

    Example:

        {
            "type": "minimum",
            "value": 10
        }

    or:

        {
            "type": "range",
            "min": 8,
            "max": 12
        }

    If only a qualitative preference is given:

        {
            "type": "qualitative",
            "preference": "high"
        }
    """

    # --------------------------------------------------------
    # Explicit return range
    #
    # 8-12%
    # 8 to 12%
    # between 8 and 12%
    # --------------------------------------------------------

    pattern = re.search(
        r"(?:between\s+)?"
        r"(\d+(?:\.\d+)?)\s*"
        r"(?:-|to)\s*"
        r"(\d+(?:\.\d+)?)\s*%",
        query,
        re.IGNORECASE
    )

    if pattern:
        return {
            "type": "range",
            "min": float(pattern.group(1)),
            "max": float(pattern.group(2))
        }

    # --------------------------------------------------------
    # Minimum return
    #
    # at least 10%
    # minimum 10%
    # above 10%
    # over 10%
    # more than 10%
    # --------------------------------------------------------

    pattern = re.search(
        r"(?:at\s+least|minimum|min\.?|above|over|more\s+than)"
        r"\s*(\d+(?:\.\d+)?)\s*%",
        query,
        re.IGNORECASE
    )

    if pattern:
        return {
            "type": "minimum",
            "value": float(pattern.group(1))
        }

    # --------------------------------------------------------
    # Maximum return
    #
    # below 10%
    # under 10%
    # maximum 10%
    # up to 10%
    # --------------------------------------------------------

    pattern = re.search(
        r"(?:below|under|maximum|max\.?|up\s+to)"
        r"\s*(\d+(?:\.\d+)?)\s*%",
        query,
        re.IGNORECASE
    )

    if pattern:
        return {
            "type": "maximum",
            "value": float(pattern.group(1))
        }

    # --------------------------------------------------------
    # Around a specific return
    #
    # around 10%
    # approximately 10%
    # about 10%
    # near 10%
    # --------------------------------------------------------

    pattern = re.search(
        r"(?:around|approximately|approx\.?|about|near)"
        r"\s*(\d+(?:\.\d+)?)\s*%",
        query,
        re.IGNORECASE
    )

    if pattern:
        return {
            "type": "target",
            "value": float(pattern.group(1))
        }

    # --------------------------------------------------------
    # Qualitative return preference
    # --------------------------------------------------------

    if re.search(
        r"\b(high|higher|maximum|maximize|best|better)\s+"
        r"(?:return|returns|yield|yields)\b",
        query,
        re.IGNORECASE
    ):
        return {
            "type": "qualitative",
            "preference": "high"
        }

    if re.search(
        r"\b(low|lower)\s+"
        r"(?:return|returns|yield|yields)\b",
        query,
        re.IGNORECASE
    ):
        return {
            "type": "qualitative",
            "preference": "low"
        }

    return None


# ============================================================
# 7. MAIN QUERY PARSER
# ============================================================

def parse_query(query: str) -> Dict[str, Any]:
    """
    Parse a complete natural-language financial query.

    Parameters
    ----------
    query : str
        User's natural-language requirement.

    Returns
    -------
    dict
        Structured query representation.
    """

    if not isinstance(query, str):
        raise TypeError("Query must be a string.")

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    normalized_query = normalize_text(query)

    budget = extract_budget(normalized_query)
    risk = extract_risk(normalized_query)
    tenure = extract_tenure(normalized_query)
    return_preference = extract_return_preference(normalized_query)

    result = {
        "budget": budget,
        "risk": risk,
        "tenure_months": tenure,
        "return_preference": return_preference,
        "query_text": query,

        # Useful for debugging and evaluation.
        "parsed_fields": {
            "budget": budget is not None,
            "risk": risk is not None,
            "tenure": tenure is not None,
            "return": return_preference is not None
        }
    }

    return result


# ============================================================
# 8. COMMAND-LINE TESTING
# ============================================================

if __name__ == "__main__":

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Parse a natural-language financial query."
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural-language financial requirement."
    )

    args = parser.parse_args()

    result = parse_query(args.query)

    print(json.dumps(
        result,
        indent=4,
        ensure_ascii=False
    ))