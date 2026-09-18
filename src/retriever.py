import json
import argparse
from typing import List, Dict, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from query_parser import parse_query


# ============================================================
# 1. LOAD NORMALIZED OPPORTUNITIES
# ============================================================

def load_opportunities(path: str) -> List[Dict[str, Any]]:
    """
    Load normalized opportunity records from JSON.
    """

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset must contain a list of opportunities.")

    return data


# ============================================================
# 2. TEXT REPRESENTATION
# ============================================================

def opportunity_to_text(opportunity: Dict[str, Any]) -> str:
    """
    Convert an opportunity into searchable text.

    This is only used as a secondary signal.
    Structured fields are more important for this dataset.
    """

    fields = [
        opportunity.get("name", ""),
        opportunity.get("provider", ""),
        opportunity.get("category", ""),
        opportunity.get("risk", ""),
        opportunity.get("return_type", ""),
    ]

    minimum_investment = opportunity.get("minimum_investment_inr")

    if minimum_investment is not None:
        fields.append(f"investment {minimum_investment}")

    tenure_min = opportunity.get("tenure_min_months")
    tenure_max = opportunity.get("tenure_max_months")

    if tenure_min is not None:
        fields.append(f"tenure {tenure_min}")

    if tenure_max is not None:
        fields.append(f"tenure {tenure_max}")

    return " ".join(str(x) for x in fields if x)


# ============================================================
# 3. RISK NORMALIZATION
# ============================================================

def normalize_risk(risk: Any) -> str:
    """
    Convert different risk labels into canonical levels.
    """

    if risk is None:
        return "unknown"

    value = str(risk).strip().lower()

    mapping = {
        "low": "low",
        "conservative": "low",

        "low to moderate": "low_moderate",
        "low-to-moderate": "low_moderate",

        "moderate": "moderate",
        "medium": "moderate",

        "moderate to high": "moderate_high",
        "moderate-to-high": "moderate_high",

        "moderately high": "moderate_high",

        "high": "high",
        "aggressive": "high",
    }

    return mapping.get(value, value)


RISK_ORDER = {
    "low": 0,
    "low_moderate": 1,
    "moderate": 2,
    "moderate_high": 3,
    "high": 4,
}


# ============================================================
# 4. BUDGET MATCH
# ============================================================

def budget_match(
    user_budget: float,
    minimum_investment: Any
) -> float:
    """
    Calculate budget suitability.

    IMPORTANT:
    If the user cannot afford the minimum investment,
    return 0.0.

    Otherwise:
        <= 50% of available budget -> 1.0
        50-100% -> gradually decreases
        exactly 100% -> 0.5
    """

    if user_budget is None or minimum_investment is None:
        return 0.0

    minimum_investment = float(minimum_investment)

    # HARD CONSTRAINT
    if minimum_investment > user_budget:
        return 0.0

    ratio = minimum_investment / user_budget

    if ratio <= 0.50:
        return 1.0

    # 0.50 -> 1.0
    # score: 1.0 -> 0.5
    score = 1.0 - (ratio - 0.50)

    return max(0.5, min(1.0, score))


# ============================================================
# 5. RISK MATCH
# ============================================================

def risk_match(
    user_risk: str,
    opportunity_risk: Any
) -> float:
    """
    Compare requested risk with opportunity risk.

    Exact match      -> 1.0
    One level away   -> 0.5
    More than one    -> 0.0
    """

    if user_risk is None or opportunity_risk is None:
        return 0.0

    user = normalize_risk(user_risk)
    opportunity = normalize_risk(opportunity_risk)

    if user not in RISK_ORDER or opportunity not in RISK_ORDER:
        return 0.0

    difference = abs(
        RISK_ORDER[user] - RISK_ORDER[opportunity]
    )

    if difference == 0:
        return 1.0

    if difference == 1:
        return 0.5

    return 0.0


# ============================================================
# 6. TENURE MATCH
# ============================================================

def tenure_match(
    requested_months: float,
    tenure_min: Any,
    tenure_max: Any
) -> float:
    """
    Match requested investment duration with
    opportunity tenure.

    Inside range       -> 1.0
    <= 6 months away   -> 0.75
    <= 12 months away  -> 0.50
    <= 24 months away  -> 0.25
    Otherwise          -> 0.0
    """

    if requested_months is None:
        return 0.0

    if tenure_min is None and tenure_max is None:
        return 0.0

    requested_months = float(requested_months)

    min_months = (
        float(tenure_min)
        if tenure_min is not None
        else None
    )

    max_months = (
        float(tenure_max)
        if tenure_max is not None
        else None
    )

    # Requested duration is directly inside the range
    if min_months is not None and max_months is not None:

        if min_months <= requested_months <= max_months:
            return 1.0

        if requested_months < min_months:
            difference = min_months - requested_months
        else:
            difference = requested_months - max_months

    # Only minimum tenure is known
    elif min_months is not None:

        if requested_months >= min_months:
            return 1.0

        difference = min_months - requested_months

    # Only maximum tenure is known
    else:

        if requested_months <= max_months:
            return 1.0

        difference = requested_months - max_months

    if difference <= 6:
        return 0.75

    if difference <= 12:
        return 0.50

    if difference <= 24:
        return 0.25

    return 0.0


# ============================================================
# 7. RETURN MATCH
# ============================================================

def return_match(
    query_return: Any,
    opportunity: Dict[str, Any]
) -> float:
    """
    Match requested return preference.

    If the user did not specify return preference,
    return 0.0 because it should not affect ranking.

    Supports:
        minimum
        maximum
        target
        range
        qualitative
    """

    if not query_return:
        return 0.0

    return_min = opportunity.get("expected_return_min_pct")
    return_max = opportunity.get("expected_return_max_pct")

    if return_min is None and return_max is None:
        return 0.0

    try:
        return_min = float(return_min) if return_min is not None else None
        return_max = float(return_max) if return_max is not None else None
    except (TypeError, ValueError):
        return 0.0

    preference_type = query_return.get("type")

    # --------------------------------------------------------
    # Minimum requested return
    # --------------------------------------------------------

    if preference_type == "minimum":

        requested = query_return.get("value")

        if requested is None:
            return 0.0

        if return_max is None:
            return 0.0

        if return_max >= requested:
            return min(1.0, return_max / requested)

        return 0.0

    # --------------------------------------------------------
    # Maximum requested return
    # --------------------------------------------------------

    if preference_type == "maximum":

        requested = query_return.get("value")

        if requested is None:
            return 0.0

        if return_min is None:
            return 0.0

        if return_min <= requested:
            return 1.0

        return 0.0

    # --------------------------------------------------------
    # Target return
    # --------------------------------------------------------

    if preference_type == "target":

        target = query_return.get("value")

        if target is None:
            return 0.0

        if return_min is None and return_max is None:
            return 0.0

        if return_min is None:
            difference = abs(return_max - target)
        elif return_max is None:
            difference = abs(return_min - target)
        elif return_min <= target <= return_max:
            return 1.0
        else:
            difference = min(
                abs(return_min - target),
                abs(return_max - target)
            )

        if difference <= 1:
            return 0.9

        if difference <= 2:
            return 0.7

        if difference <= 5:
            return 0.4

        return 0.0

    # --------------------------------------------------------
    # Return range requested
    # --------------------------------------------------------

    if preference_type == "range":

        requested_min = query_return.get("min")
        requested_max = query_return.get("max")

        if requested_min is None or requested_max is None:
            return 0.0

        if return_min is None or return_max is None:
            return 0.0

        # Ranges overlap
        if return_max >= requested_min and return_min <= requested_max:
            return 1.0

        return 0.0

    # --------------------------------------------------------
    # Qualitative preference
    # --------------------------------------------------------

    if preference_type == "qualitative":

        level = query_return.get("value")

        if return_max is None:
            return 0.0

        if level == "high":
            return min(return_max / 20.0, 1.0)

        if level == "low":
            if return_min is None:
                return 0.0

            return max(0.0, 1.0 - (return_min / 20.0))

    return 0.0


# ============================================================
# 8. STRUCTURED MATCH SIGNALS
# ============================================================

def calculate_structured_signals(
    opportunity: Dict[str, Any],
    parsed_query: Dict[str, Any]
) -> Dict[str, float]:

    user_budget = parsed_query.get("budget")
    user_risk = parsed_query.get("risk")
    user_tenure = parsed_query.get("tenure_months")
    user_return = parsed_query.get("return_preference")

    minimum_investment = opportunity.get(
        "minimum_investment_inr"
    )

    signals = {
        "budget_match": budget_match(
            user_budget,
            minimum_investment
        ) if user_budget is not None else 0.0,

        "risk_match": risk_match(
            user_risk,
            opportunity.get("risk")
        ) if user_risk is not None else 0.0,

        "tenure_match": tenure_match(
            user_tenure,
            opportunity.get("tenure_min_months"),
            opportunity.get("tenure_max_months")
        ) if user_tenure is not None else 0.0,

        "return_match": return_match(
            user_return,
            opportunity
        ) if user_return is not None else 0.0,
    }

    return signals


# ============================================================
# 9. HARD ELIGIBILITY CHECK
# ============================================================

def is_budget_feasible(
    opportunity: Dict[str, Any],
    parsed_query: Dict[str, Any]
) -> bool:
    """
    Determine whether the user can afford the opportunity.

    This is a hard constraint.

    Example:
        User budget = 2 lakh
        Minimum investment = 3 lakh

        -> False
    """

    user_budget = parsed_query.get("budget")
    minimum_investment = opportunity.get(
        "minimum_investment_inr"
    )

    if user_budget is None:
        return True

    if minimum_investment is None:
        return True

    return float(minimum_investment) <= float(user_budget)


# ============================================================
# 10. OPPORTUNITY RETRIEVER
# ============================================================

class OpportunityRetriever:

    def __init__(
        self,
        opportunities: List[Dict[str, Any]]
    ):

        self.opportunities = opportunities

        # TF-IDF is retained as a secondary textual signal.
        self.documents = [
            opportunity_to_text(op)
            for op in opportunities
        ]

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        self.document_matrix = self.vectorizer.fit_transform(
            self.documents
        )

    # --------------------------------------------------------
    # TEXT SIMILARITY
    # --------------------------------------------------------

    def calculate_text_similarity(
        self,
        query_text: str
    ) -> np.ndarray:

        query_vector = self.vectorizer.transform(
            [query_text]
        )

        similarities = cosine_similarity(
            query_vector,
            self.document_matrix
        )[0]

        return similarities

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    def retrieve(
        self,
        parsed_query: Dict[str, Any],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:

        query_text = parsed_query.get(
            "query_text",
            ""
        )

        text_scores = self.calculate_text_similarity(
            query_text
        )

        candidates = []

        for index, opportunity in enumerate(
            self.opportunities
        ):

            # ----------------------------------------------
            # HARD BUDGET FILTER
            # ----------------------------------------------

            budget_feasible = is_budget_feasible(
                opportunity,
                parsed_query
            )

            if not budget_feasible:
                continue

            # ----------------------------------------------
            # STRUCTURED SIGNALS
            # ----------------------------------------------

            signals = calculate_structured_signals(
                opportunity,
                parsed_query
            )

            # ----------------------------------------------
            # COUNT USER-SPECIFIED ATTRIBUTES
            # ----------------------------------------------

            parsed_fields = parsed_query.get(
                "parsed_fields",
                {}
            )

            active_scores = []

            if parsed_fields.get("budget"):
                active_scores.append(
                    signals["budget_match"]
                )

            if parsed_fields.get("risk"):
                active_scores.append(
                    signals["risk_match"]
                )

            if parsed_fields.get("tenure"):
                active_scores.append(
                    signals["tenure_match"]
                )

            if parsed_fields.get("return"):
                active_scores.append(
                    signals["return_match"]
                )

            # ----------------------------------------------
            # STRUCTURED SCORE
            # ----------------------------------------------

            if active_scores:
                structured_score = sum(
                    active_scores
                ) / len(active_scores)
            else:
                structured_score = 0.0

            # ----------------------------------------------
            # TEXT SCORE
            # ----------------------------------------------

            text_score = float(
                text_scores[index]
            )

            # ----------------------------------------------
            # FINAL RETRIEVAL SCORE
            #
            # Structured attributes are primary.
            # Text is secondary.
            # ----------------------------------------------

            retrieval_score = (
                0.85 * structured_score
                +
                0.15 * text_score
            )

            candidates.append({

                "opportunity": opportunity,

                "retrieval_score": float(
                    retrieval_score
                ),

                "text_similarity": text_score,

                "budget_match": signals[
                    "budget_match"
                ],

                "risk_match": signals[
                    "risk_match"
                ],

                "tenure_match": signals[
                    "tenure_match"
                ],

                "return_match": signals[
                    "return_match"
                ],

                "budget_feasible": True
            })

        # ----------------------------------------------------
        # SORT BY RETRIEVAL SCORE
        # ----------------------------------------------------

        candidates.sort(
            key=lambda x: x["retrieval_score"],
            reverse=True
        )

        return candidates[:top_k]


# ============================================================
# 11. CONVENIENCE FUNCTION
# ============================================================

def retrieve_opportunities(
    query: str,
    dataset_path: str = "opportunities_dataset.json",
    top_k: int = 10
) -> List[Dict[str, Any]]:

    opportunities = load_opportunities(
        dataset_path
    )

    parsed_query = parse_query(query)

    retriever = OpportunityRetriever(
        opportunities
    )

    return retriever.retrieve(
        parsed_query,
        top_k=top_k
    )


# ============================================================
# 12. CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Constraint-aware financial opportunity retriever"
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Natural language user query"
    )

    parser.add_argument(
        "--dataset",
        default="opportunities_dataset.json",
        help="Path to normalized opportunity JSON"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of candidates to retrieve"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Parse query
    # --------------------------------------------------------

    parsed_query = parse_query(
        args.query
    )

    print("\nParsed Query")
    print("=" * 60)
    print(json.dumps(
        parsed_query,
        indent=4
    ))

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    opportunities = load_opportunities(
        args.dataset
    )

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    retriever = OpportunityRetriever(
        opportunities
    )

    results = retriever.retrieve(
        parsed_query,
        top_k=args.top_k
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\nRetrieved Opportunities")
    print("=" * 60)

    if not results:
        print(
            "\nNo opportunities satisfy the user's "
            "hard constraints."
        )
        return

    for i, result in enumerate(
        results,
        start=1
    ):

        opportunity = result["opportunity"]

        print(
            f"\n{i}. "
            f"{opportunity.get('opportunity_id')} - "
            f"{opportunity.get('name')}"
        )

        print(
            f"   Retrieval Score : "
            f"{result['retrieval_score']:.4f}"
        )

        print(
            f"   Text Similarity : "
            f"{result['text_similarity']:.4f}"
        )

        print(
            f"   Budget Match    : "
            f"{result['budget_match']:.4f}"
        )

        print(
            f"   Risk Match      : "
            f"{result['risk_match']:.4f}"
        )

        print(
            f"   Tenure Match    : "
            f"{result['tenure_match']:.4f}"
        )

        print(
            f"   Return Match    : "
            f"{result['return_match']:.4f}"
        )

        print(
            f"   Min Investment : "
            f"{opportunity.get('minimum_investment_inr')}"
        )

        print(
            f"   Risk            : "
            f"{opportunity.get('risk')}"
        )

        print(
            f"   Tenure          : "
            f"{opportunity.get('tenure_min_months')} - "
            f"{opportunity.get('tenure_max_months')} months"
        )

        print(
            f"   Source          : "
            f"{opportunity.get('source_document')}"
        )


if __name__ == "__main__":
    main()