"""
PIPELINE 4 - MCQ Challenge Generator
Data source: O*NET DWAs (via dimension plan) + Country config

Generates scenario-based multiple-choice questions (4 options) per skill dimension.
Scoring is deterministic server-side - no second LLM call needed for evaluation.
Options are shuffled before sending to client so the correct answer position is hidden.

Score mapping (per question):
  Correct answer  -> 10/10 -> 100/100
  Partial answer  ->  5/10 ->  50/100
  Wrong answer    ->  0/10 ->   0/100
"""

import random

from app.services.llm import chat_json

TIER_DESCRIPTIONS = {
    1: "Entry - a common everyday work situation, basic correct response expected",
    2: "Functional - requires judgment with incomplete information or a tricky customer",
    3: "Mastery - complex or competing priorities, must reason through the best approach",
}

MCQ_PROMPT = """You are creating a multiple-choice assessment question for an informal economy worker.
This question will be answered by tapping one of four options on a phone - no typing required.

Worker occupation: {occupation_title}
Skill dimension: {dimension_label} - {dimension_description}
Real work task this is based on: "{primary_dwa}"

Local context:
- Location: {location_context}
- Tools & materials typical here: {local_tools}
- Currency: {currency}
- Local examples: {local_examples}
- Language register: {language_register}
- Opportunity types: {opportunity_types}
- Difficulty: Tier {tier} - {tier_description}

Create ONE realistic scenario question with exactly 4 options.

Rules:
- The scenario must describe a real situation this worker encounters regularly
- Use local language, tools, prices in {currency}, and customer types from {location_context}
- All 4 options must sound plausible - no obviously silly answers
- Exactly ONE option scores 10 (clearly the best action)
- Exactly ONE option scores 5 (reasonable but not optimal - a common near-miss)
- Exactly TWO options score 0 (wrong - represent real mistakes workers make)
- The explanation for each must be honest, specific, and educational (1 sentence)

Return ONLY valid JSON in this exact structure:
{{
  "question": "Scenario + question in 2-3 sentences. End with a clear question.",
  "options": [
    {{"id": "A", "text": "Option text (max 20 words)", "score": 10, "explanation": "Why this is the best action."}},
    {{"id": "B", "text": "Option text (max 20 words)", "score": 5,  "explanation": "Why this is close but not optimal."}},
    {{"id": "C", "text": "Option text (max 20 words)", "score": 0,  "explanation": "Why this leads to a problem."}},
    {{"id": "D", "text": "Option text (max 20 words)", "score": 0,  "explanation": "Why this leads to a problem."}}
  ]
}}"""


def _sanitise_for_client(challenge: dict) -> dict:
    """
    Return a version of the challenge safe to send to the browser.
    Strips score and explanation from options so the client cannot see the answer key.
    """
    return {
        "question": challenge["question"],
        "dimension_id": challenge["dimension_id"],
        "dimension_label": challenge["dimension_label"],
        "tier": challenge["tier"],
        "options": [
            {"id": opt["id"], "text": opt["text"]}
            for opt in challenge["options"]
        ],
    }


def generate_challenge(
    occupation_title: str,
    dimension: dict,
    country_config: dict,
    tier: int = 1,
) -> dict:
    """
    Pipeline 4: Generate a 4-option MCQ for a given skill dimension.

    Returns TWO dicts:
      "full"   - stored server-side in session._current_challenge (contains answer keys)
      "client" - sent to the browser (options have no score/explanation)
    """
    prompt = MCQ_PROMPT.format(
        occupation_title=occupation_title,
        dimension_label=dimension["label"],
        dimension_description=dimension["description"],
        primary_dwa=dimension["primary_dwa"],
        location_context=country_config["location_context"],
        local_tools=country_config["local_tools"],
        currency=country_config["currency"],
        local_examples=", ".join(country_config.get("local_examples", [])) or "Not specified",
        language_register=country_config.get("language_register", "Standard English"),
        opportunity_types=", ".join(country_config.get("opportunity_types", [])) or "Not specified",
        tier=tier,
        tier_description=TIER_DESCRIPTIONS[tier],
    )

    raw = chat_json(
        system="You generate MCQ assessment questions. Always respond with valid JSON only.",
        user=prompt,
        max_tokens=700,
        temperature=0.4,
        preferred_openai_model="gpt-4o",
    )

    # Shuffle options so the correct answer isn't always position A
    options = raw.get("options", [])
    random.shuffle(options)

    # Re-label A/B/C/D after shuffle
    for i, opt in enumerate(options):
        opt["id"] = ["A", "B", "C", "D"][i]

    full_challenge = {
        "question": raw["question"],
        "options": options,
        "dimension_id": dimension["dimension_id"],
        "dimension_label": dimension["label"],
        "employer_signal": dimension["employer_signal"],
        "tier": tier,
        "occupation_title": occupation_title,
        "region": country_config["region_id"],
    }

    return {
        "full": full_challenge,
        "client": _sanitise_for_client(full_challenge),
    }


def score_mcq_answer(challenge: dict, selected_id: str) -> dict:
    """
    Pipeline 4b: Deterministic MCQ scoring - no API call required.

    Args:
        challenge:    The FULL challenge dict (with scores) from session._current_challenge
        selected_id:  The option ID the worker selected ("A", "B", "C", or "D")

    Returns:
        Evaluation dict compatible with p5_difficulty.advance()
    """
    options_by_id = {opt["id"]: opt for opt in challenge["options"]}
    selected = options_by_id.get(selected_id)

    if not selected:
        selected = {"id": selected_id, "score": 0, "explanation": "Invalid option selected.", "text": ""}

    raw_score = selected["score"]
    score_100 = raw_score * 10

    current_tier = challenge["tier"]
    if score_100 == 100:
        tier_achieved = current_tier
    elif score_100 == 50:
        tier_achieved = max(1, current_tier - 1)
    else:
        tier_achieved = 1

    passed = score_100 >= 70
    correct = next((o for o in challenge["options"] if o["score"] == 10), None)

    if score_100 == 100:
        feedback = f"Correct. {selected['explanation']}"
    elif score_100 == 50:
        feedback = (
            f"Close, but not the best action. {selected['explanation']} "
            f"The stronger approach: {correct['explanation'] if correct else ''}"
        )
    else:
        feedback = (
            f"Not the right approach. {selected['explanation']} "
            f"The correct action: {correct['text']} - {correct['explanation'] if correct else ''}"
        )

    tier_labels = {1: "Entry", 2: "Functional", 3: "Mastery"}
    if passed:
        employer_signal = (
            f"{challenge['dimension_label']}: demonstrated at "
            f"{tier_labels.get(current_tier, 'Entry')} level."
        )
    else:
        employer_signal = (
            f"{challenge['dimension_label']}: below {tier_labels.get(current_tier, 'Entry')} "
            f"threshold - development needed."
        )

    return {
        "score": score_100,
        "tier_achieved": tier_achieved,
        "passed": passed,
        "feedback": feedback,
        "selected_option": selected_id,
        "correct_option": correct["id"] if correct else None,
        "employer_signal": employer_signal,
        "dimension_id": challenge["dimension_id"],
        "dimension_label": challenge["dimension_label"],
    }
