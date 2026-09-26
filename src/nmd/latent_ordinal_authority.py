from __future__ import annotations

from dataclasses import dataclass

from .contracts import LogicalOption

DOMAINS_ORDER = ("CV", "CW", "CX", "CY")
DOMAIN_SEEDS = {
    "CV": 381101,
    "CW": 381107,
    "CX": 381119,
    "CY": 381131,
}
CASES_PER_DOMAIN = 72
TOTAL_CASES = 288
VIEW_IDS = ("D0", "D1", "D2")

DOMAIN_CONTEXT = {
    "CV": "municipal noise-impact report",
    "CW": "appliance service-impact triage report",
    "CX": "campus facility incident assessment",
    "CY": "neighborhood mobility disruption report",
}

SEVERITY_FIELD_PHRASES = (
    (
        "The disruption is faint enough that ordinary activity continues almost unchanged.",
        "Practical consequences are small and remain easy to absorb.",
        "Normal operation is only lightly affected by the reported condition.",
        "The effect stays narrow and causes little interruption.",
        "Everyday activity continues with only a modest trace of disruption.",
        "The reported condition produces a low-impact inconvenience rather than a serious interruption.",
    ),
    (
        "The disruption is clearly noticeable and requires attention, while routine activity can still continue.",
        "Consequences are meaningful enough to matter but remain within manageable operating conditions.",
        "Normal operation is affected in a visible way without becoming severely impaired.",
        "The condition causes a contained interruption that should be addressed in ordinary follow-up.",
        "The effect is substantial enough to notice yet remains controllable without emergency handling.",
        "The report describes a material inconvenience that stays within normal service capacity.",
    ),
    (
        "The disruption materially interferes with normal operation and calls for accelerated attention.",
        "Consequences are pronounced enough to impair routine activity in a serious way.",
        "The condition creates a strong operational interruption that should be handled promptly.",
        "Normal service is significantly degraded by the reported effect.",
        "The impact is severe enough that delayed handling would be inappropriate.",
        "The report describes a major interruption with clear operational consequences.",
    ),
    (
        "The disruption has reached the highest-impact condition and requires immediate intervention.",
        "Consequences are extreme enough that normal operating procedures are no longer sufficient.",
        "The condition produces an acute interruption that cannot safely wait for routine handling.",
        "Normal service is overwhelmed by the reported effect and immediate escalation is warranted.",
        "The impact sits at the emergency end of the scale with exceptionally serious consequences.",
        "The report describes an extreme operational breakdown requiring immediate response.",
    ),
)

CONFIDENCE_FIELD_PHRASES = (
    (
        "The account is supported only by weak or conflicting indications and remains doubtful.",
        "Available evidence is too uncertain to establish the report with confidence.",
        "The information lacks dependable corroboration and may still be mistaken.",
        "Support for the account is fragmentary and leaves substantial uncertainty.",
        "The report rests on evidence that is not yet reliable enough to trust.",
        "The available indications remain ambiguous and do not establish the facts.",
    ),
    (
        "The account has credible preliminary support, although final confirmation is still incomplete.",
        "Available evidence points in the same direction but has not yet been fully corroborated.",
        "The report is supported by plausible evidence while a final verification step remains open.",
        "There is meaningful support for the account, though complete confirmation has not been reached.",
        "The evidence is credible enough for a provisional conclusion but still needs final checking.",
        "The facts have preliminary backing without complete independent confirmation.",
    ),
    (
        "The account has been independently corroborated and the supporting evidence is complete.",
        "Available evidence has been fully checked and consistently confirms the report.",
        "The report is backed by completed verification from dependable sources.",
        "The facts are established by strong corroborating evidence with no pending confirmation step.",
        "The evidence has passed full confirmation and supports the account reliably.",
        "Independent checks consistently verify the reported facts.",
    ),
)

FLAT_SEVERITY_DEFINITIONS = (
    (
        "Choose the lowest impact category when normal activity is almost unaffected.",
        "This category covers narrow consequences that remain easy to absorb.",
        "Use this category for only light operational disturbance.",
    ),
    (
        "Choose the next impact category when disruption is noticeable but still manageable.",
        "This category covers meaningful consequences that remain within ordinary handling.",
        "Use this category for contained interruption without severe impairment.",
    ),
    (
        "Choose the higher impact category when routine operation is materially impaired.",
        "This category covers pronounced consequences that need accelerated attention.",
        "Use this category for a major interruption short of the maximum emergency condition.",
    ),
    (
        "Choose the maximum impact category when consequences are extreme and immediate escalation is required.",
        "This category covers the most serious operational condition.",
        "Use this category when ordinary handling is no longer sufficient.",
    ),
)

FLAT_CONFIDENCE_DEFINITIONS = (
    (
        "Choose the lowest certainty category when support is ambiguous or unreliable.",
        "This category covers reports whose evidence remains doubtful.",
        "Use this category when the facts are not dependably established.",
    ),
    (
        "Choose the middle certainty category when credible support exists but confirmation is incomplete.",
        "This category covers preliminary evidence awaiting final corroboration.",
        "Use this category for plausibly supported but not fully verified facts.",
    ),
    (
        "Choose the highest certainty category when independent corroboration is complete.",
        "This category covers fully checked evidence that consistently supports the facts.",
        "Use this category when the report has dependable completed verification.",
    ),
)

SEVERITY_THRESHOLD_DEFINITIONS = (
    (
        (
            "FALSE means the consequences remain only light and do not exceed minor disruption.",
            "FALSE applies while ordinary activity is almost unaffected.",
            "FALSE means the effect stays within the lowest-impact band.",
        ),
        (
            "TRUE means the consequences exceed a merely light disturbance and are meaningfully disruptive.",
            "TRUE applies once the effect rises above the lowest-impact band.",
            "TRUE means there is more than a slight operational consequence.",
        ),
    ),
    (
        (
            "FALSE means the consequences remain below a major operational interruption.",
            "FALSE applies while the effect has not reached a strongly impairing level.",
            "FALSE means disruption is still short of the higher-impact band.",
        ),
        (
            "TRUE means the consequences have reached a major operational interruption requiring prompt attention.",
            "TRUE applies once routine activity is materially impaired.",
            "TRUE means the effect is at least strongly disruptive.",
        ),
    ),
    (
        (
            "FALSE means the consequences remain below the maximum emergency end of the impact scale.",
            "FALSE applies while immediate emergency escalation is not yet warranted.",
            "FALSE means the effect has not reached the highest-impact band.",
        ),
        (
            "TRUE means the consequences have reached the maximum emergency end of the impact scale.",
            "TRUE applies once immediate intervention is warranted.",
            "TRUE means the effect is at the highest-impact condition.",
        ),
    ),
)

CONFIDENCE_THRESHOLD_DEFINITIONS = (
    (
        (
            "FALSE means the evidence remains too doubtful for a credible preliminary conclusion.",
            "FALSE applies when support is ambiguous or unreliable.",
            "FALSE means dependable preliminary backing is absent.",
        ),
        (
            "TRUE means the evidence has at least credible preliminary support.",
            "TRUE applies once the report has meaningful backing even if final confirmation is incomplete.",
            "TRUE means support is strong enough for at least a provisional conclusion.",
        ),
    ),
    (
        (
            "FALSE means the evidence has not yet reached complete independent confirmation.",
            "FALSE applies while any final corroboration step remains open.",
            "FALSE means full verification has not been achieved.",
        ),
        (
            "TRUE means the evidence has reached complete independent confirmation.",
            "TRUE applies when corroboration is finished and consistently supports the facts.",
            "TRUE means the report is fully verified.",
        ),
    ),
)


@dataclass(frozen=True)
class LatentOrdinalAuthorityCase:
    case_id: str
    domain_id: str
    severity: int
    confidence_index: int
    variant: int
    severity_field: str
    confidence_field: str


def _binary_options(prefix: str, definitions):
    return tuple(
        LogicalOption(option_id=f"{prefix}-{index}", criterion_text=views[0])
        for index, views in enumerate(definitions)
    )


FLAT_SEVERITY_OPTIONS = tuple(
    LogicalOption(option_id=f"severity19-{index}", criterion_text=views[0])
    for index, views in enumerate(FLAT_SEVERITY_DEFINITIONS)
)
FLAT_CONFIDENCE_OPTIONS = tuple(
    LogicalOption(option_id=f"confidence19-{index}", criterion_text=views[0])
    for index, views in enumerate(FLAT_CONFIDENCE_DEFINITIONS)
)
SEVERITY_THRESHOLD_OPTIONS = tuple(
    _binary_options(f"severity19-t{index+1}", definitions)
    for index, definitions in enumerate(SEVERITY_THRESHOLD_DEFINITIONS)
)
CONFIDENCE_THRESHOLD_OPTIONS = tuple(
    _binary_options(f"confidence19-t{index+1}", definitions)
    for index, definitions in enumerate(CONFIDENCE_THRESHOLD_DEFINITIONS)
)


def generate_all_w19() -> tuple[LatentOrdinalAuthorityCase, ...]:
    rows = []
    for domain_id in DOMAINS_ORDER:
        context = DOMAIN_CONTEXT[domain_id]
        for severity in range(4):
            for confidence in range(3):
                for variant in range(6):
                    severity_text = (
                        f"In this {context}, {SEVERITY_FIELD_PHRASES[severity][variant]}"
                    )
                    confidence_text = (
                        f"For this {context}, {CONFIDENCE_FIELD_PHRASES[confidence][variant]}"
                    )
                    rows.append(
                        LatentOrdinalAuthorityCase(
                            case_id=f"{domain_id.lower()}-s{severity}-c{confidence}-v{variant}",
                            domain_id=domain_id,
                            severity=severity,
                            confidence_index=confidence,
                            variant=variant,
                            severity_field=severity_text,
                            confidence_field=confidence_text,
                        )
                    )
    if len(rows) != TOTAL_CASES:
        raise RuntimeError("W19 requires exactly 288 cases")
    for domain_id in DOMAINS_ORDER:
        subset = [row for row in rows if row.domain_id == domain_id]
        if len(subset) != CASES_PER_DOMAIN:
            raise RuntimeError("W19 requires exactly 72 cases/domain")
        if [sum(row.severity == s for row in subset) for s in range(4)] != [18] * 4:
            raise RuntimeError("W19 severity balance changed")
        if [sum(row.confidence_index == c for row in subset) for c in range(3)] != [24] * 3:
            raise RuntimeError("W19 confidence balance changed")
    return tuple(rows)


def all_w19_text_atoms() -> set[str]:
    values = set()
    for row in generate_all_w19():
        values.add(row.severity_field)
        values.add(row.confidence_field)
    for group in (FLAT_SEVERITY_DEFINITIONS, FLAT_CONFIDENCE_DEFINITIONS):
        for views in group:
            values.update(views)
    for threshold_group in (
        SEVERITY_THRESHOLD_DEFINITIONS,
        CONFIDENCE_THRESHOLD_DEFINITIONS,
    ):
        for threshold in threshold_group:
            for option in threshold:
                values.update(option)
    return values


__all__ = [
    "CASES_PER_DOMAIN",
    "CONFIDENCE_THRESHOLD_DEFINITIONS",
    "CONFIDENCE_THRESHOLD_OPTIONS",
    "DOMAINS_ORDER",
    "FLAT_CONFIDENCE_DEFINITIONS",
    "FLAT_CONFIDENCE_OPTIONS",
    "FLAT_SEVERITY_DEFINITIONS",
    "FLAT_SEVERITY_OPTIONS",
    "LatentOrdinalAuthorityCase",
    "SEVERITY_THRESHOLD_DEFINITIONS",
    "SEVERITY_THRESHOLD_OPTIONS",
    "TOTAL_CASES",
    "VIEW_IDS",
    "all_w19_text_atoms",
    "generate_all_w19",
]
