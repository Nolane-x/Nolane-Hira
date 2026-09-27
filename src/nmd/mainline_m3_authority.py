from __future__ import annotations

from dataclasses import dataclass

from .contracts import LogicalOption, Primitive

M3_PARTITIONS = {
    "dev": ("MVA", "MVB"),
    "confirm": ("MVC",),
}

M3_SEALED_PARTITIONS = {"confirm"}

DOMAIN_CONTEXT = {
    "MVA": {
        "en": "neighborhood repair-clinic registration desk",
        "vi": "quầy đăng ký phòng sửa chữa cộng đồng",
    },
    "MVB": {
        "en": "regional shared-studio equipment checkout",
        "vi": "quầy mượn thiết bị xưởng dùng chung của khu vực",
    },
    "MVC": {
        "en": "municipal community-kitchen booking office",
        "vi": "văn phòng đặt lịch bếp cộng đồng của thành phố",
    },
}

ROUTES = {
    "MVA": {
        "en": ("river route", "cedar route", "lantern route"),
        "vi": ("tuyến sông", "tuyến tuyết tùng", "tuyến đèn lồng"),
    },
    "MVB": {
        "en": ("harbor route", "meadow route", "granite route"),
        "vi": ("tuyến bến cảng", "tuyến đồng cỏ", "tuyến đá granit"),
    },
    "MVC": {
        "en": ("orchard route", "cobalt route", "willow route"),
        "vi": ("tuyến vườn cây", "tuyến cô-ban", "tuyến liễu"),
    },
}

STYLE_TEXT = {
    "direct": {
        "en": "The entry is direct and contains no competing instruction.",
        "vi": "Mục ghi chép nêu trực tiếp và không có chỉ dẫn cạnh tranh.",
    },
    "qualified": {
        "en": "The entry also contains one administrative note that does not change the decision.",
        "vi": "Mục ghi chép có thêm một ghi chú hành chính nhưng không làm thay đổi quyết định.",
    },
    "sparse": {
        "en": "The entry is short and gives only the decision-bearing statement.",
        "vi": "Mục ghi chép ngắn và chỉ nêu thông tin quyết định cần thiết.",
    },
}

QUESTION_TEXT = {
    "choice": {
        "en": "Which processing route is explicitly assigned to this request?",
        "vi": "Yêu cầu này được chỉ định rõ vào tuyến xử lý nào?",
    },
    "score": {
        "en": "Which workload level is explicitly stated for this request?",
        "vi": "Yêu cầu này được nêu rõ ở mức khối lượng công việc nào?",
    },
    "noul": {
        "en": "Does the record explicitly require an independent staff check?",
        "vi": "Hồ sơ có yêu cầu rõ phải có một lần kiểm tra độc lập của nhân viên hay không?",
    },
}

SCORE_LABELS = (
    ("low", "thấp", 0.0),
    ("medium", "trung bình", 1.0),
    ("high", "cao", 2.0),
)


@dataclass(frozen=True)
class M3PairedCase:
    pair_id: str
    partition: str
    domain_id: str
    primitive: Primitive
    gold_index: int
    en_state_text: str
    vi_state_text: str
    en_question_text: str
    vi_question_text: str
    en_options: tuple[LogicalOption, ...]
    vi_options: tuple[LogicalOption, ...]


def _choice_options(domain: str, language: str) -> tuple[LogicalOption, ...]:
    routes = ROUTES[domain][language]
    if language == "en":
        return tuple(
            LogicalOption(
                option_id=f"{domain.lower()}-route-{index}",
                criterion_text=f"The request belongs to the {route}.",
                aliases=(f"Use the {route} for this request.",),
            )
            for index, route in enumerate(routes)
        )
    return tuple(
        LogicalOption(
            option_id=f"{domain.lower()}-route-{index}",
            criterion_text=f"Yêu cầu thuộc {route}.",
            aliases=(f"Hãy xử lý yêu cầu này theo {route}.",),
        )
        for index, route in enumerate(routes)
    )


def _score_options(domain: str, language: str) -> tuple[LogicalOption, ...]:
    if language == "en":
        return tuple(
            LogicalOption(
                option_id=f"{domain.lower()}-load-{index}",
                criterion_text=f"The stated workload level is {en}.",
                aliases=(f"Workload band {index} corresponds to {en}.",),
                value=value,
            )
            for index, (en, _vi, value) in enumerate(SCORE_LABELS)
        )
    return tuple(
        LogicalOption(
            option_id=f"{domain.lower()}-load-{index}",
            criterion_text=f"Mức khối lượng công việc được nêu là {vi}.",
            aliases=(f"Mức khối lượng {index} tương ứng với mức {vi}.",),
            value=value,
        )
        for index, (_en, vi, value) in enumerate(SCORE_LABELS)
    )


def _noul_options(domain: str, language: str) -> tuple[LogicalOption, ...]:
    if language == "en":
        return (
            LogicalOption(
                option_id=f"{domain.lower()}-check-no",
                criterion_text="An independent staff check is not required.",
                aliases=("The request may proceed without an independent staff check.",),
                value=0.0,
            ),
            LogicalOption(
                option_id=f"{domain.lower()}-check-yes",
                criterion_text="An independent staff check is required.",
                aliases=("The request must receive an independent staff check.",),
                value=1.0,
            ),
        )
    return (
        LogicalOption(
            option_id=f"{domain.lower()}-check-no",
            criterion_text="Không cần một lần kiểm tra độc lập của nhân viên.",
            aliases=("Yêu cầu có thể tiếp tục mà không cần kiểm tra độc lập của nhân viên.",),
            value=0.0,
        ),
        LogicalOption(
            option_id=f"{domain.lower()}-check-yes",
            criterion_text="Cần một lần kiểm tra độc lập của nhân viên.",
            aliases=("Yêu cầu phải được một nhân viên khác kiểm tra độc lập.",),
            value=1.0,
        ),
    )


def _options(domain: str, primitive: Primitive, language: str) -> tuple[LogicalOption, ...]:
    if primitive == "choice":
        return _choice_options(domain, language)
    if primitive == "score":
        return _score_options(domain, language)
    if primitive == "noul":
        return _noul_options(domain, language)
    raise ValueError(f"unsupported M3 primitive: {primitive}")


def _state(
    domain: str,
    primitive: Primitive,
    target: int,
    style: str,
    variant: int,
    language: str,
) -> str:
    context = DOMAIN_CONTEXT[domain][language]
    style_text = STYLE_TEXT[style][language]

    if primitive == "choice":
        route = ROUTES[domain][language][target]
        if language == "en":
            fact = (
                f"The request is explicitly assigned to the {route}; "
                "the other listed routes are not assigned."
            )
        else:
            fact = (
                f"Yêu cầu được chỉ định rõ vào {route}; "
                "các tuyến còn lại trong danh sách không được chỉ định."
            )
    elif primitive == "score":
        en, vi, _value = SCORE_LABELS[target]
        if language == "en":
            fact = (
                f"The record explicitly states a {en} workload level; "
                "no different workload level is stated."
            )
        else:
            fact = (
                f"Hồ sơ nêu rõ mức khối lượng công việc là {vi}; "
                "không có mức khối lượng nào khác được nêu."
            )
    elif primitive == "noul":
        if language == "en":
            fact = (
                "The record explicitly requires an independent staff check before processing."
                if target == 1
                else "The record explicitly states that an independent staff check is not required."
            )
        else:
            fact = (
                "Hồ sơ yêu cầu rõ phải có một lần kiểm tra độc lập của nhân viên trước khi xử lý."
                if target == 1
                else "Hồ sơ nêu rõ rằng không cần một lần kiểm tra độc lập của nhân viên."
            )
    else:
        raise ValueError(f"unsupported M3 primitive: {primitive}")

    if language == "en":
        return f"At the {context}, paired entry {variant:02d} records a request. {fact} {style_text}"
    return f"Tại {context}, mục đối chiếu {variant:02d} ghi nhận một yêu cầu. {fact} {style_text}"


def _generate_domain(domain: str, partition: str) -> tuple[M3PairedCase, ...]:
    rows: list[M3PairedCase] = []
    primitives: tuple[Primitive, ...] = ("choice", "score", "noul")
    styles = ("direct", "qualified", "sparse")

    for primitive_index, primitive in enumerate(primitives):
        k = 2 if primitive == "noul" else 3
        en_options = _options(domain, primitive, "en")
        vi_options = _options(domain, primitive, "vi")

        if tuple(o.option_id for o in en_options) != tuple(o.option_id for o in vi_options):
            raise RuntimeError("M3 EN/VI option ID contract changed")
        if tuple(o.value for o in en_options) != tuple(o.value for o in vi_options):
            raise RuntimeError("M3 EN/VI option value contract changed")

        for style_index, style in enumerate(styles):
            for repeat in range(4):
                target = (repeat + style_index + primitive_index) % k
                variant = primitive_index * 12 + style_index * 4 + repeat
                rows.append(
                    M3PairedCase(
                        pair_id=f"m3-{domain.lower()}-{primitive}-{variant:02d}",
                        partition=partition,
                        domain_id=domain,
                        primitive=primitive,
                        gold_index=target,
                        en_state_text=_state(
                            domain,
                            primitive,
                            target,
                            style,
                            variant,
                            "en",
                        ),
                        vi_state_text=_state(
                            domain,
                            primitive,
                            target,
                            style,
                            variant,
                            "vi",
                        ),
                        en_question_text=QUESTION_TEXT[primitive]["en"],
                        vi_question_text=QUESTION_TEXT[primitive]["vi"],
                        en_options=en_options,
                        vi_options=vi_options,
                    )
                )
    if len(rows) != 36:
        raise RuntimeError("M3 paired authority requires 36 pairs per domain")
    return tuple(rows)


def generate_m3_paired_authority(
    partition: str,
    *,
    allow_sealed: bool = False,
) -> tuple[M3PairedCase, ...]:
    if partition not in M3_PARTITIONS:
        raise ValueError(f"unknown M3 partition: {partition}")
    if partition in M3_SEALED_PARTITIONS and not allow_sealed:
        raise RuntimeError(f"M3 {partition} authority is sealed")

    rows: list[M3PairedCase] = []
    for domain in M3_PARTITIONS[partition]:
        rows.extend(_generate_domain(domain, partition))
    return tuple(rows)


def all_m3_text_atoms(*, include_sealed: bool = True) -> set[str]:
    values: set[str] = set()
    for partition in M3_PARTITIONS:
        if partition in M3_SEALED_PARTITIONS and not include_sealed:
            continue
        for row in generate_m3_paired_authority(
            partition,
            allow_sealed=partition in M3_SEALED_PARTITIONS,
        ):
            values.update(
                (
                    row.en_state_text,
                    row.vi_state_text,
                    row.en_question_text,
                    row.vi_question_text,
                )
            )
            for options in (row.en_options, row.vi_options):
                for option in options:
                    values.add(option.criterion_text)
                    values.update(option.aliases)
    return values


__all__ = [
    "DOMAIN_CONTEXT",
    "M3_PARTITIONS",
    "M3_SEALED_PARTITIONS",
    "M3PairedCase",
    "QUESTION_TEXT",
    "ROUTES",
    "SCORE_LABELS",
    "STYLE_TEXT",
    "all_m3_text_atoms",
    "generate_m3_paired_authority",
]
