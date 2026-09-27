from __future__ import annotations

from .contracts import LogicalOption, Primitive
from .mainline_m3_authority import M3PairedCase

M3_R1_PARTITIONS = {
    "train": ("MVD", "MVE", "MVF", "MVG"),
    "dev": ("MVH", "MVI"),
}

R1_CONTEXT = {
    "MVD": {
        "en": "district makerspace tool reservation counter",
        "vi": "quầy đặt dụng cụ của xưởng sáng tạo cấp quận",
    },
    "MVE": {
        "en": "regional reading-club room allocation desk",
        "vi": "quầy phân phòng câu lạc bộ đọc sách của khu vực",
    },
    "MVF": {
        "en": "community seed-exchange pickup service",
        "vi": "dịch vụ nhận hạt giống trao đổi của cộng đồng",
    },
    "MVG": {
        "en": "municipal mobile-library stop scheduling office",
        "vi": "văn phòng xếp lịch điểm dừng thư viện lưu động của thành phố",
    },
    "MVH": {
        "en": "county public-workshop equipment desk",
        "vi": "quầy thiết bị hội thảo công cộng cấp huyện",
    },
    "MVI": {
        "en": "regional neighborhood-event permit office",
        "vi": "văn phòng cấp phép sự kiện khu dân cư của khu vực",
    },
}

R1_ROUTES = {
    "MVD": {
        "en": ("maple path", "quartz path", "tide path"),
        "vi": ("lối phong", "lối thạch anh", "lối thủy triều"),
    },
    "MVE": {
        "en": ("copper path", "orchid path", "ridge path"),
        "vi": ("lối đồng", "lối lan", "lối sườn núi"),
    },
    "MVF": {
        "en": ("spruce path", "amber path", "delta path"),
        "vi": ("lối vân sam", "lối hổ phách", "lối châu thổ"),
    },
    "MVG": {
        "en": ("clover path", "slate path", "harbor path"),
        "vi": ("lối cỏ ba lá", "lối đá phiến", "lối bến cảng"),
    },
    "MVH": {
        "en": ("birch path", "coral path", "summit path"),
        "vi": ("lối bạch dương", "lối san hô", "lối đỉnh núi"),
    },
    "MVI": {
        "en": ("willow path", "indigo path", "valley path"),
        "vi": ("lối liễu", "lối chàm", "lối thung lũng"),
    },
}

R1_STYLE = {
    "explicit": {
        "en": "The decisive instruction is explicit and unambiguous.",
        "vi": "Chỉ dẫn quyết định được nêu rõ và không mơ hồ.",
    },
    "noted": {
        "en": "An extra scheduling note is present but does not alter the decision.",
        "vi": "Có thêm một ghi chú về lịch nhưng ghi chú đó không làm thay đổi quyết định.",
    },
    "minimal": {
        "en": "Only the decision-bearing detail is recorded.",
        "vi": "Chỉ thông tin trực tiếp quyết định kết quả được ghi lại.",
    },
}

R1_QUESTION = {
    "choice": {
        "en": "Which handling path does the record assign?",
        "vi": "Hồ sơ chỉ định lối xử lý nào?",
    },
    "score": {
        "en": "Which priority level does the record state?",
        "vi": "Hồ sơ nêu mức ưu tiên nào?",
    },
    "noul": {
        "en": "Does the record require coordinator verification before completion?",
        "vi": "Hồ sơ có yêu cầu điều phối viên xác minh trước khi hoàn tất hay không?",
    },
}

R1_PRIORITY = (
    ("low", "thấp", 0.0),
    ("medium", "trung bình", 1.0),
    ("high", "cao", 2.0),
)


def _choice_options(domain: str, language: str) -> tuple[LogicalOption, ...]:
    paths = R1_ROUTES[domain][language]
    if language == "en":
        return tuple(
            LogicalOption(
                option_id=f"{domain.lower()}-path-{index}",
                criterion_text=f"The assigned handling path is the {path}.",
                aliases=(f"Route this request through the {path}.",),
            )
            for index, path in enumerate(paths)
        )
    return tuple(
        LogicalOption(
            option_id=f"{domain.lower()}-path-{index}",
            criterion_text=f"Lối xử lý được chỉ định là {path}.",
            aliases=(f"Hãy chuyển yêu cầu này qua {path}.",),
        )
        for index, path in enumerate(paths)
    )


def _score_options(domain: str, language: str) -> tuple[LogicalOption, ...]:
    if language == "en":
        return tuple(
            LogicalOption(
                option_id=f"{domain.lower()}-priority-{index}",
                criterion_text=f"The stated priority level is {en}.",
                aliases=(f"Priority band {index} means {en}.",),
                value=value,
            )
            for index, (en, _vi, value) in enumerate(R1_PRIORITY)
        )
    return tuple(
        LogicalOption(
            option_id=f"{domain.lower()}-priority-{index}",
            criterion_text=f"Mức ưu tiên được nêu là {vi}.",
            aliases=(f"Mức ưu tiên {index} tương ứng với mức {vi}.",),
            value=value,
        )
        for index, (_en, vi, value) in enumerate(R1_PRIORITY)
    )


def _noul_options(domain: str, language: str) -> tuple[LogicalOption, ...]:
    if language == "en":
        return (
            LogicalOption(
                option_id=f"{domain.lower()}-verify-no",
                criterion_text="Coordinator verification is not required.",
                aliases=("The request may finish without coordinator verification.",),
                value=0.0,
            ),
            LogicalOption(
                option_id=f"{domain.lower()}-verify-yes",
                criterion_text="Coordinator verification is required.",
                aliases=("A coordinator must verify the request before completion.",),
                value=1.0,
            ),
        )
    return (
        LogicalOption(
            option_id=f"{domain.lower()}-verify-no",
            criterion_text="Không cần điều phối viên xác minh.",
            aliases=("Yêu cầu có thể hoàn tất mà không cần điều phối viên xác minh.",),
            value=0.0,
        ),
        LogicalOption(
            option_id=f"{domain.lower()}-verify-yes",
            criterion_text="Cần điều phối viên xác minh.",
            aliases=("Điều phối viên phải xác minh yêu cầu trước khi hoàn tất.",),
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
    raise ValueError(f"unsupported M3-R1 primitive: {primitive}")


def _state(
    domain: str,
    primitive: Primitive,
    target: int,
    style: str,
    variant: int,
    language: str,
) -> str:
    context = R1_CONTEXT[domain][language]
    style_text = R1_STYLE[style][language]

    if primitive == "choice":
        path = R1_ROUTES[domain][language][target]
        if language == "en":
            fact = (
                f"The record assigns the {path} and explicitly excludes the two other listed paths."
            )
        else:
            fact = (
                f"Hồ sơ chỉ định {path} và loại trừ rõ hai lối còn lại trong danh sách."
            )
    elif primitive == "score":
        en, vi, _ = R1_PRIORITY[target]
        if language == "en":
            fact = f"The record states that the request has {en} priority."
        else:
            fact = f"Hồ sơ nêu rằng yêu cầu có mức ưu tiên {vi}."
    elif primitive == "noul":
        if language == "en":
            fact = (
                "The record requires coordinator verification before the request is completed."
                if target == 1
                else "The record states that coordinator verification is not required."
            )
        else:
            fact = (
                "Hồ sơ yêu cầu điều phối viên xác minh trước khi yêu cầu được hoàn tất."
                if target == 1
                else "Hồ sơ nêu rằng không cần điều phối viên xác minh."
            )
    else:
        raise ValueError(f"unsupported M3-R1 primitive: {primitive}")

    if language == "en":
        return f"At the {context}, bilingual training entry {variant:02d} is recorded. {fact} {style_text}"
    return f"Tại {context}, mục huấn luyện song ngữ {variant:02d} được ghi lại. {fact} {style_text}"


def _generate_domain(domain: str, partition: str) -> tuple[M3PairedCase, ...]:
    rows: list[M3PairedCase] = []
    primitives: tuple[Primitive, ...] = ("choice", "score", "noul")
    styles = ("explicit", "noted", "minimal")

    for primitive_index, primitive in enumerate(primitives):
        k = 2 if primitive == "noul" else 3
        en_options = _options(domain, primitive, "en")
        vi_options = _options(domain, primitive, "vi")
        if tuple(o.option_id for o in en_options) != tuple(o.option_id for o in vi_options):
            raise RuntimeError("M3-R1 EN/VI option identity changed")
        if tuple(o.value for o in en_options) != tuple(o.value for o in vi_options):
            raise RuntimeError("M3-R1 EN/VI typed values changed")

        for style_index, style in enumerate(styles):
            for repeat in range(4):
                target = (repeat + 2 * style_index + primitive_index) % k
                variant = primitive_index * 12 + style_index * 4 + repeat
                rows.append(
                    M3PairedCase(
                        pair_id=f"m3r1-{domain.lower()}-{primitive}-{variant:02d}",
                        partition=partition,
                        domain_id=domain,
                        primitive=primitive,
                        gold_index=target,
                        en_state_text=_state(domain, primitive, target, style, variant, "en"),
                        vi_state_text=_state(domain, primitive, target, style, variant, "vi"),
                        en_question_text=R1_QUESTION[primitive]["en"],
                        vi_question_text=R1_QUESTION[primitive]["vi"],
                        en_options=en_options,
                        vi_options=vi_options,
                    )
                )
    if len(rows) != 36:
        raise RuntimeError("M3-R1 authority requires 36 pairs/domain")
    return tuple(rows)


def generate_m3_r1_authority(partition: str) -> tuple[M3PairedCase, ...]:
    if partition not in M3_R1_PARTITIONS:
        raise ValueError(f"unknown M3-R1 partition: {partition}")
    rows: list[M3PairedCase] = []
    for domain in M3_R1_PARTITIONS[partition]:
        rows.extend(_generate_domain(domain, partition))
    return tuple(rows)


def all_m3_r1_text_atoms() -> set[str]:
    values: set[str] = set()
    for partition in M3_R1_PARTITIONS:
        for row in generate_m3_r1_authority(partition):
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
    "M3_R1_PARTITIONS",
    "R1_CONTEXT",
    "R1_PRIORITY",
    "R1_QUESTION",
    "R1_ROUTES",
    "R1_STYLE",
    "all_m3_r1_text_atoms",
    "generate_m3_r1_authority",
]
