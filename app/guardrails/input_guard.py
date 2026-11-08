"""Task 10 — input-side guardrails.

1. PII masking: Indian mobile numbers (the scenario's fixed-format PII field). Applied BEFORE the text
   reaches agents, memory, cache, or logs (Task 12 requires the same masking for logs).
   Out of scope (acknowledged in the brief): patient name, diagnosis/condition and insurance ID are
   free text with no universal format, so a keyless regex masker cannot match them reliably.
2. Prompt-injection detection: pattern matching on normalized text. Detected input is rejected, not
   sanitized. Pattern lists are bypassable; this is a baseline control, not a guarantee.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

PHONE_MASK = "[PHONE_MASKED]"

# Optional +91 / 91 / 0 prefix, then 10 digits starting 6-9, optionally split 5+5 by space or hyphen.
_PHONE_RE = re.compile(r"(?<![\w+])(?:(?:\+|00)?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)

_INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("override_instructions", re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|all|any|your|the)\b"
        r".{0,20}\b(instruction|instructions|rules|prompt|prompts|guidelines|policies)\b")),
    ("reveal_prompt", re.compile(
        r"\b(reveal|show|print|repeat|output|leak|tell me)\b.{0,30}\b(system|hidden|initial|original)\b"
        r".{0,15}\b(prompt|instructions|message)\b")),
    ("role_hijack", re.compile(r"\b(you are now|from now on you are|act as|pretend to be|roleplay as)\b")),
    ("jailbreak_mode", re.compile(r"\b(developer mode|dan mode|jailbreak|god mode|unrestricted mode)\b")),
    ("fake_role_tag", re.compile(r"(<\s*/?\s*(system|assistant|developer)\s*>|^\s*(system|assistant)\s*:|###\s*instruction)",
                                 re.MULTILINE)),
    ("data_exfiltration", re.compile(
        r"\b(list|dump|export|show)\b.{0,20}\b(all|every|other)\b.{0,20}\b(patients?|appointments?|records?)\b")),
]


@dataclass
class InputGuardResult:
    masked_text: str
    pii_masked: int = 0
    injection_detected: bool = False
    injection_rules: list[str] = field(default_factory=list)

    @property
    def events(self) -> list[str]:
        events = [f"pii_masked:{self.pii_masked}"] if self.pii_masked else []
        return events + [f"injection:{rule}" for rule in self.injection_rules]


class InputRejectedError(Exception):
    def __init__(self, result: InputGuardResult):
        self.result = result
        super().__init__(f"Input rejected by guardrail: {', '.join(result.injection_rules)}")


def mask_pii(text: str) -> tuple[str, int]:
    masked, count = _PHONE_RE.subn(PHONE_MASK, text)
    return masked, count


def normalize_for_detection(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).casefold()
    return re.sub(r"[ \t]+", " ", text)


def detect_injection(text: str) -> list[str]:
    normalized = normalize_for_detection(text)
    return [name for name, pattern in _INJECTION_PATTERNS if pattern.search(normalized)]


def guard_input(text: str) -> InputGuardResult:
    """Mask first, then detect on the masked text; the raw text is never returned or stored."""
    masked, count = mask_pii(text)
    rules = detect_injection(masked)
    return InputGuardResult(masked_text=masked, pii_masked=count, injection_detected=bool(rules), injection_rules=rules)
