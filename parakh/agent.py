"""Agents under test.

An agent is anything with `start()` and `reply(transcript)`. This file has a
baseline rule-based COD (cash-on-delivery) confirmation agent. It makes results
reproducible and costs nothing to run; an LLM or hosted agent can replace it by
implementing the same two methods.
"""

from dataclasses import dataclass, field
from typing import Protocol

from .normalize import detect_day, detect_intent, detect_yes_no, extract_digits

MAX_RETRIES = 2


@dataclass
class AgentTurn:
    state: str  # what the agent is now waiting for; see the caller replies in the scenarios
    say: str
    readback: str | None = None  # value the agent read aloud, which the caller checks


@dataclass
class Result:
    outcome: str | None = None  # confirmed | address_changed | cancelled | rescheduled | handoff
    pincode: str | None = None
    day: str | None = None
    log: list = field(default_factory=list)


class Agent(Protocol):
    result: Result

    def start(self) -> AgentTurn: ...
    def reply(self, transcript: str) -> AgentTurn | None: ...  # None means the call has ended


class RuleBasedCODAgent:
    """Confirm the order, read back any pincode, and hand off to a human when unsure."""

    def __init__(self):
        self.result = Result()
        self.state = "confirm_order"
        self.retries = 0
        self.pending_outcome = "confirmed"
        self.candidate = None

    def start(self) -> AgentTurn:
        return AgentTurn("confirm_order", "Namaste! Aapka COD order confirm karna hai?")

    def _retry(self, say: str) -> AgentTurn | None:
        self.retries += 1
        if self.retries > MAX_RETRIES:
            return self._finish("handoff")
        return AgentTurn(self.state, say, self.candidate if self.state == "readback" else None)

    def _goto(self, state: str, say: str, readback: str | None = None) -> AgentTurn:
        self.state, self.retries = state, 0
        return AgentTurn(state, say, readback)

    def _finish(self, outcome: str) -> None:
        self.result.outcome = outcome
        self.state = "done"
        return None

    def reply(self, transcript: str) -> AgentTurn | None:
        handler = getattr(self, f"_on_{self.state}")
        return handler(transcript)

    def _on_confirm_order(self, text):
        intent = detect_intent(text)
        if intent == "confirm":
            self.pending_outcome = "confirmed"
            return self._goto("ask_pincode", "Delivery ka pincode bataiye.")
        if intent == "change_address":
            self.pending_outcome = "address_changed"
            return self._goto("ask_pincode", "Naya pincode bataiye.")
        if intent in ("cancel", "no"):
            return self._goto("confirm_cancel", "Kya aap pakka order cancel karna chahte hain?")
        if intent == "reschedule":
            return self._goto("ask_day", "Kis din delivery karein?")
        return self._retry("Maaf kijiye, kya aapka order bhej dein?")

    def _on_ask_pincode(self, text):
        digits = extract_digits(text)
        if len(digits) == 6:
            self.candidate = digits
            return self._goto("readback", f"Aapka pincode {' '.join(digits)} hai, sahi?", digits)
        return self._retry("Pincode ke chhah ank dobara bataiye.")

    def _on_readback(self, text):
        answer = detect_yes_no(text)
        if answer == "yes":
            self.result.pincode = self.candidate
            return self._finish(self.pending_outcome)
        if answer == "no":
            self.retries += 1
            if self.retries > MAX_RETRIES:
                return self._finish("handoff")
            self.state = "ask_pincode"
            return AgentTurn("ask_pincode", "Maaf kijiye, pincode dobara bataiye.")
        return self._retry(f"Pincode {' '.join(self.candidate)} sahi hai?")

    def _on_confirm_cancel(self, text):
        answer = detect_yes_no(text)
        if answer == "yes":
            return self._finish("cancelled")
        if answer == "no":
            return self._goto("confirm_order", "Theek hai. Kya order bhej dein?")
        return self._retry("Haan ya nahi, order cancel karein?")

    def _on_ask_day(self, text):
        day = detect_day(text)
        if day:
            self.result.day = day
            return self._finish("rescheduled")
        return self._retry("Kis din delivery karein?")
