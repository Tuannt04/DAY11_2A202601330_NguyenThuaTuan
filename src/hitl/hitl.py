"""
Lab 11 — Part 4: Human-in-the-Loop Design
  TODO 11: Confidence Router
  TODO 12: Design 3 HITL decision points + a runnable review lifecycle
"""
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ============================================================
# TODO 11: Implement ConfidenceRouter
#
# Route agent responses based on confidence scores:
#   - HIGH (>= 0.9): Auto-send to user
#   - MEDIUM (0.7 - 0.9): Queue for human review
#   - LOW (< 0.7): Escalate to human immediately
#
# Special case: if the action is HIGH_RISK (e.g., money transfer,
# account deletion), ALWAYS escalate regardless of confidence.
#
# Implement the route() method.
# ============================================================

HIGH_RISK_ACTIONS = [
    "transfer_money",
    "close_account",
    "change_password",
    "delete_data",
    "update_personal_info",
]


# ============================================================
# action_type classification — text -> HIGH_RISK_ACTIONS (or "general")
#
# ConfidenceRouter.route() takes action_type as a parameter — it does not
# derive it. Somewhere upstream has to turn the customer's raw message into
# that string, or "HIGH_RISK_ACTIONS always escalate" only ever fires in
# tests that hand it a pre-labelled action_type by hand, never on anything
# a real user typed. This is that upstream step: a deliberately narrow,
# imperative-intent classifier (same "regex is one signal, not the whole
# boundary" spirit as guardrails/input_guardrails.detect_injection).
#
# Precision matters as much as recall here: an *informational* question
# ("How do I transfer money to another account?") must NOT classify as an
# actual transfer_money action — SAFE_QUERIES already contains exactly that
# question, and mis-escalating it would be a false positive on a perfectly
# normal support query. So patterns require either an explicit first-person
# imperative ("I want to...", "please...") or a concrete amount+account,
# not just the presence of a verb like "transfer".
# ============================================================

_ACTION_INTENT_PATTERNS = {
    "transfer_money": (
        r"\b(i\s+want\s+to|i'd\s+like\s+to|please)\s+transfer\b",
        r"\btransfer\s+[\d,.]+.*\bto\s+account\b",
        r"chuy[eể]n\b.{0,20}?[\d.,]+\s*(vn[dđ]|tri[eệ]u|đ)\b",
    ),
    "close_account": (
        r"\b(i\s+want\s+to|i'd\s+like\s+to|please)\s+close\s+(my\s+)?account\b",
        r"đóng\s+(gi[uú]p\s+)?t[aà]i\s+kho[aả]n",
    ),
    "change_password": (
        r"\b(i\s+want\s+to|i'd\s+like\s+to|please)\s+change\s+(my\s+)?password\b",
        r"đổi\s+(gi[uú]p\s+)?m[aậ]t\s+kh[aẩ]u",
    ),
    "update_personal_info": (
        r"\b(i\s+want\s+to|i'd\s+like\s+to|please)\s+update\s+my\s+(phone|address|email)\b",
        r"c[aậ]p\s+nh[aậ]t\s+(gi[uú]p\s+)?(s[oố]\s+)?(đi[eệ]n\s+tho[aạ]i|đ[iị]a\s+ch[iỉ])",
    ),
    "delete_data": (
        r"\b(i\s+want\s+to|i'd\s+like\s+to|please)\s+delete\s+(my\s+)?(data|account|history)\b",
        r"xóa\s+(gi[uú]p\s+)?(d[uữ]\s+li[eệ]u|t[aà]i\s+kho[aả]n)",
    ),
}


def classify_action_type(text: str) -> str:
    """Return a HIGH_RISK_ACTIONS label if `text` reads as an imperative
    request to perform that action, else 'general'. Deliberately narrow —
    false negatives on unusual phrasing are expected (same trade-off as
    every other regex layer in this lab); the goal is to stop treating
    ConfidenceRouter's HIGH_RISK_ACTIONS branch as untestable against real
    text, not to build a full intent classifier.
    """
    lowered = text.lower()
    for action_type, patterns in _ACTION_INTENT_PATTERNS.items():
        if any(re.search(p, lowered, re.IGNORECASE) for p in patterns):
            return action_type
    return "general"


@dataclass
class RoutingDecision:
    """Result of the confidence router."""
    action: str          # "auto_send", "queue_review", "escalate"
    confidence: float
    reason: str
    priority: str        # "low", "normal", "high"
    requires_human: bool


class ConfidenceRouter:
    """Route agent responses based on confidence and risk level.

    Thresholds:
        HIGH:   confidence >= 0.9 -> auto-send
        MEDIUM: 0.7 <= confidence < 0.9 -> queue for review
        LOW:    confidence < 0.7 -> escalate to human

    High-risk actions always escalate regardless of confidence.
    """

    HIGH_THRESHOLD = 0.9
    MEDIUM_THRESHOLD = 0.7

    def route(self, response: str, confidence: float,
              action_type: str = "general") -> RoutingDecision:
        """Route a response based on confidence score and action type.

        Args:
            response: The agent's response text
            confidence: Confidence score between 0.0 and 1.0
            action_type: Type of action (e.g., "general", "transfer_money")

        Returns:
            RoutingDecision with routing action and metadata
        """
        if action_type in HIGH_RISK_ACTIONS:
            return RoutingDecision(
                action="escalate",
                confidence=confidence,
                reason=f"High-risk action: {action_type}",
                priority="high",
                requires_human=True,
            )

        if confidence >= self.HIGH_THRESHOLD:
            return RoutingDecision(
                action="auto_send",
                confidence=confidence,
                reason="High confidence",
                priority="low",
                requires_human=False,
            )

        if confidence >= self.MEDIUM_THRESHOLD:
            return RoutingDecision(
                action="queue_review",
                confidence=confidence,
                reason="Medium confidence — needs review",
                priority="normal",
                requires_human=True,
            )

        return RoutingDecision(
            action="escalate",
            confidence=confidence,
            reason="Low confidence — escalating",
            priority="high",
            requires_human=True,
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# TODO 12 (runnable half): Review lifecycle
#
# ConfidenceRouter only decides *that* a human is needed. This is the part
# that actually models a reviewer's queue: a ticket carrying the intent +
# proposed diff + correlation ID (audit_fields from hitl_decision_points
# below), with approve/reject/timeout transitions — timeout is fail-closed
# (never silently sent), matching approval_path in every decision point.
# ============================================================

@dataclass
class ReviewTicket:
    """A pending human-review request. `ticket_id` matches the approval_id
    format the reference boundary (agents/security_boundary.py) requires
    before authorizing a high-risk action: HITL-XXXXXXXX."""

    ticket_id: str
    correlation_id: str
    intent: str
    proposed_action: str
    context: dict = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)
    _created_ts: float = field(default_factory=time.time, repr=False)
    status: str = "pending"  # pending | approved | rejected | timeout
    reviewer_id: str | None = None
    decision_at: str | None = None
    decision_reason: str | None = None

    def audit_row(self) -> dict:
        """The audit_fields a decision point promises to log: correlation
        ID, intent, proposed action, reviewer, decision, timestamps."""
        return {
            "ticket_id": self.ticket_id,
            "correlation_id": self.correlation_id,
            "intent": self.intent,
            "proposed_action": self.proposed_action,
            "status": self.status,
            "reviewer_id": self.reviewer_id,
            "created_at": self.created_at,
            "decision_at": self.decision_at,
            "decision_reason": self.decision_reason,
        }


class ReviewQueue:
    """In-memory HITL review queue with an approve/reject/timeout lifecycle.

    A real deployment backs this with a ticketing system / DB; the state
    machine and audit trail (what this lab grades) are framework-agnostic.
    """

    def __init__(self, *, timeout_seconds: float = 600):
        self.timeout_seconds = timeout_seconds
        self.tickets: dict[str, ReviewTicket] = {}

    def submit(
        self, *, correlation_id: str, intent: str, proposed_action: str,
        context: dict | None = None,
    ) -> ReviewTicket:
        """Open a ticket — called whenever ConfidenceRouter.route() returns
        requires_human=True. Never auto-sends; the action stays proposed
        until a reviewer (or a timeout) resolves it."""
        ticket = ReviewTicket(
            ticket_id=f"HITL-{uuid.uuid4().hex[:8].upper()}",
            correlation_id=correlation_id,
            intent=intent,
            proposed_action=proposed_action,
            context=context or {},
        )
        self.tickets[ticket.ticket_id] = ticket
        return ticket

    def _pending(self, ticket_id: str) -> ReviewTicket:
        ticket = self.tickets.get(ticket_id)
        if ticket is None:
            raise KeyError(f"no such ticket: {ticket_id}")
        if ticket.status != "pending":
            raise ValueError(f"ticket {ticket_id} already resolved: {ticket.status}")
        return ticket

    def approve(self, ticket_id: str, *, reviewer_id: str) -> ReviewTicket:
        ticket = self._pending(ticket_id)
        ticket.status = "approved"
        ticket.reviewer_id = reviewer_id
        ticket.decision_at = _utc_now_iso()
        return ticket

    def reject(self, ticket_id: str, *, reviewer_id: str, reason: str) -> ReviewTicket:
        ticket = self._pending(ticket_id)
        ticket.status = "rejected"
        ticket.reviewer_id = reviewer_id
        ticket.decision_at = _utc_now_iso()
        ticket.decision_reason = reason
        return ticket

    def check_timeout(self, ticket_id: str, *, now: float | None = None) -> ReviewTicket:
        """Fail-closed: a ticket nobody acted on within the SLA becomes
        'timeout', never silently 'approved'."""
        ticket = self.tickets[ticket_id]
        if ticket.status == "pending":
            elapsed = (now if now is not None else time.time()) - ticket._created_ts
            if elapsed >= self.timeout_seconds:
                ticket.status = "timeout"
                ticket.decision_at = _utc_now_iso()
                ticket.decision_reason = f"no reviewer action within {self.timeout_seconds}s SLA"
        return ticket


# ============================================================
# TODO 12: Design 3 HITL decision points + a review lifecycle
#
# For each decision point, define:
# - trigger: What condition activates this HITL check?
# - hitl_model: Which model? (human-in-the-loop, human-on-the-loop,
#   human-as-tiebreaker)
# - context_needed: What info does the human reviewer need?
# - example: A concrete scenario
# - approval_path: What approve/reject/timeout decision is recorded?
# - audit_fields: Which correlation ID, intent and proposed action/diff are logged?
#
# Think about real banking scenarios where human judgment is critical.
# ============================================================

hitl_decision_points = [
    {
        "id": 1,
        "name": "High-value money transfer",
        "trigger": (
            "action_type == 'transfer_money' (any amount) — HIGH_RISK_ACTIONS always "
            "escalate regardless of the model's confidence score."
        ),
        "hitl_model": "human-in-the-loop",
        "context_needed": (
            "Source account, destination account/beneficiary, amount, currency, the "
            "customer's original request text, and the agent's proposed action payload."
        ),
        "example": (
            "Customer asks the agent to 'transfer 500,000,000 VND to account 0123456789 "
            "at another bank'. The agent drafts the transfer but a human reviewer must "
            "approve it before any egress call is made — the LLM only proposes, it never "
            "authorizes a fund movement."
        ),
        "approval_path": (
            "approve: reviewer confirms beneficiary + amount match customer intent -> "
            "transfer proceeds, approval_id recorded. reject: reviewer flags mismatch or "
            "suspected fraud -> transfer cancelled, customer notified. timeout (no reviewer "
            "action within SLA, e.g. 10 min): fail closed — action is NOT auto-sent, "
            "customer is told the request needs manual follow-up."
        ),
        "audit_fields": (
            "correlation/request_id, customer intent (raw text), proposed action + diff "
            "(destination, amount before/after any correction), reviewer_id, decision "
            "(approve/reject/timeout), decision_timestamp, approval_id."
        ),
    },
    {
        "id": 2,
        "name": "Account-changing action (close account / change password / update personal info)",
        "trigger": (
            "action_type in {'close_account', 'change_password', 'update_personal_info', "
            "'delete_data'} — irreversible or hard-to-reverse account changes."
        ),
        "hitl_model": "human-in-the-loop",
        "context_needed": (
            "Customer identity verification status, the specific field being changed "
            "(old value -> new value diff), channel the request came from (chat vs "
            "authenticated app), and any risk signals (e.g. request right after a failed "
            "login)."
        ),
        "example": (
            "A chat message says 'please close my account and update my phone number to "
            "0999999999'. Because this changes account state, the agent cannot execute it "
            "directly; it opens a review ticket with the old/new values for a human agent "
            "to verify against the customer's identity record."
        ),
        "approval_path": (
            "approve: identity confirmed, change applied, customer notified via a verified "
            "channel. reject: identity mismatch or suspicious pattern -> change blocked, "
            "security team notified. timeout: request expires and customer must resubmit "
            "through an authenticated channel — never silently applied."
        ),
        "audit_fields": (
            "correlation/request_id, field diff (before/after), identity-verification "
            "evidence referenced, reviewer_id, decision, decision_timestamp, notification "
            "sent (Y/N)."
        ),
    },
    {
        "id": 3,
        "name": "Low-confidence / ambiguous or untrusted-source response",
        "trigger": (
            "confidence < MEDIUM_THRESHOLD (0.7), OR the response was built using "
            "untrusted external content (email/RAG/web) that the input guardrail flagged "
            "as borderline, OR the LLM judge returns a FAIL verdict."
        ),
        "hitl_model": "human-on-the-loop",
        "context_needed": (
            "The generated response, the confidence/judge scores that triggered escalation, "
            "and — if applicable — the untrusted source document and why it was flagged."
        ),
        "example": (
            "Customer asks about a promotion mentioned in a forwarded email. The agent's "
            "draft reply cites unverifiable claims from that email; confidence is low, so "
            "the reply is queued for a human to edit or approve before it goes out, rather "
            "than being sent automatically."
        ),
        "approval_path": (
            "approve: reviewer edits or confirms the reply, then it is sent. reject: "
            "reviewer discards the draft and the agent falls back to a safe templated "
            "response. timeout: safe fallback response is sent automatically and the "
            "case is logged for later review (fail-safe, not fail-open)."
        ),
        "audit_fields": (
            "correlation/request_id, confidence score, judge verdict, source provenance "
            "(trusted/untrusted + source id), reviewer_id, decision, decision_timestamp, "
            "final text sent."
        ),
    },
]


# ============================================================
# Quick tests
# ============================================================

def test_confidence_router():
    """Test ConfidenceRouter with sample scenarios."""
    router = ConfidenceRouter()

    test_cases = [
        ("Balance inquiry", 0.95, "general"),
        ("Interest rate question", 0.82, "general"),
        ("Ambiguous request", 0.55, "general"),
        ("Transfer $50,000", 0.98, "transfer_money"),
        ("Close my account", 0.91, "close_account"),
    ]

    print("Testing ConfidenceRouter:")
    print("=" * 80)
    print(f"{'Scenario':<25} {'Conf':<6} {'Action Type':<18} {'Decision':<15} {'Priority':<10} {'Human?'}")
    print("-" * 80)

    for scenario, conf, action_type in test_cases:
        decision = router.route(scenario, conf, action_type)
        print(
            f"{scenario:<25} {conf:<6.2f} {action_type:<18} "
            f"{decision.action:<15} {decision.priority:<10} "
            f"{'Yes' if decision.requires_human else 'No'}"
        )

    print("=" * 80)


def test_hitl_points():
    """Display HITL decision points."""
    print("\nHITL Decision Points:")
    print("=" * 60)
    for point in hitl_decision_points:
        print(f"\n  Decision Point #{point['id']}: {point['name']}")
        print(f"    Trigger:  {point['trigger']}")
        print(f"    Model:    {point['hitl_model']}")
        print(f"    Context:  {point['context_needed']}")
        print(f"    Example:  {point['example']}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_confidence_router()
    test_hitl_points()
