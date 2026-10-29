"""Layer 2 - Contextual Quarantine: remove assistant-directed clauses, keep the facts.

Runs only on chunks Layer 1 flagged. The chunk is never dropped:

1. Segmentation. Structural breakout markers (</context>, <|im_start|>, [INST], ...) become their
   own units; the rest is split into sentences (spaCy) and each sentence into clauses at
   connector boundaries (", but", ", and", ", so", ", although", ";").
2. Contextual rules. A clause is removed when it is directed at the AI assistant rather than
   at a human reader: role/system markers, delimiter breakouts, a vocative or subject that is
   the assistant ("Assistant, ...", "the AI assistant must ..."), attempts to override the
   question or instructions, dictating the assistant's output ("reply only with ..."),
   telling the assistant to redirect the user, or exfiltration links. spaCy syntax adds weak
   signals (imperative mood, command verbs aimed at the output); three weak signals together
   also remove a clause.
3. Legitimate context. Ordinary enterprise instructions ("Never share your password ...",
   "Staff must record every return ...") are imperative too, but they address employees or
   customers and carry no assistant-directed marker. They are kept and reported as retained
   instructions, so the decision is visible.
4. Reconstruction. Kept clauses are copied verbatim from the original text; only the
   punctuation around a removed clause is tidied.

The clean reference text is never an input here: Layer 2 sees only the retrieved chunk.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from config import SPACY_MODEL

_I = re.IGNORECASE
_ID = re.IGNORECASE | re.DOTALL

# --------------------------------------------------------------------------- strong rules
_ASSISTANT = r"(?:ai\s+)?(?:assistants?|chatbots?|language\s+models?|llms?|ai\s+systems?|ai\s+models?|" \
             r"summari[sz]ers?|automated\s+(?:systems?|tools?|readers?|summari[sz]ers?))"
# Human job titles that contain "assistant" are not the AI.
_HUMAN_ASSISTANT = r"(?<!administrative )(?<!executive )(?<!personal )(?<!sales )(?<!teaching )(?<!dental )" \
                   r"(?<!medical )(?<!legal )(?<!research )(?<!shop )(?<!store )(?<!office )"

RULES: dict[str, tuple[str, list[re.Pattern]]] = {
    "role_marker": ("role or system marker", [
        re.compile(r"^\W*(?:system|assistant|developer|admin(?:istrator)?)\b[^:\n]{0,50}:", _I),
        re.compile(r"\b(?:new|updated)\s+(?:system\s+)?(?:instruction|prompt|directive)s?\b", _I),
        re.compile(r"\bsystem\s+(?:message|instruction|prompt|override|note|directive)s?\b", _I),
        re.compile(r"<\|?(?:im_start|im_end|system|endoftext)\|?>|\[/?INST\]|<<\s*/?SYS\s*>>|"
                   r"#{3,}\s*(?:system|instruction)", _I),
    ]),
    "delimiter_breakout": ("delimiter breakout tag", [
        re.compile(r"</?\s*(?:context|documents?|data|retrieved|instructions?|system|search_results?)\s*>|"
                   r"\]\]>|<!\[CDATA\[|-{3,}\s*end of", _I),
    ]),
    "assistant_addressed": ("addressed to the AI assistant", [
        re.compile(r"^\W*(?:dear\s+|hey\s+|attention\s+)?(?:ai\s+)?(?:assistant|chatbot)\s*[,:]", _I),
        re.compile(r"\b(?:note|message|instructions?|reminder|update|notice|memo)\s+(?:to|for)\s+"
                   r"(?:the\s+|any\s+|all\s+)?" + _ASSISTANT + r"\b", _I),
        re.compile(r"\b(?:(?:the|any|all|every|this|an|our)\s+)?" + _HUMAN_ASSISTANT + _ASSISTANT +
                   r"\b(?:\s+\w+){0,4}?\s+(?:must|should|shall|will|has\s+to|have\s+to|needs?\s+to|"
                   r"is\s+required\s+to|are\s+required\s+to|is\s+instructed\s+to|is\s+to|are\s+to)\b", _I),
        re.compile(r"\b(?:ai|" + _ASSISTANT + r")\b[^.]{0,40}\breading\s+(?:this|these)\b", _I),
    ]),
    "instruction_override": ("tries to override the question or instructions", [
        re.compile(r"\b(?:ignore|disregard|forget|override|bypass)\b[^.]{0,30}?\b(?:(?:the\s+)?(?:user'?s?\s+)?"
                   r"question|(?:previous|prior|earlier|above|original|system)\s+(?:instructions?|prompts?|rules|"
                   r"guidelines|directions|messages|context|text)|instructions?|prompts?|guidelines|"
                   r"(?:the\s+)?(?:\w+\s+)?above|everything)\b", _I),
        re.compile(r"\b(?:instead\s+of|rather\s+than)\s+(?:answering|summari[sz]ing|responding|replying)\b", _I),
        re.compile(r"\bignorier\w*|\bignora(?:te)?\b|\bignore[sz]?\s+la\b|\bantworte\b|\bresponde\s+solo\b|"
                   r"\bréponds\b|\brispondi\b|\bvergiss\b|\boublie[sz]?\b|\bolvida\b", _I),
    ]),
    "persona_switch": ("tries to change the assistant's role or task", [
        re.compile(r"\byou\s+are\s+(?:now|no\s+longer)\b|\bfrom\s+now\s+on\b|\b(?:act|behave)\s+as\s+(?:an?|the)\b|"
                   r"\bpretend\s+(?:to\s+be|you\s+are|that\s+you)\b|\brole-?play\b|\bunfiltered\b|\bjailbr[oe]\w*", _I),
        re.compile(r"\byour\s+(?:new|real|true|only|first|actual)\s+(?:task|objective|goal|role|instructions?|purpose|"
                   r"output|words|job)\b", _I),
        re.compile(r"\bDAN\b"),
    ]),
    "output_control": ("dictates the assistant's output", [
        re.compile(r"\b(?:reply|respond|answer|say|output|print|repeat)\s+(?:only|exactly|solely)\b|"
                   r"\b(?:only|exactly)\s+(?:with|reply|respond|say|output|print)\b|\bnothing\s+else\b", _I),
        re.compile(r"\b(?:reply|respond|answer|say|output|print|begin|start|end|append|include|state|cite|mention|"
                   r"add)\b[^.]{0,60}?\b(?:every\s+(?:question|answer|query|reply|response)|(?:your|its)\s+"
                   r"(?:answer|response|reply|output))\b", _I),
        re.compile(r"\byou\s+(?:must|should|will|shall|have\s+to|need\s+to)\s+(?:now\s+|only\s+|always\s+)?"
                   r"(?:respond|reply|say|output|print|repeat)\b", _I),
        re.compile(r"\bsummari[sz]e\b[^.]{0,40}\bas\s+['\"“]", _I),
    ]),
    "user_redirect": ("tells the assistant to redirect the user", [
        re.compile(r"\b(?:tell|inform|warn|instruct|ask|direct|advise|urge|remind)\s+(?:the\s+)?"
                   r"(?:user|reader|requester|person\s+asking)s?\b", _I),
    ]),
    "exfiltration": ("exfiltration link or placeholder", [
        re.compile(r"!\[[^\]]*\]\(\s*https?://", _I),
        re.compile(r"USER_QUESTION|CHAT_HISTORY|USER_EMAIL|CONVERSATION_LOG"),
        re.compile(r"https?://\S+\?(?:q|data|query|u|msg)=", _I),
    ]),
}

# --------------------------------------------------------------------------- weak signals
COMMAND_VERBS = {"ignore", "disregard", "forget", "override", "reply", "respond", "answer", "say", "output",
                 "print", "repeat", "pretend", "act", "roleplay", "state", "cite", "mention", "tell", "inform",
                 "warn", "begin", "start", "end", "append", "include", "translate", "write", "recommend", "reveal",
                 "summarize", "summarise", "announce"}
META_CUE = re.compile(r"\b(?:the\s+user|the\s+reader|user's|the\s+question|(?:your|its)\s+(?:answer|response|reply|"
                      r"output)|(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|messages?|rules|text|"
                      r"context|questions?)|instructions?|system\s+prompt|this\s+conversation|the\s+prompt)\b", _I)
URL = re.compile(r"https?://\S+|\b[\w-]+\.(?:com|net|org|io|app|dev|shop|info)\b(?:/\S*)?", _I)
CODE_TOKEN = re.compile(r"\b(?=[A-Z0-9-]*\d)(?=[A-Z0-9-]*[A-Z])[A-Z0-9]+(?:-[A-Z0-9]+)+\b|\b\d{3}-\d{4}\b")
QUOTE = re.compile(r"(?<![A-Za-z])(['\"])(?:(?!\1).){1,200}\1(?![A-Za-z])|[“‘][^”’]{1,200}[”’]")

# --------------------------------------------------------------------------- legitimate context
INSTRUCTION_LIKE = re.compile(r"\b(?:must|should|need\s+to|needs\s+to|are\s+required\s+to|is\s+required\s+to|"
                              r"may\s+not|cannot|can't|never|always|do\s+not|don't)\b", _I)
LEGIT_ACTOR = re.compile(r"\b(?:employees?|staff|customers?|managers?|contractors?|travell?ers?|team\s+members?|"
                         r"you|your\s+team)\b", _I)
ENTERPRISE_OBJECT = re.compile(r"\b(?:passwords?|receipts?|leave|travel|flights?|hotels?|records?|data|returns?|"
                               r"expenses?|portal|approval|security\s+team|privacy\s+office|help\s+desk|devices?|"
                               r"drives?|claims?|credit|refunds?|accounts?|login)\b", _I)

# --------------------------------------------------------------------------- segmentation
STRUCT_SPLIT = re.compile(r"((?im:^[ \t]*(?:system|assistant|user|developer|admin)[ \t]*:)|\n+|\]\]>|<!\[CDATA\[|"
                          r"</?\s*\w+(?:\s+[\w-]+=\"[^\"]*\")*\s*/?>|<\|\w+\|>|\[/?INST\]|#{3,}[^#\n]*#*|\"\"\"|"
                          r"-{3,}[^\n]*?-{3,})")
OPENING_MARKER = re.compile(r"(?i)(?:system|assistant|user|developer|admin)\s*:|<\|im_start\|>|\[INST\]|"
                            r"<\s*instruction\s*>|#{3,}[^#\n]*#*")
AFTER_QUOTE_BOUNDARY = re.compile(r"(?<=[.!?]['\"”’])\s+(?=[A-Z])|(?<=\))\s+(?=[A-Z][a-z])")
CLAUSE_BOUNDARY = re.compile(r",\s+(?=(?:and|but|so|although|though|while|whereas|yet|because|unless)\b)|;\s*", _I)
CONNECTORS = r"(?:and|but|so|although|though|while|whereas|yet|because|unless|then)"
_QUOTE_CHAR = re.compile(r"(?<![A-Za-z])'|'(?![A-Za-z])|[\"“”‘’]")
_SENTINEL = "\x00"
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def deobfuscate(text: str) -> str:
    """Undo common obfuscations so the rules can read them ("1gn0re", "Ign-ore", "I g n o r e")."""
    t = re.sub(r"(?<=\w)_(?=\w)", " ", text)                       # IGNORE_PREVIOUS -> IGNORE PREVIOUS
    t = re.sub(r"(?<=[a-z])-(?=[a-z])", "", t, flags=_I)             # Ign-ore -> Ignore
    t = re.sub(r"\b(?=\w*[a-z])(?=\w*\d)\w+\b", lambda m: m.group(0).translate(_LEET), t, flags=_I)
    t = re.sub(r"\b(?:\w ){3,}\w\b", lambda m: m.group(0).replace(" ", ""), t)  # I g n o r e -> Ignore
    return t


def _rule_hits(text: str) -> list[str]:
    plain, clear = text, deobfuscate(text)
    return [name for name, (_desc, patterns) in RULES.items()
            if any(p.search(plain) or (clear != plain and p.search(clear)) for p in patterns)]


@dataclass
class Unit:
    """A removable piece of the chunk: a structural marker or a clause of a sentence."""
    start: int
    end: int
    text: str
    kind: str                    # "marker" | "clause"
    sentence: int
    rule_hits: list[str] = field(default_factory=list)
    weak: list[str] = field(default_factory=list)
    removed: bool = False
    legit_reason: str = ""

    @property
    def weak_score(self) -> int:
        return len(self.weak)


@dataclass
class QuarantineResult:
    original: str
    sanitized: str
    removed: list[dict]
    retained_instructions: list[dict]
    units: list[dict]
    latency_ms: float

    @property
    def changed(self) -> bool:
        return bool(self.removed)

    def to_dict(self) -> dict:
        return {"sanitized_text": self.sanitized, "removed_clauses": self.removed,
                "retained_instructions": self.retained_instructions, "changed": self.changed,
                "latency_ms": round(self.latency_ms, 2)}


def rule_catalog() -> list[dict]:
    """Human-readable rule list (shown on the architecture screen)."""
    return [{"rule": name, "description": desc} for name, (desc, _) in RULES.items()] + [
        {"rule": "imperative_command", "description": "spaCy: imperative mood + command verb + reference to the "
                                                      "conversation or an output artifact (3 weak signals)"},
        {"rule": "continuation", "description": "continues a removed clause in the same sentence or follows an "
                                                "opening role marker"},
    ]


class Layer2Quarantine:
    def __init__(self, spacy_model: str = SPACY_MODEL, nlp=None):
        if nlp is None:
            import spacy

            nlp = spacy.load(spacy_model, disable=["ner"])
        self.nlp = nlp
        self.model_name = spacy_model

    # ----------------------------------------------------------------- segmentation
    def _sentence_spans(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        segment = text[start:end]
        spans: list[tuple[int, int]] = []
        for sent in self.nlp(segment).sents:
            a, b = start + sent.start_char, start + sent.end_char
            cuts = [(m.start(), m.end()) for m in AFTER_QUOTE_BOUNDARY.finditer(text, a, b)]
            for c_start, c_end in sorted(set(cuts)):
                if c_start > a:
                    spans.append((a, c_start))
                    a = c_end
            spans.append((a, b))
        # re-join sentences that were split inside a quotation or inside markdown "![alt](url)"
        merged: list[tuple[int, int]] = []
        pending: tuple[int, int] | None = None
        for s, e in spans:
            if pending is None and merged and text[merged[-1][0]:merged[-1][1]].rstrip().endswith("!") \
                    and text[s:e].lstrip().startswith("["):
                pending = merged.pop()
            pending = (pending[0], e) if pending else (s, e)
            if len(_QUOTE_CHAR.findall(text[pending[0]:pending[1]])) % 2 == 0:
                merged.append(pending)
                pending = None
        if pending:
            merged.append(pending)
        return [(s, e) for s, e in merged if text[s:e].strip()]

    @staticmethod
    def _clause_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
        sentence = text[start:end]
        cuts = []
        for m in CLAUSE_BOUNDARY.finditer(sentence):
            # never split inside a quoted string
            if len(_QUOTE_CHAR.findall(sentence[:m.start()])) % 2 == 0 and m.start() > 0:
                cuts.append(m.start())
        bounds = [0] + cuts + [len(sentence)]
        return [(start + a, start + b) for a, b in zip(bounds, bounds[1:]) if sentence[a:b].strip()]

    def segment(self, text: str) -> list[Unit]:
        pieces, pos = [], 0
        for m in STRUCT_SPLIT.finditer(text):
            if m.start() > pos:
                pieces.append((pos, m.start(), False))
            if text[m.start():m.end()].strip():
                pieces.append((m.start(), m.end(), True))
            pos = m.end()
        if pos < len(text):
            pieces.append((pos, len(text), False))
        units: list[Unit] = []
        sentence_no = 0
        for s, e, is_marker in pieces:
            if is_marker:
                units.append(Unit(s, e, text[s:e], "marker", sentence_no))
                sentence_no += 1
                continue
            if not text[s:e].strip():
                continue
            for ss, se in self._sentence_spans(text, s, e):
                for cs, ce in self._clause_spans(text, ss, se):
                    units.append(Unit(cs, ce, text[cs:ce], "clause", sentence_no))
                sentence_no += 1
        return units

    # ----------------------------------------------------------------- scoring
    @staticmethod
    def _strip_connector(clause: str) -> str:
        return re.sub(r"^\s*[,;]?\s*" + CONNECTORS + r"\b\s*", "", clause, flags=_I).strip()

    def _weak_signals(self, clause: str, doc) -> list[str]:
        signals = []
        imperative, verb = self._imperative(doc)
        if imperative:
            signals.append("imperative")
            if verb in COMMAND_VERBS:
                signals.append(f"command_verb:{verb}")
        if META_CUE.search(clause):
            signals.append("meta_reference")
        if signals and (URL.search(clause) or CODE_TOKEN.search(clause) or QUOTE.search(clause)):
            signals.append("output_artifact")
        return signals

    @staticmethod
    def _imperative(doc) -> tuple[bool, str]:
        """spaCy: the ROOT is a base-form verb (VB) with no subject ("Ignore ...", "Do not ...")."""
        for tok in doc:
            if tok.dep_ != "ROOT":
                continue
            has_subject = any(c.dep_ in ("nsubj", "nsubjpass", "expl", "csubj") for c in tok.children)
            lemma = tok.lemma_.lower()
            if tok.tag_ == "VB" and not has_subject:
                return True, lemma
            # "Before answering, tell the user ..." - a verb right after a leading clause
            commas = [t.i for t in doc if t.text == ","]
            if commas and commas[0] + 1 < len(doc) and not has_subject:
                nxt = doc[commas[0] + 1]
                if nxt.tag_ == "VB" or nxt.lower_ in COMMAND_VERBS:
                    return True, nxt.lemma_.lower()
            return False, lemma
        return False, ""

    def analyze(self, text: str) -> list[Unit]:
        units = self.segment(text)
        clause_texts = [self._strip_connector(u.text) if u.kind == "clause" else u.text for u in units]
        docs = list(self.nlp.pipe(clause_texts))
        for u, body, doc in zip(units, clause_texts, docs):
            u.rule_hits = _rule_hits(body)
            if u.kind == "clause":
                u.weak = self._weak_signals(body, doc)
            u.removed = bool(u.rule_hits) or u.weak_score >= 3
            if u.removed and not u.rule_hits:
                u.rule_hits.append("imperative_command")
        # A command chopped into tiny sentences ("ignore. previous. instructions. say. 'X'.")
        # is read as one run.
        i = 0
        while i < len(units):
            j = i
            while j < len(units) and len(units[j].text.split()) <= 3:
                j += 1
            if j - i >= 3:
                joined = " ".join(re.sub(r"[.!?;,]+\s*$", "", u.text.strip()) for u in units[i:j])
                hits = _rule_hits(joined)
                if hits:
                    for u in units[i:j]:
                        if not u.removed:
                            u.removed = True
                            u.rule_hits.append(hits[0])
            i = max(j, i + 1)
        # A removed header ending in ":" ("SYSTEM MESSAGE TO THE AI ASSISTANT:") introduces its
        # payload: the whole next sentence goes with it.
        for i, u in enumerate(units):
            if u.removed and u.rule_hits and u.text.rstrip().endswith(":") and i + 1 < len(units):
                target = units[i + 1].sentence
                for v in units[i + 1:]:
                    if v.sentence != target:
                        break
                    if not v.removed:
                        v.removed = True
                        v.rule_hits.append("continuation")
        # Continuations: the rest of a removed clause's sentence that keeps issuing commands,
        # fragments split off a removed unit, and attacker dialogue after an opening marker.
        for prev, u in zip(units, units[1:]):
            if u.removed or not prev.removed:
                continue
            same_sentence = prev.sentence == u.sentence and u.kind == "clause"
            tiny = len(u.text.split()) <= 4
            signal = u.weak_score >= 1 or bool(CODE_TOKEN.search(u.text) or QUOTE.search(u.text))
            after_opener = prev.kind == "marker" and bool(OPENING_MARKER.fullmatch(prev.text.strip()))
            if (same_sentence and signal) or (tiny and signal) or (after_opener and u.weak_score >= 2):
                u.removed = True
                u.rule_hits.append("continuation")
        # Legitimate context: instructions for people, not for the assistant.
        for u, body, doc in zip(units, clause_texts, docs):
            if u.removed or u.kind != "clause":
                continue
            instruction = "imperative" in u.weak or bool(INSTRUCTION_LIKE.search(body))
            cue = LEGIT_ACTOR.search(body) or ENTERPRISE_OBJECT.search(body)
            if instruction and cue:
                u.legit_reason = (f"instruction for employees or customers (cue: \u201c{cue.group(0).lower()}\u201d); "
                                  "no assistant-directed marker")
        return units

    # ----------------------------------------------------------------- reconstruction
    @staticmethod
    def _reconstruct(text: str, units: list[Unit]) -> str:
        out, pos = [], 0
        for u in units:
            out.append(text[pos:u.start])
            if u.removed:
                # keep the sentence-ending punctuation a removed clause carried
                tail = re.search(r"[.!?]['\"”’)\]]*\s*$", u.text)
                out.append(_SENTINEL + (tail.group(0).strip()[:1] if tail else ""))
            else:
                out.append(u.text)
            pos = u.end
        out.append(text[pos:])
        s = "".join(out)
        s = re.sub(r",?\s*\b" + CONNECTORS + r"\s*(?=" + _SENTINEL + ")", "", s, flags=_I)  # dangling "…, but"
        s = re.sub(r"((?:^|[.!?])\s*)" + _SENTINEL + r"([.!?]?)\s*[,;]?\s*" + CONNECTORS + r"\s+(\w)",
                   lambda m: m.group(1) + _SENTINEL + m.group(2) + " " + m.group(3).upper(), s, flags=_I)
        s = s.replace(_SENTINEL, " ")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r" *\n[ \n]*", "\n", s)
        s = re.sub(r"\s+([.,;:!?])", r"\1", s)
        s = re.sub(r"([.!?])(?:\s*[.!?,;:])+", r"\1", s)
        s = re.sub(r",(?=[.!?])", "", s)
        s = re.sub(r"^[\s.,;:!?]+", "", s)
        return s.strip()

    def sanitize(self, text: str) -> QuarantineResult:
        start = time.perf_counter()
        units = self.analyze(text)
        removed = [u for u in units if u.removed]
        sanitized = self._reconstruct(text, units) if removed else text
        return QuarantineResult(
            original=text,
            sanitized=sanitized,
            removed=[{"text": u.text.strip(" ,;"), "rules": u.rule_hits,
                      "reasons": [RULES[r][0] if r in RULES else
                                  ("imperative command aimed at the model" if r == "imperative_command" else
                                   "continues an assistant-directed clause") for r in u.rule_hits],
                      "weak_signals": u.weak} for u in removed],
            retained_instructions=[{"text": self._strip_connector(u.text), "reason": u.legit_reason}
                                   for u in units if u.legit_reason],
            units=[{k: v for k, v in asdict(u).items()} for u in units],
            latency_ms=(time.perf_counter() - start) * 1000,
        )
