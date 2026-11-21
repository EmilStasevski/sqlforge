"""Template-based DE/EN question rendering from a Spec.

Returns the question plus the strings a paraphrase must preserve:
  literals: every value/number that appears in the SQL (as written in the question)
  terms:    glossary terms of the columns mentioned
German templates use relative clauses ("deren ... ist") so they stay grammatical regardless of noun gender.
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field

from sqlforge.sampler.spec import Filter, Sel, Spec, fmt_num
from sqlforge.schemas.card import Card, Column

MONTHS = {
    "en": "January February March April May June July August September October November December".split(),
    "de": "Januar Februar März April Mai Juni Juli August September Oktober November Dezember".split(),
}
AGG_EN = {"AVG": "average", "SUM": "total", "MIN": "minimum", "MAX": "maximum"}
AGG_DE_NOM = {"AVG": "der Durchschnitt", "SUM": "die Summe", "MIN": "das Minimum", "MAX": "das Maximum"}
AGG_DE_ACC = {"AVG": "den Durchschnitt", "SUM": "die Summe", "MIN": "das Minimum", "MAX": "das Maximum"}


@dataclass
class Rendered:
    question: str
    literals: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)


class _Ctx:
    def __init__(self, card: Card, lang: str, rng: random.Random):
        self.card, self.lang, self.rng = card, lang, rng
        self.literals: list[str] = []
        self.terms: list[str] = []

    def term(self, c: Column) -> str:
        t = c.en if self.lang == "en" else c.de
        if t is None:
            raise ValueError(f"no {self.lang} term for {c.table}.{c.name}")
        self.terms.append(t)
        return t

    def tab(self, table: str, plural: bool = True) -> str:
        t = self.card.tables[table]
        return (t.en if self.lang == "en" else t.de)[1 if plural else 0]

    def lit(self, v) -> str:
        if isinstance(v, dt.datetime):
            s = f"{v:%Y-%m-%d}"
        elif isinstance(v, dt.date):
            s = f"{v:%Y-%m-%d}"
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            s = fmt_num(v)
        else:
            s = str(v)
        self.literals.append(s)
        return s

    def num(self, n: int) -> str:
        s = str(n)
        self.literals.append(s)
        return s

    def pick(self, *options: str) -> str:
        return self.rng.choice(options)

    # ------------------------------------------------------------ filters
    def filt(self, f: Filter, base: str) -> str:
        en = self.lang == "en"
        if f.op == "BOOL":
            t = self.term(f.col)
            if f.col.table != base:
                owner = self.tab(f.col.table, plural=False)
                if en:
                    return f"whose {owner} is {'' if f.values[0] else 'not '}{t}"
                return f"deren {owner} {'' if f.values[0] else 'nicht '}{t} ist"
            return (
                f"that are {'' if f.values[0] else 'not '}{t}"
                if en
                else f"die {'' if f.values[0] else 'nicht '}{t} sind"
            )
        t = self.term(f.col)
        v = f.values
        if f.op == "=":
            return f"whose {t} is {self.lit(v[0])}" if en else f"deren {t} {self.lit(v[0])} ist"
        if f.op == "IN":
            a, b = self.lit(v[0]), self.lit(v[1])
            return f"whose {t} is {a} or {b}" if en else f"deren {t} {a} oder {b} ist"
        if f.op in (">", "<") and f.col.role == "num":
            if en:
                return f"whose {t} is {'greater' if f.op == '>' else 'less'} than {self.lit(v[0])}"
            return f"deren {t} {'größer' if f.op == '>' else 'kleiner'} als {self.lit(v[0])} ist"
        if f.op == "BETWEEN":
            a, b = self.lit(v[0]), self.lit(v[1])
            return f"whose {t} is between {a} and {b}" if en else f"deren {t} zwischen {a} und {b} liegt"
        if f.op == "YEAR":
            y = self.lit(v[0])
            return f"whose {t} is in {y}" if en else f"deren {t} im Jahr {y} liegt"
        if f.op == "MONTH":
            m = MONTHS[self.lang][v[0].month - 1]
            self.literals.append(m)
            y = self.lit(v[0].year)
            return f"whose {t} is in {m} {y}" if en else f"deren {t} im {m} {y} liegt"
        if f.op == "DAY":
            d = self.lit(v[0])
            return f"whose {t} is on {d}" if en else f"deren {t} auf den {d} fällt"
        if f.op == ">=":
            d = self.lit(v[0])
            return f"whose {t} is on or after {d}" if en else f"deren {t} am oder nach dem {d} liegt"
        if f.op == "<":
            d = self.lit(v[0])
            return f"whose {t} is before {d}" if en else f"deren {t} vor dem {d} liegt"
        raise ValueError(f"no phrase for {f.op}")

    def filters(self, fs: list[Filter], base: str, mode: str = "end") -> str:
        """EN: ' whose ... and whose ...'. DE end: ', deren ...'; DE mid: ', deren ...,'"""
        if not fs:
            return ""
        joined = (" and " if self.lang == "en" else " und ").join(self.filt(f, base) for f in fs)
        if self.lang == "en":
            return " " + joined
        return ", " + joined + ("," if mode == "mid" else "")

    def agg(self, s: Sel, case: str = "nom") -> str:
        if self.lang == "en":
            return AGG_EN[s.func]
        return (AGG_DE_NOM if case == "nom" else AGG_DE_ACC)[s.func]

    def round_phrase(self, s: Sel) -> str:
        if s.round is None:
            return ""
        n = self.num(s.round)
        if self.lang == "en":
            return f" Round the result to {n} decimal place{'s' if s.round != 1 else ''}."
        return f" Runde das Ergebnis auf {n} Nachkommastelle{'n' if s.round != 1 else ''}."

    def cols(self, cs: list[Column]) -> str:
        names = [self.term(c) for c in cs]
        return (" and " if self.lang == "en" else " und ").join(names)


def render_question(spec: Spec, card: Card, lang: str, rng: random.Random) -> Rendered:
    x = _Ctx(card, lang, rng)
    en = lang == "en"
    T = x.tab(spec.base)
    k = spec.kind
    sel = spec.select

    if k == "count_all":
        q = (
            x.pick(f"How many {T} are there?", f"Count all {T}.", f"What is the number of {T}?")
            if en
            else x.pick(f"Wie viele {T} gibt es?", f"Zähle alle {T}.", f"Wie groß ist die Anzahl der {T}?")
        )

    elif k == "count_where":
        F = x.filters(spec.filters, spec.base)
        q = (
            x.pick(f"How many {T} are there{F}?", f"Count the {T}{F}.", f"What is the number of {T}{F}?")
            if en
            else x.pick(
                f"Wie viele {T} gibt es{F}?", f"Zähle die {T}{F}.", f"Wie groß ist die Anzahl der {T}{F}?"
            )
        )

    elif k == "agg":
        s = sel[0]
        C = x.term(s.col)
        F = x.filters(spec.filters, spec.base)
        R = x.round_phrase(s)
        if en:
            A = x.agg(s)
            q = (
                x.pick(
                    f"What is the {A} {C} of {T}{F}?",
                    f"Compute the {A} {C} across all {T}{F}.",
                    f"Find the {A} {C} for {T}{F}.",
                )
                + R
            )
        else:
            q = (
                x.pick(
                    f"Was ist {x.agg(s)} von {C} über alle {T}{F}?",
                    f"Berechne {x.agg(s, 'acc')} von {C} für alle {T}{F}.",
                )
                + R
            )

    elif k == "list_where":
        Cs = x.cols([s.col for s in sel])
        F = x.filters(spec.filters, spec.base)
        if spec.limit is None:
            q = (
                x.pick(f"Show the {Cs} of all {T}{F}.", f"List {Cs} for {T}{F}.")
                if en
                else x.pick(f"Zeige {Cs} aller {T}{F}.", f"Liste {Cs} der {T} auf{F}.")
            )
        else:
            i, d = spec.order_by
            Co = x.term(sel[i].col)
            K = x.num(spec.limit)
            if en:
                q = (
                    f"Show the {Cs} of the first {K} {T}{F}, sorted by {Co} in "
                    f"{'descending' if d == 'DESC' else 'ascending'} order."
                )
            else:
                q = (
                    f"Zeige {Cs} der ersten {K} {T}{F}, sortiert nach {Co} "
                    f"{'absteigend' if d == 'DESC' else 'aufsteigend'}."
                )

    elif k == "topk":
        Cl, Cn = x.term(sel[0].col), x.term(sel[1].col)
        F = x.filters(spec.filters, spec.base)
        hi = spec.order_by[1] == "DESC"
        K = x.num(spec.limit) if (spec.limit != 1 or not en) else "1"  # EN top-1 template has no number
        if en:
            hl = "highest" if hi else "lowest"
            if spec.limit == 1:
                q = f"Which {x.tab(spec.base, False)}{F} has the {hl} {Cn}? Show its {Cl} and {Cn}."
            else:
                q = x.pick(
                    f"Among {T}{F}, which {K} have the {hl} {Cn}? Show {Cl} and {Cn}.",
                    f"Show the {Cl} and {Cn} of the top {K} {T} by {hl} {Cn}{F}.",
                )
        else:
            hl = "höchsten" if hi else "niedrigsten"
            q = f"Zeige {Cl} und {Cn} der Top-{K} {T} mit dem {hl} Wert für {Cn}{F}."

    elif k == "group_agg":
        G, s = x.term(sel[0].col), sel[1]
        C = x.term(s.col)
        F = x.filters(spec.filters, spec.base)
        R = x.round_phrase(s)
        if spec.limit is None:
            if en:
                A = x.agg(s)
                q = x.pick(
                    f"What is the {A} {C} of {T}{F}, per {G}?",
                    f"For each {G}, compute the {A} {C} of {T}{F}.",
                )
            else:
                q = x.pick(
                    f"Was ist {x.agg(s)} von {C} pro {G} für {T}{F}?",
                    f"Berechne {x.agg(s, 'acc')} von {C} je {G} für alle {T}{F}.",
                )
        else:
            K = x.num(spec.limit)
            if en:
                A = x.agg(s)
                q = f"Which {K} {G} values have the highest {A} {C} of {T}{F}? Show the {G} and the {A} {C}."
            else:
                q = (
                    f"Zeige {G} und {x.agg(s, 'acc')} von {C} für {T}{F.rstrip(',')}, aber nur für die "
                    f"{K} Werte von {G} mit dem höchsten Ergebnis."
                )
        q += R

    elif k == "group_count":
        G = x.term(sel[0].col)
        if spec.limit is None:
            if en:
                F = x.filters(spec.filters, spec.base)
                q = x.pick(f"For each {G}, how many {T}{F} are there?", f"Count the {T}{F} per {G}.")
            else:
                F = x.filters(spec.filters, spec.base, "mid")
                q = f"Wie viele {T}{F} gibt es pro {G}?"
        else:
            K = x.num(spec.limit)
            F = x.filters(spec.filters, spec.base)
            q = (
                f"Which {K} {G} values have the most {T}{F}? Show the {G} and the count."
                if en
                else f"Welche {K} Werte von {G} haben die meisten {T}{F}? Zeige {G} und die Anzahl."
            )

    elif k == "having":
        G = x.term(sel[0].col)
        op, v = spec.having
        V = x.num(v)
        F = x.filters(spec.filters, spec.base)
        if en:
            more = "more" if op == ">" else "fewer"
            q = x.pick(
                f"Which {G} values have {more} than {V} {T}{F}?",
                f"List every {G} with {more} than {V} {T}{F}.",
            )
        else:
            more = "mehr" if op == ">" else "weniger"
            q = x.pick(
                f"Welche Werte von {G} haben {more} als {V} {T}{F}?",
                f"Liste jeden Wert von {G} mit {more} als {V} {T}{F} auf.",
            )

    elif k == "count_distinct":
        C = x.term(sel[0].col)
        F = x.filters(spec.filters, spec.base)
        q = (
            x.pick(
                f"How many different {C} values are there among {T}{F}?",
                f"Count the distinct {C} values of {T}{F}.",
            )
            if en
            else x.pick(
                f"Wie viele verschiedene Werte von {C} gibt es bei den {T}{F}?",
                f"Zähle die unterschiedlichen Werte von {C} der {T}{F}.",
            )
        )

    elif k == "above_avg":
        Cl, Cn = x.term(sel[0].col), x.term(sel[1].col)
        F = x.filters(spec.filters, spec.base)
        if en:
            cond = f"whose {Cn} is above the average {Cn} of all {T}"
            cond = f"{F} and {cond}" if F else f" {cond}"
            if spec.limit is None:
                q = f"Show the {Cl} and {Cn} of {T}{cond}."
            else:
                K = x.num(spec.limit)
                q = f"Among {T}{cond}, show the {Cl} and {Cn} of the {K} with the highest {Cn}."
        else:
            cond = f"deren {Cn} über dem Durchschnitt von {Cn} aller {T} liegt"
            cond = f"{F} und {cond}" if F else f", {cond}"
            if spec.limit is None:
                q = f"Zeige {Cl} und {Cn} der {T}{cond}."
            else:
                K = x.num(spec.limit)
                q = f"Zeige {Cl} und {Cn} der Top-{K} {T} mit dem höchsten Wert für {Cn}{cond}."

    elif k == "time_group":
        D = x.term(sel[0].col)
        unit = ("year" if en else "Jahr") if sel[0].func == "YEAR" else ("month" if en else "Monat")
        m = sel[1]
        if m.func == "COUNT":
            if en:
                F = x.filters(spec.filters, spec.base)
                q = f"How many {T}{F} are there per {unit} of {D}? Sort chronologically."
            else:
                F = x.filters(spec.filters, spec.base, "mid")
                q = f"Wie viele {T}{F} gibt es pro {unit} ({D})? Chronologisch sortiert."
        else:
            C = x.term(m.col)
            if en:
                F = x.filters(spec.filters, spec.base)
                q = f"What is the {x.agg(m)} {C} of {T}{F} per {unit} of {D}, in chronological order?"
            else:
                F = x.filters(spec.filters, spec.base, "mid")
                q = f"Was ist {x.agg(m)} von {C} der {T}{F} pro {unit} ({D}), chronologisch sortiert?"
            q += x.round_phrase(m)

    elif k == "in_subquery":
        Cl = x.term(sel[0].col)
        sub = spec.sub
        child = sub.child.table
        F2 = x.filters(list(sub.child_filters), child)
        T2 = x.tab(child)
        neg = sub.kind == "not_in"
        if en:
            F = x.filters(spec.filters, spec.base)
            conj = " and" if F else ""
            T2s = x.tab(child, plural=False)
            have = f"that have no {T2}{F2}" if neg else f"that have at least one {T2s}{F2}"
            q = f"Show the {Cl} of {T}{F}{conj} {have}."
        else:
            F = x.filters(spec.filters, spec.base, "mid")
            sep = "" if F else ","
            q = f"Zeige {Cl} aller {T}{F}{sep} zu denen es {'keine' if neg else ''} {T2} gibt{F2}.".replace(
                "es  ", "es "
            )
    else:
        raise ValueError(k)

    return Rendered(q, x.literals, list(dict.fromkeys(x.terms)))
