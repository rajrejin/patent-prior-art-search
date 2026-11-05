"""Parse raw OPS XML (biblio, claims, family, search) into the normalised schema."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from lxml import etree

from patentrag.models import CitationInfo, Patent, PubId

_COUNTRY_SUFFIX = re.compile(r"\s*\[[A-Z]{2}\]\s*$")
_WS = re.compile(r"\s+")
_IPC = re.compile(r"^([A-H]\d{2}[A-Z])\s*(\d+)\s*/\s*(\d+)")


def _root(xml: str) -> etree._Element:
    parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True)
    root = etree.fromstring(xml.encode("utf-8"), parser)
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _text(el: etree._Element | None) -> str:
    if el is None:
        return ""
    return _WS.sub(" ", "".join(el.itertext())).strip()


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:8], "%Y%m%d").date()
    except ValueError:
        return None


def _docdb_date(el: etree._Element) -> date | None:
    for d in el.iter("document-id"):
        if d.get("document-id-type") == "docdb":
            return _parse_date(d.findtext("date"))
    return None


def _docdb_id(el: etree._Element) -> PubId | None:
    """PubId from the docdb <document-id> inside `el` (a patcit or publication-reference)."""
    for d in el.iter("document-id"):
        if d.get("document-id-type") == "docdb":
            c, n, k = d.findtext("country"), d.findtext("doc-number"), d.findtext("kind")
            if c and n:
                return PubId(country=c, number=n, kind=k or "")
    return None


def parse_search(xml: str) -> tuple[int, list[dict[str, str]]]:
    """Return (total result count, [{country, number, kind, family_id}])."""
    root = _root(xml)
    bs = root.find(".//biblio-search")
    total = int(bs.get("total-result-count", "0")) if bs is not None else 0
    out: list[dict[str, str]] = []
    for ref in root.iter("publication-reference"):
        pid = _docdb_id(ref)
        if pid:
            out.append(
                {
                    "country": pid.country,
                    "number": pid.number,
                    "kind": pid.kind,
                    "family_id": ref.get("family-id", ""),
                }
            )
    return total, out


def _cpc_symbols(doc: etree._Element) -> list[str]:
    seen: dict[str, None] = {}
    for pc in doc.iter("patent-classification"):
        scheme = pc.find("classification-scheme")
        if scheme is not None and not (scheme.get("scheme") or "").startswith("CPC"):
            continue
        sec, cls, sub = pc.findtext("section"), pc.findtext("class"), pc.findtext("subclass")
        mg, sg = pc.findtext("main-group"), pc.findtext("subgroup")
        if sec and cls and sub and mg and sg:
            seen[f"{sec}{cls}{sub}{mg.strip()}/{sg.strip()}"] = None
    return list(seen)


def _ipc_symbols(doc: etree._Element) -> list[str]:
    seen: dict[str, None] = {}
    for t in doc.iter("classification-ipcr"):
        m = _IPC.match(_text(t.find("text")))
        if m:
            seen[f"{m.group(1)}{int(m.group(2))}/{m.group(3)}"] = None
    return list(seen)


def _parties(doc: etree._Element, tag: str, name_tag: str) -> list[str]:
    by_format: dict[str, list[str]] = {}
    for p in doc.iter(tag):
        name = _text(p.find(f"{name_tag}/name"))
        if name:
            by_format.setdefault(p.get("data-format", ""), []).append(name)
    names = by_format.get("epodoc") or by_format.get("original")
    if names is None:
        names = next(iter(by_format.values()), [])
    cleaned = [_COUNTRY_SUFFIX.sub("", n).strip() for n in names]
    return list(dict.fromkeys(n for n in cleaned if n))


def _pick_lang(items: list[tuple[str, str]]) -> tuple[str, str]:
    """(lang, text) preferring English, then the first non-empty item."""
    items = [(lang, t) for lang, t in items if t]
    for lang, t in items:
        if lang == "en":
            return lang, t
    return items[0] if items else ("", "")


def parse_biblio(xml: str) -> Patent | None:
    root = _root(xml)
    doc = root.find(".//exchange-document")
    if doc is None:
        return None
    bib = doc.find("bibliographic-data")
    if bib is None:
        return None
    pub = bib.find("publication-reference")
    pid = _docdb_id(pub) if pub is not None else None
    if pid is None:
        pid = PubId(
            country=doc.get("country", ""),
            number=doc.get("doc-number", ""),
            kind=doc.get("kind", ""),
        )
    pub_date = filing = None
    if pub is not None:
        for d in pub.iter("document-id"):
            if d.get("document-id-type") == "docdb":
                pub_date = _parse_date(d.findtext("date"))
    app = bib.find("application-reference")
    if app is not None:
        for d in app.iter("document-id"):
            if d.get("document-id-type") == "docdb":
                filing = _parse_date(d.findtext("date"))

    titles = {
        (t.get("lang") or "").lower(): _text(t) for t in bib.iter("invention-title") if _text(t)
    }
    title_lang, title = _pick_lang(list(titles.items()))
    abstracts = [((a.get("lang") or "").lower(), _text(a)) for a in doc.iter("abstract")]
    abs_lang, abstract = _pick_lang(abstracts)

    cites: list[PubId] = []
    details: list[CitationInfo] = []
    for cit in bib.iter("citation"):
        pat = cit.find("patcit")
        if pat is None:
            continue  # non-patent literature
        cid = _docdb_id(pat)
        if cid is None:
            continue
        cites.append(cid)
        details.append(
            CitationInfo(
                pub_id=cid,
                category=cit.findtext("category") or "",
                phase=cit.get("cited-phase", ""),
                cited_by=cit.get("cited-by", ""),
                pub_date=_docdb_date(pat),
            )
        )

    applicants = _parties(bib, "applicant", "applicant-name")
    cpc = _cpc_symbols(bib)
    present = [
        name
        for name, ok in (
            ("title", bool(title)),
            ("abstract", bool(abstract)),
            ("applicants", bool(applicants)),
            ("cpc", bool(cpc)),
            ("cites", bool(cites)),
        )
        if ok
    ]
    return Patent(
        pub_id=pid,
        title=title,
        titles=titles,
        abstract=abstract,
        language=abs_lang or title_lang,
        applicants=applicants,
        inventors=_parties(bib, "inventor", "inventor-name"),
        filing_date=filing,
        pub_date=pub_date,
        cpc=cpc,
        ipc=_ipc_symbols(bib),
        cites=cites,
        citation_details=details,
        family_id=doc.get("family-id") or None,
        fields_present=present,
    )


def parse_claims(xml: str) -> tuple[str, str] | None:
    """Return (language, claims text) preferring English, or None when no claims."""
    root = _root(xml)
    found: list[tuple[str, str]] = []
    for c in root.iter("claims"):
        texts = [_text(ct) for ct in c.iter("claim-text")]
        text = "\n".join(t for t in texts if t)
        if text:
            found.append(((c.get("lang") or "").lower(), text))
    lang, text = _pick_lang(found)
    return (lang, text) if text else None


def parse_family(xml: str) -> list[PubId]:
    root = _root(xml)
    out: dict[str, PubId] = {}
    for m in root.iter("family-member"):
        ref = m.find("publication-reference")
        pid = _docdb_id(ref) if ref is not None else None
        if pid:
            out[str(pid)] = pid
    return list(out.values())


def attach(patent: Patent, claims_xml: str | None, family_xml: str | None) -> Patent:
    """Return `patent` with claims and family members filled in from their raw XML."""
    data: dict[str, Any] = {}
    present = list(patent.fields_present)
    if claims_xml:
        parsed = parse_claims(claims_xml)
        if parsed:
            data["claims"] = parsed[1]
            present.append("claims")
    if family_xml:
        members = [m for m in parse_family(family_xml) if str(m) != str(patent.pub_id)]
        data["family_members"] = members
        present.append("family")
    data["fields_present"] = present
    return patent.model_copy(update=data)
