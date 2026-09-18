"""S2-R4 source tracing: turn a generated value into a named, checkable support level.

Modality is never pooled. A value whose only citation is an image note is
UNKNOWN_IMAGE_SUPPORT - a mechanical string check cannot make a negative claim about
pixels. A citation label that does not resolve is INVALID_LABEL and counts against the
claim; it is never filtered out before the claim is described as fully cited.
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote

SUPPORT_LEVELS = (
    'EXTERNAL_EVIDENCE_TEXT_TRACEABLE',
    'QUERY_VISIBLE_TRACEABLE',
    'IMAGE_ONLY_UNKNOWN_IMAGE_SUPPORT',
    'INVALID_LABEL',
    'NO_SOURCE',
    'MALFORMED_REF',
)

TRANSFORM_NAMES = ('NONE', 'URL_DECODE', 'UNDERSCORE_TO_SPACE', 'EXPLICIT_YEAR_EXTRACTION')
_YEAR = re.compile(r'(?:19|20)\d{2}')


def apply_transform(text: str, transform: str) -> str:
    if transform == 'URL_DECODE':
        return unquote(text)
    if transform == 'UNDERSCORE_TO_SPACE':
        return text.replace('_', ' ')
    if transform == 'EXPLICIT_YEAR_EXTRACTION':
        match = _YEAR.search(text)
        return match.group(0) if match else text
    return text


def verify_evidence_refs(refs: list[dict], label_map: dict[str, str],
                         evidence_text: dict[str, str]) -> dict:
    """Check each evidence citation for label validity and verbatim substring support."""
    valid, invalid, unquoted = [], [], []
    for ref in refs:
        label = str(ref.get('id', ''))
        quote = str(ref.get('quote', ''))
        if label not in label_map:
            invalid.append({'id': label, 'quote': quote})
            continue
        asset_id = label_map[label]
        body = evidence_text.get(asset_id)
        if body is None:
            # Label resolves but the cited asset is an image: no text to match against.
            unquoted.append({'id': label, 'asset_id': asset_id, 'quote': quote,
                             'modality': 'image'})
            continue
        if not quote:
            invalid.append({'id': label, 'quote': quote, 'reason': 'empty_quote'})
        elif quote in body:
            valid.append({'id': label, 'quote': quote, 'asset_id': asset_id})
        else:
            invalid.append({'id': label, 'quote': quote, 'reason': 'quote_not_verbatim'})
    return {'traceable': valid, 'invalid': invalid, 'image_cited': unquoted}


def verify_query_refs(refs: list[dict], query_cells: list[dict]) -> dict:
    """A query-visible value must be an exact substring of a visible cell after its transform."""
    cells = {int(c['column_id']): c['text'] for c in query_cells}
    valid, invalid = [], []
    for ref in refs:
        try:
            column_id = int(ref.get('column_id'))
        except (TypeError, ValueError):
            invalid.append({'ref': ref, 'reason': 'bad_column_id'})
            continue
        transform = str(ref.get('transform', 'NONE')).upper()
        quote = str(ref.get('quote', ''))
        if column_id not in cells:
            invalid.append({'ref': ref, 'reason': 'column_not_visible'})
            continue
        if transform not in TRANSFORM_NAMES:
            invalid.append({'ref': ref, 'reason': f'unknown_transform:{transform}'})
            continue
        haystack = apply_transform(cells[column_id], transform)
        if quote and quote in haystack:
            valid.append({'column_id': column_id, 'quote': quote, 'transform': transform})
        else:
            invalid.append({'ref': ref, 'reason': 'quote_not_visible_after_transform',
                            'transformed_cell': haystack[:200]})
    return {'traceable': valid, 'invalid': invalid}


def classify_value(value: dict, label_map: dict[str, str], evidence_text: dict[str, str],
                   query_cells: list[dict]) -> dict:
    """Assign a single named support level to one reported value."""
    evidence = verify_evidence_refs(value.get('evidence_refs') or [], label_map, evidence_text)
    query = verify_query_refs(value.get('query_refs') or [], query_cells)
    images = value.get('image_refs') or []
    declared = value.get('source_kind')

    if evidence['invalid'] or query['invalid']:
        level = 'INVALID_LABEL' if evidence['invalid'] else 'MALFORMED_REF'
    elif evidence['traceable']:
        level = 'EXTERNAL_EVIDENCE_TEXT_TRACEABLE'
    elif query['traceable']:
        level = 'QUERY_VISIBLE_TRACEABLE'
    elif images or evidence['image_cited']:
        level = 'IMAGE_ONLY_UNKNOWN_IMAGE_SUPPORT'
    else:
        level = 'NO_SOURCE'
    return {
        'value': value.get('value'),
        'declared_source_kind': declared,
        'support_level': level,
        'declared_matches_checked': (
            (declared == 'EXTERNAL_EVIDENCE' and level == 'EXTERNAL_EVIDENCE_TEXT_TRACEABLE')
            or (declared == 'QUERY_VISIBLE' and level == 'QUERY_VISIBLE_TRACEABLE')
        ),
        'evidence_traceable': len(evidence['traceable']),
        'evidence_invalid': evidence['invalid'],
        'query_traceable': len(query['traceable']),
        'query_invalid': query['invalid'],
        'image_refs': len(images),
        'qualifier': value.get('qualifier') or {},
    }


def summarize(values: list[dict]) -> dict:
    levels = {name: 0 for name in SUPPORT_LEVELS}
    for value in values:
        levels[value['support_level']] = levels.get(value['support_level'], 0) + 1
    return {
        'values': len(values),
        'levels': levels,
        'external_evidence_supported': levels.get('EXTERNAL_EVIDENCE_TEXT_TRACEABLE', 0),
        'query_visible_supported': levels.get('QUERY_VISIBLE_TRACEABLE', 0),
        'image_only_unknown': levels.get('IMAGE_ONLY_UNKNOWN_IMAGE_SUPPORT', 0),
        'invalid_or_unsourced': (levels.get('INVALID_LABEL', 0) + levels.get('NO_SOURCE', 0)
                                 + levels.get('MALFORMED_REF', 0)),
        'note': 'image-only support is unknown, not unsupported; it is never counted as a '
                'text-grounding failure and never counted as verified external evidence',
    }
