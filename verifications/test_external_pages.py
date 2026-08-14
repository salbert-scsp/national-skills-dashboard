"""
Tests for reading a reference page that is not on Wikipedia.

    python3.11 -m pytest test_external_pages.py -q

No network. Every case runs the extraction and text-selection functions over HTML
fixtures written here, so the suite cannot be broken by a vendor redesigning a page.
The live behaviour these fixtures stand in for was measured against 32 real URLs.

The defect being guarded against: the extract becomes `wikipedia_summary`, which is the
text the bi-encoder scores. Before this, results were bimodal -- a navigation stub of
61-163 characters, or the whole page against a 4000-character cap -- and never
definition-shaped. A 4000-character dump of nginx.org measured ml_pipeline_sim 0.299
against 0.114 for a real one-line definition, which crossed ML_PIPELINE_FLOOR and
classified the skill AI Enabling where Wikipedia had it Not AI.
"""

import pytest

from scraping import (
    MAX_EXTERNAL_EXTRACT_CHARS,
    MIN_EXTERNAL_EXTRACT_CHARS,
    _best_definition_text,
    _extract_page_text,
    _trim_to_sentence,
)

REAL_DEFINITION = (
    "Nginx is a web server that can also be used as a reverse proxy, load balancer, "
    "mail proxy and HTTP cache. It was originally written by Igor Sysoev and is now "
    "maintained by F5 Networks, and it is widely deployed in front of application "
    "servers to terminate TLS and serve static files."
)


def page(body="", head="", description=None):
    meta = f'<meta name="description" content="{description}">' if description else ""
    return f"""<!doctype html><html><head><title>A Product</title>{meta}{head}</head>
    <body>{body}</body></html>""".encode("utf-8")


# --- chrome removal -----------------------------------------------------------

def test_div_built_navigation_is_removed():
    """
    _BOILERPLATE_TAGS already drops <nav> and <footer>, which does nothing for the many
    commercial sites that build the same furniture out of divs. bentley.com yielded
    "All Software / CAD Modeling and Visualization / MicroSta..." with every semantic
    tag already stripped.
    """
    html = page(body="""
        <div class="site-header"><a>All Software</a><a>CAD Modeling and Visualization</a></div>
        <div class="nav-primary"><a>Products</a><a>Support</a></div>
        <div id="content"><p>MicroStation is a CAD platform for infrastructure design.</p></div>
        <div class="footer-links"><a>Cookie settings</a></div>
    """)
    _, text, _ = _extract_page_text(html)

    assert "MicroStation is a CAD platform" in text
    for chrome in ("All Software", "CAD Modeling and Visualization", "Cookie settings", "Support"):
        assert chrome not in text, f"{chrome!r} survived chrome removal"


def test_a_class_containing_a_chrome_word_is_not_removed():
    """Matched as whole words, so 'mainContent' and 'navigator' are not furniture."""
    html = page(body='<div class="mainContent"><p>' + REAL_DEFINITION + "</p></div>")
    _, text, _ = _extract_page_text(html)
    assert "reverse proxy" in text


def test_a_content_wrapper_wearing_a_state_class_survives():
    """
    A class name cannot reliably tell furniture from content. blender.org wraps its whole
    article in "type-page ps-about has-header", and any pattern catching "site-header"
    also catches "has-header" -- which took that page from 3,005 characters to 15. The
    size guard is what distinguishes them: chrome is a small part of a page.
    """
    body = (
        '<div class="site-header"><a>Home</a><a>Pricing</a></div>'
        '<div class="type-page ps-about has-header"><p>' + REAL_DEFINITION * 3 + "</p></div>"
    )
    _, text, _ = _extract_page_text(page(body=body))

    assert "reverse proxy" in text, "the content wrapper was removed as chrome"
    assert "Pricing" not in text, "real chrome survived"


def test_chrome_is_still_removed_when_the_page_is_mostly_content():
    big = "<p>" + REAL_DEFINITION * 6 + "</p>"
    body = f'<div class="navbar"><a>Home</a><a>Support</a></div><div id="content">{big}</div>'
    _, text, _ = _extract_page_text(page(body=body))
    assert "Support" not in text and "reverse proxy" in text


def test_aria_roles_are_removed():
    html = page(body="""
        <div role="navigation"><a>Home</a><a>Pricing</a></div>
        <div role="banner">Buy now</div>
        <div id="main"><p>Redis is an in-memory data store.</p></div>
    """)
    _, text, _ = _extract_page_text(html)
    assert "Redis is an in-memory data store." in text
    assert "Pricing" not in text and "Buy now" not in text


# --- the description is preferred --------------------------------------------

def test_the_meta_description_wins_over_body_prose():
    """
    og:description and <meta name="description"> are hand-written one-paragraph
    summaries of exactly the shape wanted, and are server-rendered even on sites whose
    body is built by JavaScript. Preferring them is what turns a vendor page from a menu
    dump into a definition.
    """
    html = page(
        description=REAL_DEFINITION,
        body="<div id='content'><p>" + ("Marketing filler. " * 60) + "</p></div>",
    )
    _, text, description = _extract_page_text(html)
    chosen = _best_definition_text(text, description)

    assert chosen.startswith("Nginx is a web server")
    assert "Marketing filler" not in chosen


def test_og_description_is_preferred_over_the_name_variant():
    html = page(
        description="Short and unhelpful tagline that is nonetheless long enough to pass.",
        head='<meta property="og:description" content="' + REAL_DEFINITION + '">',
    )
    _, _, description = _extract_page_text(html)
    assert description.startswith("Nginx is a web server")


def test_a_description_shorter_than_the_floor_does_not_win():
    """A 40-character tagline is no better than the stub it would replace."""
    html = page(
        description="Fast. Reliable.",
        body="<div id='content'><p>" + REAL_DEFINITION + "</p></div>",
    )
    _, text, description = _extract_page_text(html)
    chosen = _best_definition_text(text, description)
    assert chosen.startswith("Nginx is a web server")


def test_the_description_is_read_even_when_the_body_is_empty():
    """The JavaScript-rendered case: nothing in the markup, a real summary in the head."""
    html = page(description=REAL_DEFINITION, body="<div id='root'></div>")
    _, text, description = _extract_page_text(html)
    assert len(text.strip()) < 40
    assert _best_definition_text(text, description).startswith("Nginx is a web server")


# --- length bounds ------------------------------------------------------------

def test_the_result_is_definition_shaped_not_a_page_dump():
    """
    Stored Wikipedia definitions run a median of 191 characters. The old cap was 4000,
    and body text hit it, which is what changed the classification.
    """
    html = page(body="<div id='content'><p>" + ("Sentence about the product. " * 400) + "</p></div>")
    _, text, description = _extract_page_text(html)
    chosen = _best_definition_text(text, description)
    assert len(chosen) <= MAX_EXTERNAL_EXTRACT_CHARS
    assert MAX_EXTERNAL_EXTRACT_CHARS <= 800, "the cap must stay definition-shaped"


def test_a_navigation_stub_falls_under_the_floor():
    """
    kafka.apache.org returned the 12-character "Documentation Redirect" and it was
    stored as a definition with score 0.996.
    """
    html = page(body="<div id='content'><p>Documentation Redirect</p></div>")
    _, text, description = _extract_page_text(html)
    chosen = _best_definition_text(text, description)
    assert len(chosen) < MIN_EXTERNAL_EXTRACT_CHARS


def test_text_is_cut_at_a_sentence_boundary():
    long_text = "First sentence here. Second sentence here. " + ("padding word " * 200)
    trimmed = _trim_to_sentence(long_text, 60)
    assert trimmed.endswith(".")
    assert len(trimmed) <= 60


def test_a_hard_cut_is_used_when_there_is_no_sentence_end():
    """Better than returning something far shorter than asked for."""
    trimmed = _trim_to_sentence("word " * 200, 100)
    assert 90 <= len(trimmed) <= 100


def test_trimming_collapses_whitespace():
    assert _trim_to_sentence("a  \n\n  b\tc", 100) == "a b c"


def test_short_text_is_returned_whole():
    assert _trim_to_sentence("Exactly this.", 100) == "Exactly this."


# --- the regression this is really about --------------------------------------

@pytest.mark.real_model
def test_a_page_dump_no_longer_outclassifies_a_real_definition():
    """
    The measured defect: 4000 characters of nginx.org navigation scored ml_pipeline_sim
    0.299 against 0.114 for a one-line definition, crossing ML_PIPELINE_FLOOR and
    classifying the skill AI Enabling where Wikipedia had it Not AI. A vendor-sourced
    skill must not systematically outrank the same skill sourced from an encyclopedia.
    """
    from sortingalgorithmnew import calculate_ai_correlation

    chrome = (
        "Basic HTTP server features Other HTTP server features Mail proxy server "
        "features TCP UDP proxy server features Architecture and scalability Download "
        "Documentation Security advisories Books Support Blog Twitter "
    ) * 12
    html = page(description=REAL_DEFINITION, body=f"<div id='content'><p>{chrome}</p></div>")

    _, text, description = _extract_page_text(html)
    chosen = _best_definition_text(text, description)

    from_page = calculate_ai_correlation("Nginx", "software", chosen)
    from_wiki = calculate_ai_correlation("Nginx", "software", REAL_DEFINITION)

    assert from_page["category_bucket"] == from_wiki["category_bucket"]
    assert from_page["ml_pipeline_sim"] == pytest.approx(
        from_wiki["ml_pipeline_sim"], abs=0.05
    )


# --- configuration ------------------------------------------------------------

def test_markdown_is_an_accepted_content_type():
    """docs.docker.com serves text/markdown and was refused outright as not_html."""
    from scraping import EXTERNAL_CONTENT_TYPES

    assert "text/markdown".startswith(EXTERNAL_CONTENT_TYPES)
    assert "application/pdf".startswith(EXTERNAL_CONTENT_TYPES) is False


def test_every_error_code_has_a_reviewer_sentence():
    from main import REMEDIATION_ERRORS
    from scraping import REMEDIATION_ERROR_CODES

    for code in REMEDIATION_ERROR_CODES:
        assert REMEDIATION_ERRORS.get(code), f"{code} has no reviewer-facing sentence"


def test_too_thin_is_a_declared_code():
    from scraping import REMEDIATION_ERROR_CODES

    assert "too_thin" in REMEDIATION_ERROR_CODES
