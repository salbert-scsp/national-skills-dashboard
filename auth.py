"""
STAGE: Web App

One shared password over the review UI. Not user accounts.

The app had no authentication at all before this: its entire security model was
"bound to 127.0.0.1". That is fine on a laptop and not fine anywhere else, and every
review form mutates the store.

A signed cookie rather than a session table. The cookie carries an expiry and an HMAC
of that expiry, so it verifies with nothing on the server side -- which is the point.
There is no session machinery in this app and one boolean does not justify introducing
any. HTTP Basic was the other candidate and lost on three counts: there is no username
to supply, there is no way to log out, and Safari re-prompts for it inside the progress
iframe.

WHAT IS AND IS NOT PROTECTED
----------------------------
Public: the dashboard, its static assets, and POST /report. Everything else needs the
cookie.

Note the asymmetry and accept it deliberately: POST /report WRITES to the store -- it
flips an approved skill back to pending -- and it stays unauthenticated because
static/dashboard.js calls it by fetch() from the public dashboard. That was already
true before this module existed. It is now the only unauthenticated write path in the
app, which is worth knowing rather than discovering.

FAILS CLOSED
------------
A missing ADMIN_PASSWORD denies every protected request rather than defaulting to open.
The failure mode of the other choice is a mutable review queue on an open endpoint.
"""

import hashlib
import hmac
import logging
import os
import secrets
import time
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

# Regenerated per process when unset, which means a server restart logs everyone out.
# That is a reasonable default -- it bounds the lifetime of a stolen cookie to one
# server lifetime -- but set SESSION_SECRET in .env if sessions should outlive restarts.
SESSION_SECRET = os.getenv("SESSION_SECRET") or secrets.token_hex(32)
_SECRET_IS_EPHEMERAL = not os.getenv("SESSION_SECRET")

COOKIE_NAME = "sd_session"
MAX_AGE = 60 * 60 * 12

# Exact paths that never require a cookie, plus the two prefixes that must. Exact
# matching for the rest: a bare startswith("/dashboard") would also open a route named
# /dashboard-admin if one were ever added.
PUBLIC_EXACT = frozenset({
    "/dashboard",
    "/occupations",
    "/report",
    "/login",
    "/logout",
    "/favicon.ico",
})
PUBLIC_PREFIXES = ("/static/",)


def is_public(path: str) -> bool:
    """True if `path` may be served without a session cookie."""
    return path in PUBLIC_EXACT or path.startswith(PUBLIC_PREFIXES)


def configured() -> bool:
    return bool(ADMIN_PASSWORD)


def check_password(supplied: str) -> bool:
    """
    Verifies the supplied password in constant time.

    compare_digest rather than ==, so the password cannot be recovered one character at
    a time by timing the response. That matters more here than usual: there is no rate
    limiting and no account to lock.
    """
    if not ADMIN_PASSWORD:
        logger.critical(
            "ADMIN_PASSWORD is not set, so every protected route is denied. Set it in "
            ".env to use the review UI."
        )
        return False
    return hmac.compare_digest((supplied or "").encode("utf-8"),
                               ADMIN_PASSWORD.encode("utf-8"))


def issue() -> str:
    """Mints a session token: an expiry timestamp and an HMAC over it."""
    expiry = str(int(time.time()) + MAX_AGE)
    signature = hmac.new(
        SESSION_SECRET.encode("utf-8"), expiry.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"{expiry}.{signature}"


def valid(token: Optional[str]) -> bool:
    """
    True if `token` was issued by this process and has not expired.

    Both halves are checked: a valid signature over a past expiry is still refused, or
    a leaked cookie would work forever.
    """
    try:
        expiry, signature = (token or "").split(".", 1)
    except ValueError:
        return False

    expected = hmac.new(
        SESSION_SECRET.encode("utf-8"), expiry.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return False

    try:
        return int(expiry) > time.time()
    except ValueError:
        return False


def safe_next(target: Optional[str]) -> str:
    """
    Sanitizes a post-login redirect target.

    Must start with a single slash. Rejecting "//evil.com" specifically matters: the
    browser reads a protocol-relative URL as another host, so without this check the
    login form would be an open redirect that borrows this site's credibility.
    """
    value = (target or "").strip()
    if not value.startswith("/") or value.startswith("//"):
        return "/"
    return value


def startup_warnings(host: str, secure_scheme: bool = False) -> None:
    """
    Logs the configuration mistakes that matter, at startup rather than on discovery.

    Called from the app lifespan. Three things are worth a line here: no password set
    at all, a password sent in the clear to a non-loopback interface, and an ephemeral
    session secret in a deployment where restarts are routine.
    """
    if not ADMIN_PASSWORD:
        logger.critical(
            "ADMIN_PASSWORD is not set. The review UI will refuse every request until "
            "it is. Add ADMIN_PASSWORD to .env."
        )
    elif host not in ("127.0.0.1", "localhost", "::1") and not secure_scheme:
        logger.critical(
            "Serving on %s over plain HTTP with a password. The password and the "
            "session cookie both cross the network in the clear. Put this behind TLS "
            "before anyone outside this machine uses it.",
            host,
        )

    if _SECRET_IS_EPHEMERAL:
        logger.info(
            "SESSION_SECRET is unset, so a new one is generated per process and every "
            "restart signs everyone out. Set it in .env to keep sessions across "
            "restarts."
        )
