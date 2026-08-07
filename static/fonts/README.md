# Fonts

`style.css` declares `--font: "Outfit", system-ui, ...` and carries four `@font-face`
rules pointing into this directory. Until the files below exist, the browser requests
them, gets a 404, and falls through to `system-ui`. The page stays readable the whole
time; the only symptom is that it does not look like Outfit.

## Files expected, named exactly this

| Filename | font-weight |
|---|---|
| `Outfit-Regular.woff2` | 400 |
| `Outfit-Medium.woff2` | 500 |
| `Outfit-SemiBold.woff2` | 600 |
| `Outfit-Bold.woff2` | 700 |

Drop them in and hard-reload. No CSS change is needed, and nothing has to be restarted:
FastAPI serves this directory with `StaticFiles`, so a new file is picked up immediately.

Only `.woff2` is declared. It is supported by every browser that can run this dashboard,
and adding `.woff` or `.ttf` fallbacks would double the directory for no reach.

## Where to get them

Outfit is under the SIL Open Font License. Download the static weights from Google Fonts
(`fonts.google.com/specimen/Outfit`) or the upstream repository, and convert the `.ttf`
files to `.woff2` if the download does not already include them.

**Do not add a `<link>` to a font CDN instead.** The scoring models run offline from
local ONNX files and the pipeline makes no Hugging Face calls; fetching a font from a
third-party host at page load would break that for a cosmetic gain.

## After the files land

The stylesheet uses weights 550, 650 and 680 in a few places. Those synthesize from the
nearest declared weight, which is fine for a geometric sans but will not be pixel-identical
to a real 550. Worth a look at the header and the KPI tiles once the font is actually
rendering, and worth snapping those three to 500/600/700 if the synthesis looks heavy.
