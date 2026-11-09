"""Platform shell: global styles, header and page navigation.

Both pages call :func:`page_shell` first. It pins the document to the viewport
(the pages scroll inside their own columns), renders the platform header and
marks the active page, so the two pages cannot drift into different chrome.
"""

from __future__ import annotations

from nicegui import ui

HEADER_HEIGHT = "3.5rem"

PAGES = (
    ("aef", "AEF 无监督变化检测", "/"),
    ("dw", "AEF 变化检测与 DW 标签对比", "/dw"),
)

_GLOBAL_CSS = """
.nicegui-content{padding:0;align-items:stretch;justify-content:flex-start;}
html,body{height:100%;overflow:hidden;}
body{background:#f5f6f8;}
/* Thin, always-visible scrollbar so each column's own scroll position is
   distinguishable from the (disabled) page scroll. */
.panel-scroll{scrollbar-width:thin;scrollbar-color:#b8bdc7 #eceef2;}
.panel-scroll::-webkit-scrollbar{width:10px;}
.panel-scroll::-webkit-scrollbar-track{background:#eceef2;border-radius:5px;}
.panel-scroll::-webkit-scrollbar-thumb{background:#b8bdc7;border-radius:5px;
border:2px solid #eceef2;}
.panel-scroll::-webkit-scrollbar-thumb:hover{background:#97a0ad;}
.page-nav{color:#4b5563;text-decoration:none;border-radius:6px;padding:6px 12px;
font-size:0.875rem;white-space:nowrap;}
.page-nav:hover{background:#e8ebf0;color:#111827;}
.page-nav-active{background:#1f2937;color:#ffffff;font-weight:600;}
.page-nav-active:hover{background:#1f2937;color:#ffffff;}
"""


def install_styles() -> None:
    """Inject the document-level CSS once per page build."""
    ui.dark_mode(False)
    ui.add_head_html(f"<style>{_GLOBAL_CSS}</style>")


def page_shell(active: str, subtitle: str = "") -> None:
    """Render the platform header for ``active`` (``aef`` or ``dw``)."""
    install_styles()
    with ui.row().classes(
        "w-full items-center no-wrap gap-6 px-4 bg-white border-b border-gray-200 shadow-sm"
    ).style(f"height:{HEADER_HEIGHT};min-height:{HEADER_HEIGHT}"):
        ui.label("AEF 变化检测平台").classes("text-base font-semibold text-gray-800 shrink-0")
        with ui.row().classes("items-center gap-1 no-wrap"):
            for key, label, href in PAGES:
                classes = "page-nav" + (" page-nav-active" if key == active else "")
                ui.link(label, href).classes(classes)
        if subtitle:
            ui.label(subtitle).classes(
                "text-xs text-gray-500 ml-auto truncate hidden md:block"
            )
