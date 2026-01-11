# app.py
# pip install dash pandas openpyxl

from __future__ import annotations

import io
import uuid
from typing import Dict, List, Tuple

import pandas as pd
import dash
from dash import Dash, html, dcc, Input, Output, State, ALL


# -----------------------------
# Defaults / Helpers
# -----------------------------
DEFAULT_CATEGORIES = ["Geld", "Haus", "Garten", "Feld", "Wald", "Gegenstände"]

RELATIONS = [
    {"label": "Ehegatte / eingetr. Lebenspartner", "value": "spouse"},
    {"label": "Kind / Stiefkind", "value": "child"},
    {"label": "Enkel (Eltern leben)", "value": "grandchild"},
    {"label": "Enkel (Eltern vorverstorben)", "value": "grandchild_parent_dead"},
    {"label": "Eltern / Großeltern (Erwerb von Todes wegen)", "value": "parents"},
    {"label": "Geschwister / Nichte/Neffe / Schwiegerkind / Schwiegereltern / gesch. Ehegatte", "value": "class2"},
    {"label": "Sonstige (nicht verwandt / entfernte Verwandte)", "value": "class3"},
]


def new_id() -> str:
    return str(uuid.uuid4())[:8]


def clamp_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def normalize_name(x: str, fallback: str) -> str:
    x = (x or "").strip()
    return x if x else fallback


def get_ctx_value_for(component_type: str, component_id: str, prop: str):
    """
    Reads the current value from dash.ctx.inputs for a pattern-matching component.
    Example key: {"id":"abcd","type":"heir-relation"}.value
    """
    key = f'{{"id":"{component_id}","type":"{component_type}"}}.{prop}'
    return dash.ctx.inputs.get(key)


# -----------------------------
# Erbschaftsteuer (DE, vereinfacht)
# -----------------------------
BRACKETS = [75_000, 300_000, 600_000, 6_000_000, 13_000_000, 26_000_000, float("inf")]

RATES = {
    "I":  [0.07, 0.11, 0.15, 0.19, 0.23, 0.27, 0.30],
    "II": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.43],
    "III":[0.30, 0.30, 0.30, 0.30, 0.50, 0.50, 0.50],
}


def relation_to_class_and_allowance(rel: str) -> Tuple[str, float]:
    if rel == "spouse":
        return "I", 500_000.0
    if rel == "child":
        return "I", 400_000.0
    if rel == "grandchild":
        return "I", 200_000.0
    if rel == "grandchild_parent_dead":
        return "I", 400_000.0
    if rel == "parents":
        return "I", 100_000.0
    if rel == "class2":
        return "II", 20_000.0
    return "III", 20_000.0


def tax_rate(tax_class: str, taxable: float) -> float:
    rates = RATES[tax_class]
    for idx, limit in enumerate(BRACKETS):
        if taxable <= limit:
            return rates[idx]
    return rates[-1]


def compute_inheritance_tax(net_acquisition: float, rel: str) -> Tuple[str, float, float, float, float]:
    tax_class, allowance = relation_to_class_and_allowance(rel)
    taxable = max(0.0, net_acquisition - allowance)
    rate = tax_rate(tax_class, taxable) if taxable > 0 else 0.0
    tax = taxable * rate
    return tax_class, allowance, taxable, rate, tax


# -----------------------------
# Allocation logic
# -----------------------------
def greedy_partition_assets(assets: List[dict], heir_names: List[str], target_remaining: Dict[str, float]):
    allocation = {h: [] for h in heir_names}
    totals = {h: 0.0 for h in heir_names}
    need = {h: max(0.0, float(target_remaining.get(h, 0.0))) for h in heir_names}

    assets_sorted = sorted(assets, key=lambda a: a["value"], reverse=True)

    for a in assets_sorted:
        if not heir_names:
            break

        max_need = max(need.values()) if need else 0.0
        if max_need > 0:
            chosen = max(need, key=lambda h: need[h])
        else:
            chosen = min(totals, key=lambda h: totals[h])

        allocation[chosen].append(a)
        totals[chosen] += a["value"]
        need[chosen] = max(0.0, need[chosen] - a["value"])

    return allocation, totals


def compute_payments(heir_names: List[str], totals: Dict[str, float], targets: Dict[str, float]) -> pd.DataFrame:
    diff = {h: float(totals.get(h, 0.0) - targets.get(h, 0.0)) for h in heir_names}
    debtors = [(h, diff[h]) for h in heir_names if diff[h] > 1e-9]
    creditors = [(h, -diff[h]) for h in heir_names if diff[h] < -1e-9]

    payments = []
    i = j = 0
    while i < len(debtors) and j < len(creditors):
        d_name, d_amt = debtors[i]
        c_name, c_amt = creditors[j]
        pay = min(d_amt, c_amt)

        if pay > 1e-9:
            payments.append({"Von": d_name, "An": c_name, "Betrag": round(pay, 2)})

        d_amt -= pay
        c_amt -= pay
        debtors[i] = (d_name, d_amt)
        creditors[j] = (c_name, c_amt)

        if d_amt <= 1e-9:
            i += 1
        if c_amt <= 1e-9:
            j += 1

    return pd.DataFrame(payments)


def compute_allocation(state: dict):
    categories = state.get("categories") or DEFAULT_CATEGORIES
    assets = state.get("assets") or []
    heirs = state.get("heirs") or []
    favorites = state.get("favorites") or {}

    # sanitize assets
    clean_assets = []
    for a in assets:
        aid = a.get("id") or new_id()
        name = normalize_name(a.get("name"), f"Asset {aid}")
        cat = a.get("category") or (categories[0] if categories else "Sonstiges")
        val = clamp_float(a.get("value"), 0.0)
        clean_assets.append({"id": aid, "name": name, "category": cat, "value": float(val)})

    # sanitize heirs
    clean_heirs = []
    for h in heirs:
        hid = h.get("id") or new_id()
        name = normalize_name(h.get("name"), f"Erbe {hid}")
        rel = h.get("relation") or "class3"
        clean_heirs.append({"id": hid, "name": name, "relation": rel})

    heir_names = [h["name"] for h in clean_heirs]
    estate_total = sum(a["value"] for a in clean_assets)

    # targets: equal split
    targets = {}
    if clean_heirs:
        per = estate_total / len(clean_heirs)
        for hn in heir_names:
            targets[hn] = per

    # Resolve favorites: each asset max 1 heir (first in heirs list wins)
    id_to_name = {h["id"]: h["name"] for h in clean_heirs}
    heir_order = [h["name"] for h in clean_heirs]

    desired = []
    for hid, asset_ids in favorites.items():
        hname = id_to_name.get(hid)
        if not hname:
            continue
        for aid in (asset_ids or []):
            desired.append((aid, hname))

    desired_sorted = sorted(
        desired,
        key=lambda x: heir_order.index(x[1]) if x[1] in heir_order else 10**9
    )

    asset_to_heir: Dict[str, str] = {}
    conflicts = []
    for aid, hname in desired_sorted:
        if aid not in asset_to_heir:
            asset_to_heir[aid] = hname
        else:
            conflicts.append((aid, asset_to_heir[aid], hname))

    fixed_assets_by_heir = {hn: [] for hn in heir_names}
    for a in clean_assets:
        if a["id"] in asset_to_heir:
            fixed_assets_by_heir[asset_to_heir[a["id"]]].append(a)

    fixed_value_by_heir = {hn: sum(x["value"] for x in fixed_assets_by_heir[hn]) for hn in heir_names}

    fixed_asset_ids = set(asset_to_heir.keys())
    remaining_assets = [a for a in clean_assets if a["id"] not in fixed_asset_ids]

    target_remaining = {hn: max(0.0, targets.get(hn, 0.0) - fixed_value_by_heir.get(hn, 0.0)) for hn in heir_names}

    greedy_alloc, _ = greedy_partition_assets(remaining_assets, heir_names, target_remaining)

    final_alloc = {hn: [] for hn in heir_names}
    final_totals = {hn: 0.0 for hn in heir_names}
    for hn in heir_names:
        final_alloc[hn].extend(fixed_assets_by_heir.get(hn, []))
        final_alloc[hn].extend(greedy_alloc.get(hn, []))
        final_totals[hn] = sum(a["value"] for a in final_alloc[hn])

    df_pay = compute_payments(heir_names, final_totals, targets)
    received = df_pay.groupby("An")["Betrag"].sum().to_dict() if not df_pay.empty else {}
    paid = df_pay.groupby("Von")["Betrag"].sum().to_dict() if not df_pay.empty else {}

    df_sum = pd.DataFrame([{
        "Erbe": hn,
        "Zielwert": round(targets.get(hn, 0.0), 2),
        "Istwert (Assets)": round(final_totals.get(hn, 0.0), 2),
        "Zahlt (Ausgleich)": round(float(paid.get(hn, 0.0)), 2),
        "Erhält (Ausgleich)": round(float(received.get(hn, 0.0)), 2),
        "Netto nach Ausgleich": round(final_totals.get(hn, 0.0) - float(paid.get(hn, 0.0)) + float(received.get(hn, 0.0)), 2),
    } for hn in heir_names])

    rows = []
    for hn in heir_names:
        items = final_alloc.get(hn, [])
        fixed_ids_h = {a["id"] for a in fixed_assets_by_heir.get(hn, [])}
        if not items:
            rows.append({"Erbe": hn, "Asset": "—", "Kategorie": "—", "Wert": 0.0, "Fix (Favorit)": ""})
        else:
            for it in items:
                rows.append({
                    "Erbe": hn,
                    "Asset": it["name"],
                    "Kategorie": it["category"],
                    "Wert": it["value"],
                    "Fix (Favorit)": "ja" if it["id"] in fixed_ids_h else "",
                })
    df_items = pd.DataFrame(rows)

    df_conf = pd.DataFrame([{
        "Asset-ID": aid,
        "Bereits zugeordnet an": h1,
        "Weitere Anfrage von": h2,
    } for (aid, h1, h2) in conflicts])

    # tax based on Netto nach Ausgleich
    name_to_rel = {h["name"]: (h.get("relation") or "class3") for h in clean_heirs}
    tax_rows = []
    for hn in heir_names:
        net = float(df_sum.loc[df_sum["Erbe"] == hn, "Netto nach Ausgleich"].iloc[0]) if not df_sum.empty else 0.0
        rel = name_to_rel.get(hn, "class3")
        tclass, allowance, taxable, rate, tax = compute_inheritance_tax(net, rel)
        tax_rows.append({
            "Erbe": hn,
            "Steuerklasse": tclass,
            "Freibetrag": round(allowance, 2),
            "Netto-Erwerb": round(net, 2),
            "Steuerpflichtig": round(taxable, 2),
            "Steuersatz": f"{rate*100:.0f}%" if rate > 0 else "0%",
            "Erbschaftsteuer": round(tax, 2),
            "Netto nach Steuer": round(net - tax, 2),
        })
    df_tax = pd.DataFrame(tax_rows)

    return {
        "estate_total": estate_total,
        "df_sum": df_sum,
        "df_items": df_items,
        "df_pay": df_pay,
        "df_tax": df_tax,
        "df_conf": df_conf,
        "clean_assets": clean_assets,
        "clean_heirs": clean_heirs,
        "categories": categories,
    }


# -----------------------------
# UI helpers
# -----------------------------
def df_to_html_table(df: pd.DataFrame):
    if df is None or df.empty:
        return html.Div("—")
    return html.Table(
        style={"borderCollapse": "collapse", "width": "100%"},
        children=[
            html.Thead(
                html.Tr([
                    html.Th(c, style={"textAlign": "left", "borderBottom": "1px solid #ddd", "padding": "6px"})
                    for c in df.columns
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(
                        f"{v:,.2f}" if isinstance(v, (int, float)) else str(v),
                        style={"borderBottom": "1px solid #f0f0f0", "padding": "6px", "verticalAlign": "top"},
                    )
                    for v in row
                ])
                for row in df.itertuples(index=False, name=None)
            ]),
        ],
    )


def make_asset_row(asset, categories):
    return html.Div(
        style={"display": "grid", "gridTemplateColumns": "2fr 1fr 1fr 44px", "gap": "8px", "alignItems": "center", "marginBottom": "8px"},
        children=[
            dcc.Input(
                id={"type": "asset-name", "id": asset["id"]},
                value=asset.get("name", ""),
                placeholder="Bezeichnung (z.B. Haus Berlin, Konto, Traktor …)",
                type="text",
                debounce=True,
                persistence=True,
            ),
            dcc.Dropdown(
                id={"type": "asset-category", "id": asset["id"]},
                options=[{"label": c, "value": c} for c in categories],
                value=asset.get("category") or (categories[0] if categories else "Sonstiges"),
                clearable=False,
                persistence=True,
            ),
            dcc.Input(
                id={"type": "asset-value", "id": asset["id"]},
                value=asset.get("value", 0),
                placeholder="Wert (€)",
                type="number",
                min=0,
                step=1000,
                debounce=True,
                persistence=True,
            ),
            html.Button("✕", id={"type": "asset-delete", "id": asset["id"]}, title="Asset entfernen"),
        ],
    )


def make_heir_row(heir):
    return html.Div(
        style={"display": "grid", "gridTemplateColumns": "1fr 1fr 44px", "gap": "8px", "alignItems": "center", "marginBottom": "8px"},
        children=[
            dcc.Input(
                id={"type": "heir-name", "id": heir["id"]},
                value=heir.get("name", ""),
                placeholder="Name des Erben",
                type="text",
                debounce=True,
                persistence=True,
            ),
            dcc.Dropdown(
                id={"type": "heir-relation", "id": heir["id"]},
                options=RELATIONS,
                value=heir.get("relation", "class3"),
                clearable=False,
                persistence=True,
            ),
            html.Button("✕", id={"type": "heir-delete", "id": heir["id"]}, title="Erbe entfernen"),
        ],
    )


def make_favorites_row(heir, asset_options, selected_asset_ids):
    return html.Div(
        style={"display": "grid", "gridTemplateColumns": "1fr 2fr", "gap": "10px", "alignItems": "center", "marginBottom": "10px"},
        children=[
            html.Div(heir["name"], style={"fontWeight": 600}),
            dcc.Dropdown(
                id={"type": "fav-assets", "id": heir["id"]},
                options=asset_options,
                value=selected_asset_ids or [],
                multi=True,
                placeholder="Favoriten auswählen (fix zuordnen)",
                persistence=True,
            ),
        ],
    )

def privacy_footer():
    return html.Footer(
        style={
            "marginTop": "40px",
            "paddingTop": "16px",
            "borderTop": "1px solid #e5e5e5",
            "fontSize": "13px",
            "color": "#555",
            "lineHeight": "1.5",
        },
        children=[
            html.Div([
                html.Strong("Datenschutzhinweis"),
            ]),
            html.Div(
                "Diese Webanwendung dient ausschließlich der interaktiven Berechnung und Visualisierung "
                "von Erbverteilungen. Alle eingegebenen Daten (z. B. Namen, Vermögenswerte, Verwandtschaftsverhältnisse) "
                "werden ausschließlich im Browser des Nutzers verarbeitet und nicht dauerhaft gespeichert."
            ),
            html.Div(
                "Es erfolgt keine Speicherung in Datenbanken, keine Weitergabe an Dritte und kein Tracking des Nutzerverhaltens. "
                "Beim Schließen oder Neuladen der Seite gehen alle Eingaben verloren."
            ),
            html.Div(
                "Die Anwendung wird auf Servern des Hosting-Anbieters Render betrieben. "
                "Die Datenübertragung erfolgt verschlüsselt (HTTPS). "
                "Es werden keine Cookies zu Analyse- oder Marketingzwecken eingesetzt."
            ),
        ],
    )

# -----------------------------
# Dash App
# -----------------------------
app = Dash(__name__)
app.title = "NachlassNavi"
server = app.server

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "24px auto", "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Arial"},
    children=[
        html.H2("NachlassNavi"),
        html.Div(
            style={"opacity": 0.85, "marginBottom": "16px"},
            children="Assets erfassen \n• Erben + Verwandtschaft \n• Favoriten je Erbe \n• Restverteilung \n• Ausgleich \n• Erbschaftsteuer \n• XLSX Export",
        ),

        dcc.Store(
            id="state",
            data={
                "categories": DEFAULT_CATEGORIES,
                "assets": [
                    {"id": new_id(), "name": "Haus 1", "category": "Haus", "value": 350000},
                    {"id": new_id(), "name": "Garten 1", "category": "Garten", "value": 30000},
                    {"id": new_id(), "name": "Konto", "category": "Geld", "value": 80000},
                ],
                "heirs": [
                    {"id": new_id(), "name": "Erbe A", "relation": "child"},
                    {"id": new_id(), "name": "Erbe B", "relation": "child"},
                ],
                "favorites": {},
            },
        ),

        dcc.Download(id="download-xlsx"),

        dcc.Tabs(
            value="tab-assets",
            children=[
                dcc.Tab(label="Assets", value="tab-assets", children=[
                    html.Div(style={"padding": "12px 6px"}, children=[
                        html.H4("Kategorien"),
                        html.Div(style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "12px"}, children=[
                            dcc.Input(id="new-category", placeholder="Neue Kategorie (z.B. Kunst)", type="text", debounce=True, persistence=True),
                            html.Button("Kategorie hinzufügen", id="add-category"),
                        ]),
                        html.Div(id="category-list", style={"marginBottom": "18px"}),

                        html.H4("Assets"),
                        html.Div(style={"display": "flex", "gap": "8px", "marginBottom": "10px"}, children=[
                            html.Button("Asset hinzufügen", id="add-asset"),
                        ]),
                        html.Div(id="assets-container"),
                    ])
                ]),

                dcc.Tab(label="Erben", value="tab-heirs", children=[
                    html.Div(style={"padding": "12px 6px"}, children=[
                        html.H4("Erben (inkl. Verwandtschaft)"),
                        html.Div(style={"display": "flex", "gap": "8px", "marginBottom": "10px"}, children=[
                            html.Button("Erben hinzufügen", id="add-heir"),
                        ]),
                        html.Div(id="heirs-container"),
                    ])
                ]),

                dcc.Tab(label="Favoriten & Ergebnisse", value="tab-results", children=[
                    html.Div(style={"padding": "12px 6px"}, children=[
                        html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "gap": "10px", "flexWrap": "wrap"}, children=[
                            html.H4("Favoriten je Erbe (fixe Zuordnung)"),
                            html.Button("XLSX exportieren", id="export-xlsx"),
                        ]),
                        html.Div(id="favorites-container"),
                        html.Hr(),

                        html.Div(id="kpi-line", style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "12px"}),

                        html.H4("Summen je Erbe (inkl. Ausgleich)"),
                        html.Div(id="sum-table"),
                        html.Hr(),

                        html.H4("Ausgleichszahlungen"),
                        html.Div(id="pay-table"),
                        html.Hr(),

                        html.H4("Erbschaftsteuer (DE, vereinfacht)"),
                        html.Div(id="tax-table"),
                        html.Hr(),

                        html.H4("Zuteilung (Details)"),
                        html.Div(id="items-table"),

                        html.Div(id="conflict-box", style={"marginTop": "14px"}),
                    ])
                ]),
            ],
        ),
    ],
)


# -----------------------------
# Render lists from state
# -----------------------------
@app.callback(
    Output("assets-container", "children"),
    Output("heirs-container", "children"),
    Output("category-list", "children"),
    Input("state", "data"),
)
def render_forms(state):
    cats = state.get("categories") or DEFAULT_CATEGORIES
    assets = state.get("assets") or []
    heirs = state.get("heirs") or []

    asset_rows = [make_asset_row(a, cats) for a in assets]
    heir_rows = [make_heir_row(h) for h in heirs]

    cat_badges = html.Div(
        style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
        children=[html.Span(c, style={"padding": "4px 10px", "border": "1px solid #ddd", "borderRadius": "999px"}) for c in cats],
    )
    return asset_rows, heir_rows, cat_badges


@app.callback(
    Output("favorites-container", "children"),
    Input("state", "data"),
)
def render_favorites(state):
    heirs = state.get("heirs") or []
    assets = state.get("assets") or []
    fav = state.get("favorites") or {}

    if not heirs:
        return html.Div("Bitte zuerst Erben anlegen.")
    if not assets:
        return html.Div("Bitte zuerst Assets anlegen.")

    asset_options = [
        {"label": f'{a.get("name","—")} ({clamp_float(a.get("value"),0):,.0f} €)', "value": a.get("id")}
        for a in assets
    ]

    return [make_favorites_row(h, asset_options, fav.get(h.get("id"), [])) for h in heirs]


# -----------------------------
# Add / Delete (robust against re-render changes)
# -----------------------------
@app.callback(
    Output("state", "data", allow_duplicate=True),
    Input("add-asset", "n_clicks"),
    Input({"type": "asset-delete", "id": ALL}, "n_clicks"),
    State("state", "data"),
    prevent_initial_call=True,
)
def mutate_assets(_add, _del_clicks, state):
    trig = dash.ctx.triggered_id

    assets = list(state.get("assets") or [])

    if trig == "add-asset":
        assets.append({"id": new_id(), "name": "", "category": (state.get("categories") or DEFAULT_CATEGORIES)[0], "value": 0})
        state["assets"] = assets
        return state

    if isinstance(trig, dict) and trig.get("type") == "asset-delete":
        # IMPORTANT: ignore callbacks caused by re-render resetting n_clicks
        if not dash.ctx.triggered or not dash.ctx.triggered[0].get("value"):
            raise dash.exceptions.PreventUpdate

        tid = trig.get("id")
        if tid:
            assets = [a for a in assets if a.get("id") != tid]
            state["assets"] = assets

            # also remove from favorites
            fav = dict(state.get("favorites") or {})
            for hid in list(fav.keys()):
                fav[hid] = [x for x in (fav[hid] or []) if x != tid]
            state["favorites"] = fav
            return state

    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("state", "data", allow_duplicate=True),
    Input("add-heir", "n_clicks"),
    Input({"type": "heir-delete", "id": ALL}, "n_clicks"),
    State("state", "data"),
    prevent_initial_call=True,
)
def mutate_heirs(_add, _del_clicks, state):
    trig = dash.ctx.triggered_id

    heirs = list(state.get("heirs") or [])

    if trig == "add-heir":
        heirs.append({"id": new_id(), "name": "", "relation": "class3"})
        state["heirs"] = heirs
        return state

    if isinstance(trig, dict) and trig.get("type") == "heir-delete":
        # IMPORTANT: ignore callbacks caused by re-render resetting n_clicks
        if not dash.ctx.triggered or not dash.ctx.triggered[0].get("value"):
            raise dash.exceptions.PreventUpdate

        tid = trig.get("id")
        if tid:
            heirs = [h for h in heirs if h.get("id") != tid]
            state["heirs"] = heirs

            fav = dict(state.get("favorites") or {})
            if tid in fav:
                del fav[tid]
            state["favorites"] = fav
            return state

    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("state", "data", allow_duplicate=True),
    Input("add-category", "n_clicks"),
    State("new-category", "value"),
    State("state", "data"),
    prevent_initial_call=True,
)
def add_category(_n, new_cat, state):
    cat = (new_cat or "").strip()
    if not cat:
        raise dash.exceptions.PreventUpdate
    cats = list(state.get("categories") or DEFAULT_CATEGORIES)
    if cat not in cats:
        cats.append(cat)
    state["categories"] = cats
    return state


# -----------------------------
# Sync: update ONLY the triggered element by ID (NO fragile ordering)
# -----------------------------
@app.callback(
    Output("state", "data", allow_duplicate=True),
    Input({"type": "heir-name", "id": ALL}, "value"),
    State("state", "data"),
    prevent_initial_call=True,
)
def sync_heir_name(_vals, state):
    trig = dash.ctx.triggered_id
    if not (isinstance(trig, dict) and trig.get("type") == "heir-name"):
        raise dash.exceptions.PreventUpdate

    hid = trig["id"]
    new_name = get_ctx_value_for("heir-name", hid, "value") or ""

    heirs = list(state.get("heirs") or [])
    for h in heirs:
        if h.get("id") == hid:
            h["name"] = new_name
            break

    state["heirs"] = heirs
    return state


@app.callback(
    Output("state", "data", allow_duplicate=True),
    Input({"type": "heir-relation", "id": ALL}, "value"),
    State("state", "data"),
    prevent_initial_call=True,
)
def sync_heir_relation(_vals, state):
    trig = dash.ctx.triggered_id
    if not (isinstance(trig, dict) and trig.get("type") == "heir-relation"):
        raise dash.exceptions.PreventUpdate

    hid = trig["id"]
    new_rel = get_ctx_value_for("heir-relation", hid, "value") or "class3"

    heirs = list(state.get("heirs") or [])
    for h in heirs:
        if h.get("id") == hid:
            h["relation"] = new_rel
            break

    state["heirs"] = heirs
    return state


@app.callback(
    Output("state", "data", allow_duplicate=True),
    Input({"type": "fav-assets", "id": ALL}, "value"),
    State("state", "data"),
    prevent_initial_call=True,
)
def sync_favorites(_vals, state):
    trig = dash.ctx.triggered_id
    if not (isinstance(trig, dict) and trig.get("type") == "fav-assets"):
        raise dash.exceptions.PreventUpdate

    hid = trig["id"]
    fav_value = get_ctx_value_for("fav-assets", hid, "value") or []

    fav = dict(state.get("favorites") or {})
    fav[hid] = fav_value
    state["favorites"] = fav
    return state


@app.callback(
    Output("state", "data", allow_duplicate=True),
    Input({"type": "asset-name", "id": ALL}, "value"),
    Input({"type": "asset-value", "id": ALL}, "value"),
    Input({"type": "asset-category", "id": ALL}, "value"),
    State("state", "data"),
    prevent_initial_call=True,
)
def sync_assets(_name_vals, _value_vals, _cat_vals, state):
    trig = dash.ctx.triggered_id
    if not (isinstance(trig, dict) and trig.get("type") in {"asset-name", "asset-value", "asset-category"}):
        raise dash.exceptions.PreventUpdate

    aid = trig["id"]
    assets = list(state.get("assets") or [])

    for a in assets:
        if a.get("id") == aid:
            if trig["type"] == "asset-name":
                a["name"] = get_ctx_value_for("asset-name", aid, "value") or ""
            elif trig["type"] == "asset-value":
                a["value"] = get_ctx_value_for("asset-value", aid, "value") or 0
            elif trig["type"] == "asset-category":
                a["category"] = get_ctx_value_for("asset-category", aid, "value") or (state.get("categories") or DEFAULT_CATEGORIES)[0]
            break

    state["assets"] = assets
    return state


# -----------------------------
# Results rendering
# -----------------------------
@app.callback(
    Output("kpi-line", "children"),
    Output("sum-table", "children"),
    Output("pay-table", "children"),
    Output("tax-table", "children"),
    Output("items-table", "children"),
    Output("conflict-box", "children"),
    Input("state", "data"),
)
def render_results(state):
    res = compute_allocation(state)

    estate_total = res["estate_total"]
    heirs_count = len(res["clean_heirs"])
    assets_count = len(res["clean_assets"])

    df_sum = res["df_sum"]
    df_items = res["df_items"]
    df_pay = res["df_pay"]
    df_tax = res["df_tax"]
    df_conf = res["df_conf"]

    kpis = [
        html.Div(style={"padding": "10px 12px", "border": "1px solid #eee", "borderRadius": "12px"}, children=[
            html.Div("Gesamtwert Nachlass", style={"fontSize": "12px", "opacity": 0.75}),
            html.Div(f"{estate_total:,.2f} €", style={"fontSize": "20px", "fontWeight": 650}),
        ]),
        html.Div(style={"padding": "10px 12px", "border": "1px solid #eee", "borderRadius": "12px"}, children=[
            html.Div("Erben", style={"fontSize": "12px", "opacity": 0.75}),
            html.Div(f"{heirs_count}", style={"fontSize": "20px", "fontWeight": 650}),
        ]),
        html.Div(style={"padding": "10px 12px", "border": "1px solid #eee", "borderRadius": "12px"}, children=[
            html.Div("Assets", style={"fontSize": "12px", "opacity": 0.75}),
            html.Div(f"{assets_count}", style={"fontSize": "20px", "fontWeight": 650}),
        ]),
    ]

    sum_table = df_to_html_table(df_sum) if not df_sum.empty else html.Div("Bitte mindestens einen Erben anlegen.")
    pay_table = df_to_html_table(df_pay) if not df_pay.empty else html.Div("Keine Ausgleichszahlungen nötig.")
    tax_table = df_to_html_table(df_tax) if not df_tax.empty else html.Div("—")
    items_table = df_to_html_table(df_items) if not df_items.empty else html.Div("Noch keine Assets erfasst.")

    conflict_box = html.Div()
    if df_conf is not None and not df_conf.empty:
        conflict_box = html.Div(
            style={"border": "1px solid #f3c2c2", "background": "#fff7f7", "padding": "10px 12px", "borderRadius": "12px"},
            children=[
                html.Div("Konflikte bei Favoriten (Asset mehrfach markiert):", style={"fontWeight": 650, "marginBottom": "8px"}),
                df_to_html_table(df_conf),
            ],
        )

    return kpis, sum_table, pay_table, tax_table, items_table, conflict_box


# -----------------------------
# XLSX Export
# -----------------------------
@app.callback(
    Output("download-xlsx", "data"),
    Input("export-xlsx", "n_clicks"),
    State("state", "data"),
    prevent_initial_call=True,
)
def export_xlsx(_n, state):
    res = compute_allocation(state)

    df_assets = pd.DataFrame(res["clean_assets"])
    df_heirs = pd.DataFrame(res["clean_heirs"])
    df_items = res["df_items"]
    df_sum = res["df_sum"]
    df_pay = res["df_pay"]
    df_tax = res["df_tax"]
    df_conf = res["df_conf"]

    fav = state.get("favorites") or {}
    hid_to_name = {h["id"]: h["name"] for h in res["clean_heirs"]}
    aid_to_name = {a["id"]: a["name"] for a in res["clean_assets"]}

    fav_rows = []
    for hid, aids in fav.items():
        for aid in (aids or []):
            fav_rows.append({
                "Erbe": hid_to_name.get(hid, hid),
                "Asset": aid_to_name.get(aid, aid),
                "Asset-ID": aid,
            })
    df_fav = pd.DataFrame(fav_rows)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_sum.to_excel(writer, sheet_name="Summen", index=False)
        df_items.to_excel(writer, sheet_name="Zuteilung", index=False)
        (df_pay if not df_pay.empty else pd.DataFrame(columns=["Von", "An", "Betrag"])).to_excel(writer, sheet_name="Ausgleich", index=False)
        (df_tax if not df_tax.empty else pd.DataFrame(columns=[
            "Erbe","Steuerklasse","Freibetrag","Netto-Erwerb","Steuerpflichtig","Steuersatz","Erbschaftsteuer","Netto nach Steuer"
        ])).to_excel(writer, sheet_name="Erbschaftsteuer", index=False)

        df_assets.to_excel(writer, sheet_name="Assets_Input", index=False)
        df_heirs.to_excel(writer, sheet_name="Erben_Input", index=False)
        (df_fav if not df_fav.empty else pd.DataFrame(columns=["Erbe", "Asset", "Asset-ID"])).to_excel(writer, sheet_name="Favoriten_Input", index=False)
        (df_conf if not df_conf.empty else pd.DataFrame(columns=["Asset-ID", "Bereits zugeordnet an", "Weitere Anfrage von"])).to_excel(writer, sheet_name="Konflikte", index=False)

    output.seek(0)
    return dcc.send_bytes(output.getvalue(), "erbschaft_tool_zuteilung.xlsx")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
