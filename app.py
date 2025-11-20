import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y ESTILO GLOBAL
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Panel de Inventario y Traslados",
    layout="wide",
    page_icon="📦"
)

# CSS para reducir espacios y dar estilo
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 0.8rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    h1, h2, h3 {
        margin-bottom: 0.3rem;
    }
    .small-margin {
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffc928, #ffb300);
        border-radius: 16px;
        padding: 14px 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
        color: #111;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        opacity: 0.9;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        margin-top: 2px;
        margin-bottom: 0px;
    }
    .metric-sub {
        font-size: 0.8rem;
        opacity: 0.9;
    }
    table.dataframe, .dataframe th, .dataframe td {
        border: 0px solid black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------------------------
def format_int(x) -> str:
    """Entero con separador de miles en formato español: 16.901"""
    try:
        n = int(round(float(x)))
    except Exception:
        return ""
    return f"{n:,}".replace(",", ".")


def format_pct_from_fraction(frac: float) -> str:
    """Recibe fracción (0.435) y devuelve '43,5%'."""
    if pd.isna(frac):
        return ""
    return f"{frac*100:.1f}".replace(".", ",") + "%"


@st.cache_data(show_spinner=False)
def load_balance(path: str = "Balance.csv"):
    """
    Carga Balance.csv.
    Encabezados:
    Codigo, Clave, Descripcion, Adelitas, Adelitas.1, San Agustin, San Agustin.1, ...
    Inventarios  = columnas que no son base y NO terminan en '.1'
    Clasificación = <InvCol> + '.1'
    """
    df = pd.read_csv(path, encoding="latin-1")

    base_cols = ["Codigo", "Clave", "Descripcion"]
    all_cols = list(df.columns)

    # columnas de inventario (almacenes) en orden alfabético
    inv_cols = [
        c for c in all_cols
        if c not in base_cols and not c.endswith(".1")
    ]
    inv_cols = sorted(inv_cols)

    # columnas de clasificación asociadas
    class_cols = {}
    for inv in inv_cols:
        cls_name = inv + ".1"
        if cls_name in all_cols:
            class_cols[inv] = cls_name
        else:
            class_cols[inv] = None  # por si hubiera algún almacén sin columna .1

    # Asegurar tipos numéricos en inventarios
    for col in inv_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    warehouses = inv_cols[:]  # nombres legibles de almacenes (ya ordenados)

    return df, warehouses, inv_cols, class_cols


def get_origin_df(data, inv_col, class_col):
    """Sub-dataframe con información básica de un almacén origen."""
    df = data[["Codigo", "Clave", "Descripcion", inv_col, class_col]].copy()
    df.rename(
        columns={inv_col: "Existencia", class_col: "Clasificacion"},
        inplace=True,
    )
    return df


def compute_kpis(df_origin, umbral_bajos_ab: int):
    """
    Calcula:
    - skus_totales
    - skus_activos
    - bajos_ab
    - conteo por clasificacion A/B/C
    - conteo de Sin Mov con existencia > 0
    """
    df = df_origin.copy()
    df["Existencia"] = pd.to_numeric(df["Existencia"], errors="coerce").fillna(0).astype(int)
    df["Clasificacion"] = df["Clasificacion"].fillna("").astype(str).str.strip()

    skus_totales = df["Clave"].nunique()

    mask_abc = df["Clasificacion"].isin(["A", "B", "C"])
    mask_sinmov = df["Clasificacion"].str.lower().str.contains("sin")
    mask_sinmov_con_exist = mask_sinmov & (df["Existencia"] > 0)

    skus_activos = df[mask_abc | mask_sinmov_con_exist]["Clave"].nunique()

    # Bajos A/B (críticos)
    mask_ab = df["Clasificacion"].isin(["A", "B"])
    bajos_ab = df[mask_ab & (df["Existencia"] > 0) & (df["Existencia"] <= umbral_bajos_ab)][
        "Clave"
    ].nunique()

    conteo_A = df[df["Clasificacion"] == "A"]["Clave"].nunique()
    conteo_B = df[df["Clasificacion"] == "B"]["Clave"].nunique()
    conteo_C = df[df["Clasificacion"] == "C"]["Clave"].nunique()
    conteo_sinmov = df[mask_sinmov_con_exist]["Clave"].nunique()

    def pct(v):
        if skus_activos == 0:
            return 0.0
        return v / skus_activos

    pct_A = pct(conteo_A)
    pct_B = pct(conteo_B)
    pct_C = pct(conteo_C)
    pct_sinmov_frac = pct(conteo_sinmov)

    return {
        "skus_totales": skus_totales,
        "skus_activos": skus_activos,
        "bajos_ab": bajos_ab,
        "A": conteo_A,
        "B": conteo_B,
        "C": conteo_C,
        "SinMov": conteo_sinmov,
        "pct_A": pct_A,
        "pct_B": pct_B,
        "pct_C": pct_C,
        "pct_SinMov": pct_sinmov_frac,
    }


def build_suggestions(
    data,
    inv_cols,
    class_cols,
    origin_inv_col: str,
    dest_inv_cols: list,
    umbral_bajos_ab: int,
):
    """
    Genera sugerencias de traslado desde varios almacenes destino
    hacia un único almacén origen.
    """
    origin_class_col = class_cols[origin_inv_col]
    registros = []

    for _, row in data.iterrows():
        cod = row["Codigo"]
        clave = row["Clave"]
        desc = row["Descripcion"]

        exist_o = int(row[origin_inv_col])
        clas_o = str(row[origin_class_col]).strip()

        if clas_o not in ["A", "B"]:
            continue

        faltante = max(0, umbral_bajos_ab - exist_o)
        if faltante <= 0:
            continue

        for dest_inv_col in dest_inv_cols:
            dest_class_col = class_cols[dest_inv_col]
            exist_d = int(row[dest_inv_col])
            clas_d = str(row[dest_class_col]).strip()

            if exist_d <= 0:
                continue

            if not (clas_d == "C" or "sin" in clas_d.lower()):
                continue

            sugerido = min(exist_d, faltante)
            if sugerido <= 0:
                continue

            registros.append(
                {
                    "Codigo": cod,
                    "Clave": clave,
                    "Descripción": desc,
                    "Almacén origen": origin_inv_col,
                    "Existencia origen": exist_o,
                    "Clasif. origen": clas_o,
                    "Almacén destino": dest_inv_col,
                    "Existencia destino": exist_d,
                    "Clasif. destino": clas_d,
                    "Faltante destino": faltante,
                    "Sugerido trasladar": sugerido,
                }
            )

    if not registros:
        return pd.DataFrame()

    df_sug = pd.DataFrame(registros)
    df_sug = df_sug.sort_values(
        by=["Sugerido trasladar", "Clave"], ascending=[False, True]
    ).head(50)

    for col in [
        "Existencia origen",
        "Existencia destino",
        "Faltante destino",
        "Sugerido trasladar",
    ]:
        df_sug[col] = pd.to_numeric(df_sug[col], errors="coerce").fillna(0).astype(int)

    return df_sug


def style_class_colors(df, pairs_for_buscar=None):
    """
    Colorea según clasificación:
    - A/B con existencia > 0 -> verde
    - A/B con existencia = 0 -> azul
    - C con existencia > 0  -> naranja
    - Sin movimiento con existencia > 0 -> rojo
    """
    if pairs_for_buscar is not None:
        inv_class_pairs = pairs_for_buscar
    else:
        inv_class_pairs = []

    def _apply_row(row):
        styles = [""] * len(row)

        if inv_class_pairs:
            # Tabla buscador
            for inv_col, class_col in inv_class_pairs:
                if inv_col not in df.columns or class_col not in df.columns:
                    continue
                try:
                    exist = float(row[inv_col])
                except Exception:
                    exist = 0.0
                clas = str(row[class_col]).strip()

                color = None
                if clas in ["A", "B"] and exist > 0:
                    color = "#1b8a3a"
                elif clas in ["A", "B"] and exist == 0:
                    color = "#1e88e5"
                elif clas == "C" and exist > 0:
                    color = "#fb8c00"
                elif "sin" in clas.lower() and exist > 0:
                    color = "#c62828"

                if color:
                    for colname in (inv_col, class_col):
                        if colname in df.columns:
                            idx = df.columns.get_loc(colname)
                            styles[idx] = f"background-color: {color}; color: white;"

        else:
            # Tabla sugeridos
            try:
                exist_o = float(row["Existencia origen"])
            except Exception:
                exist_o = 0.0
            clas_o = str(row["Clasif. origen"]).strip()

            try:
                exist_d = float(row["Existencia destino"])
            except Exception:
                exist_d = 0.0
            clas_d = str(row["Clasif. destino"]).strip()

            def color_from_cls(clas, exist):
                if clas in ["A", "B"] and exist > 0:
                    return "#1b8a3a"
                if clas in ["A", "B"] and exist == 0:
                    return "#1e88e5"
                if clas == "C" and exist > 0:
                    return "#fb8c00"
                if "sin" in clas.lower() and exist > 0:
                    return "#c62828"
                return None

            color_o = color_from_cls(clas_o, exist_o)
            color_d = color_from_cls(clas_d, exist_d)

            for colname, color in [
                ("Almacén origen", color_o),
                ("Existencia origen", color_o),
                ("Clasif. origen", color_o),
                ("Almacén destino", color_d),
                ("Existencia destino", color_d),
                ("Clasif. destino", color_d),
            ]:
                if color and colname in df.columns:
                    idx = df.columns.get_loc(colname)
                    styles[idx] = f"background-color: {color}; color: white;"

        return styles

    styler = df.style.apply(_apply_row, axis=1)
    return styler


# ---------------------------------------------------------------------
# CARGA DE DATOS
# ---------------------------------------------------------------------
try:
    data, warehouses, inv_cols, class_cols = load_balance("Balance.csv")
except Exception as e:
    st.error(
        f"No se pudo cargar **Balance.csv**.\n\n"
        f"Detalle técnico: `{e}`\n\n"
        "Verifique que el archivo esté en la misma carpeta que `app.py`."
    )
    st.stop()

# ---------------------------------------------------------------------
# ENCABEZADO
# ---------------------------------------------------------------------
st.title("🏭 Configuración de almacenes")

# ---------------------------------------------------------------------
# CONTROLES
# ---------------------------------------------------------------------
col_f1, col_f2 = st.columns([1.5, 2.0])

# Origen (con 'Todos')
origin_options = ["Todos"] + warehouses
default_origin = "Matriz" if "Matriz" in warehouses else origin_options[1]
origin_index = origin_options.index(default_origin)

with col_f1:
    origen_sel = st.selectbox(
        "Almacén que estoy revisando (origen)",
        origin_options,
        index=origin_index,
    )

with col_f2:
    umbral_bajos_ab = st.slider(
        "Umbral de existencia baja (para KPI de críticos A/B)",
        min_value=1,
        max_value=100,
        value=50,
    )

# Destinos (solo si no es 'Todos')
if origen_sel == "Todos":
    destinos_sel = warehouses[:]  # solo para que no esté vacío
else:
    opciones_destino = [w for w in warehouses if w != origen_sel]
    destinos_sel = st.multiselect(
        "Almacenes de donde puedo solicitar",
        opciones_destino,
        default=opciones_destino,
    )

# ---------------------------------------------------------------------
# PANEL INVENTARIO
# ---------------------------------------------------------------------
st.markdown("---")
st.header("🔍 Panel de Inventario y Traslados")

if origen_sel == "Todos":
    # Sumamos KPIs de todos los almacenes
    total = {
        "skus_totales": 0,
        "skus_activos": 0,
        "bajos_ab": 0,
        "A": 0,
        "B": 0,
        "C": 0,
        "SinMov": 0,
    }
    for inv in inv_cols:
        cls = class_cols[inv]
        df_o = get_origin_df(data, inv, cls)
        k = compute_kpis(df_o, umbral_bajos_ab)
        for key in total:
            total[key] += k[key]

    sa = total["skus_activos"]

    def pct(v):
        if sa == 0:
            return 0.0
        return v / sa

    pct_A = pct(total["A"])
    pct_B = pct(total["B"])
    pct_C = pct(total["C"])
    pct_sin = pct(total["SinMov"])

    kpis_to_show = {
        "Titulo_origen": "SKU en archivo",
        "Origen_txt": "Todos",
        "skus_totales": total["skus_totales"],
        "skus_activos": total["skus_activos"],
        "bajos_ab": total["bajos_ab"],
        "A": total["A"],
        "B": total["B"],
        "C": total["C"],
        "SinMov": total["SinMov"],
        "pct_A": pct_A,
        "pct_B": pct_B,
        "pct_C": pct_C,
        "pct_SinMov": pct_sin,
    }
else:
    origin_inv_col = origen_sel
    origin_class_col = class_cols[origin_inv_col]
    df_origin = get_origin_df(data, origin_inv_col, origin_class_col)
    k = compute_kpis(df_origin, umbral_bajos_ab)
    kpis_to_show = {
        "Titulo_origen": "SKU en origen",
        "Origen_txt": origen_sel,
        "skus_totales": k["skus_totales"],
        "skus_activos": k["skus_activos"],
        "bajos_ab": k["bajos_ab"],
        "A": k["A"],
        "B": k["B"],
        "C": k["C"],
        "SinMov": k["SinMov"],
        "pct_A": k["pct_A"],
        "pct_B": k["pct_B"],
        "pct_C": k["pct_C"],
        "pct_SinMov": k["pct_SinMov"],
    }

st.markdown(
    f"<p class='small-margin'>Análisis del archivo <b>Balance.csv</b> "
    f"y sugerencias entre almacenes. Origen actual del almacén: "
    f"<b>{kpis_to_show['Origen_txt']}</b>.</p>",
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5, m6, m7 = st.columns(7)

with m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{kpis_to_show['Titulo_origen']}</div>
            <div class="metric-value">{format_int(kpis_to_show['skus_totales'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">SKU´s activos</div>
            <div class="metric-value">{format_int(kpis_to_show['skus_activos'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">SKU (Bajos A/B)</div>
            <div class="metric-value">{format_int(kpis_to_show['bajos_ab'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Clasificación de SKU A</div>
            <div class="metric-value">{format_int(kpis_to_show['A'])}</div>
            <div class="metric-sub">{format_pct_from_fraction(kpis_to_show['pct_A'])} de SKU activos</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Clasificación SKU B</div>
            <div class="metric-value">{format_int(kpis_to_show['B'])}</div>
            <div class="metric-sub">{format_pct_from_fraction(kpis_to_show['pct_B'])} de SKU activos</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m6:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Clasificación SKU C</div>
            <div class="metric-value">{format_int(kpis_to_show['C'])}</div>
            <div class="metric-sub">{format_pct_from_fraction(kpis_to_show['pct_C'])} de SKU activos</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m7:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Sin movimiento</div>
            <div class="metric-value">{format_int(kpis_to_show['SinMov'])}</div>
            <div class="metric-sub">{format_pct_from_fraction(kpis_to_show['pct_SinMov'])} de SKU activos</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

# ---------------------------------------------------------------------
# TABS: BUSCADOR Y SUGERIDOS
# ---------------------------------------------------------------------
tab_buscar, tab_sugeridos = st.tabs(["🔎 Buscador", "📦 Sugeridos de traslado"])

with tab_buscar:
    st.markdown("#### Buscador de artículos")
    busqueda = st.text_input(
        "Buscar por Clave o Descripción (deja vacío para ver todo):",
        value="",
    )

    df_buscar = data.copy()
    if busqueda.strip():
        txt = busqueda.strip().lower()
        mask = (
            df_buscar["Clave"].astype(str).str.lower().str.contains(txt)
            | df_buscar["Descripcion"].astype(str).str.lower().str.contains(txt)
        )
        df_buscar = df_buscar[mask]

    # --------- CAMBIO: agrupar nombres similares en la primera tabla ---------
    # Clave / Descripción primero, luego cada almacén pegado a su .1
    cols_show = ["Clave", "Descripcion"]
    for inv in inv_cols:
        cols_show.append(inv)
        cls = class_cols[inv]
        if cls is not None:
            cols_show.append(cls)
    # -------------------------------------------------------------------------
    df_buscar = df_buscar[cols_show].copy()

    for col in inv_cols:
        df_buscar[col] = pd.to_numeric(df_buscar[col], errors="coerce").fillna(0).astype(
            int
        )

    styler_buscar = style_class_colors(
        df_buscar, pairs_for_buscar=[(c, class_cols[c]) for c in inv_cols]
    )
    styler_buscar = styler_buscar.format(format_int, subset=inv_cols)

    st.dataframe(styler_buscar, use_container_width=True, height=480)

with tab_sugeridos:
    st.markdown("#### Sugeridos de traslado")

    if origen_sel == "Todos":
        st.info("Selecciona un **almacén específico** como origen para ver sugerencias.")
    elif not destinos_sel:
        st.info("Selecciona al menos **un almacén destino** para calcular sugerencias.")
    else:
        origin_inv_col = origen_sel
        df_sug = build_suggestions(
            data,
            inv_cols,
            class_cols,
            origin_inv_col=origin_inv_col,
            dest_inv_cols=destinos_sel,
            umbral_bajos_ab=umbral_bajos_ab,
        )

        if df_sug.empty:
            st.warning("No se encontraron sugerencias de traslado con los criterios actuales.")
        else:
            st.markdown(
                f"Se muestran hasta **{len(df_sug)} artículos** (ordenados por mayor sugerencia)."
            )
            styler_sug = style_class_colors(df_sug)
            styler_sug = styler_sug.format(
                format_int,
                subset=[
                    "Existencia origen",
                    "Existencia destino",
                    "Faltante destino",
                    "Sugerido trasladar",
                ],
            )
            st.dataframe(styler_sug, use_container_width=True, height=520)
