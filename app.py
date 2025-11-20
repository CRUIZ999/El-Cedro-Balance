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
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    h1, h2, h3 {
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
    }
    .small-margin {
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffc928, #ffb300);
        border-radius: 16px;
        padding: 10px 16px;
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
    /* Reducir altura de filas en tablas */
    .dataframe td, .dataframe th {
        padding-top: 2px;
        padding-bottom: 2px;
        padding-left: 4px;
        padding-right: 4px;
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
    Encabezados esperados (ejemplo):
    Codigo, Clave, Descripcion, Matriz, Matriz.1, Adelitas, Adelitas.1, ...
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
    - skus_totales  -> todos los SKU distintos en el archivo
    - skus_activos  -> A + B + C (todos) + Sin mov (>0)
    - bajos_ab      -> A/B con existencia = 0
    - conteo A/B/C  -> TODOS los SKU de esa clasificación, sin importar existencia
    - SinMov        -> Sin mov con existencia > 0
    """
    df = df_origin.copy()
    df["Existencia"] = pd.to_numeric(df["Existencia"], errors="coerce").fillna(0).astype(int)
    df["Clasificacion"] = df["Clasificacion"].fillna("").astype(str).str.strip()

    # Total de SKU en el archivo (cualquier clasificación)
    skus_totales = df["Clave"].nunique()

    # Máscaras por clasificación
    mask_A = df["Clasificacion"] == "A"
    mask_B = df["Clasificacion"] == "B"
    mask_C = df["Clasificacion"] == "C"
    mask_sin = df["Clasificacion"].str.lower().str.contains("sin")

    # A, B, C -> TODOS los SKU de esa clasificación (sin importar existencia)
    conteo_A = df[mask_A]["Clave"].nunique()
    conteo_B = df[mask_B]["Clave"].nunique()
    conteo_C = df[mask_C]["Clave"].nunique()

    # Sin movimiento -> SOLO existencia > 0
    mask_sin_pos = mask_sin & (df["Existencia"] > 0)
    conteo_sinmov = df[mask_sin_pos]["Clave"].nunique()

    # SKU activos = A + B + C (todos) + Sin mov (>0)
    skus_activos = conteo_A + conteo_B + conteo_C + conteo_sinmov

    # Bajos A/B = existencia = 0
    mask_ab = mask_A | mask_B
    bajos_ab = df[mask_ab & (df["Existencia"] == 0)]["Clave"].nunique()

    # Porcentajes sobre SKU activos
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
    Sugeridos normales (reposiciones hacia el origen):

    - Origen: clasificación A o B (SIN importar su existencia).
    - Destino: C o Sin mov con existencia > 0.
    """
    origin_class_col = class_cols[origin_inv_col]
    registros = []

    data_filtrada = data.copy()
    data_filtrada["__exist_o__"] = pd.to_numeric(
        data_filtrada[origin_inv_col], errors="coerce"
    ).fillna(0).astype(int)
    data_filtrada["__clas_o__"] = data_filtrada[origin_class_col].astype(str).str.strip()

    # Solo A/B, sin filtrar por existencia
    data_filtrada = data_filtrada[data_filtrada["__clas_o__"].isin(["A", "B"])]

    for _, row in data_filtrada.iterrows():
        cod = row["Codigo"]
        clave = row["Clave"]
        desc = row["Descripcion"]

        exist_o = int(row["__exist_o__"])
        clas_o = row["__clas_o__"]

        for dest_inv_col in dest_inv_cols:
            dest_class_col = class_cols[dest_inv_col]
            exist_d = int(row[dest_inv_col])
            clas_d = str(row[dest_class_col]).strip()

            # Destino debe tener stock disponible
            if exist_d <= 0:
                continue

            # Destino debe ser C o Sin movimiento
            if not (clas_d == "C" or "sin" in clas_d.lower()):
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
                }
            )

    if not registros:
        return pd.DataFrame()

    df_sug = pd.DataFrame(registros)
    df_sug = df_sug.sort_values(
        by=["Existencia destino", "Clave"], ascending=[False, True]
    )
    return df_sug


def build_reverse_suggestions(
    data,
    inv_cols,
    class_cols,
    origin_inv_col: str,
    dest_inv_cols: list,
    umbral_bajos_ab: int,
):
    """
    Sugeridos inversos:
    - Origen: C o Sin mov con existencia > 0.
    - Destino: A/B (cualquier existencia).
    """
    origin_class_col = class_cols[origin_inv_col]
    registros = []

    data_filtrada = data.copy()
    data_filtrada["__exist_o__"] = pd.to_numeric(
        data_filtrada[origin_inv_col], errors="coerce"
    ).fillna(0).astype(int)
    data_filtrada["__clas_o__"] = data_filtrada[origin_class_col].astype(str).str.strip()

    # ORIGEN: C o Sin movimiento, con EXISTENCIA > 0
    mask_c = data_filtrada["__clas_o__"] == "C"
    mask_sin = data_filtrada["__clas_o__"].str.lower().str.contains("sin")
    data_filtrada = data_filtrada[(data_filtrada["__exist_o__"] > 0) & (mask_c | mask_sin)]

    for _, row in data_filtrada.iterrows():
        cod = row["Codigo"]
        clave = row["Clave"]
        desc = row["Descripcion"]

        exist_o = int(row["__exist_o__"])
        clas_o = row["__clas_o__"]

        for dest_inv_col in dest_inv_cols:
            dest_class_col = class_cols[dest_inv_col]
            exist_d = int(row[dest_inv_col])
            clas_d = str(row[dest_class_col]).strip()

            # Destino debe ser A/B
            if clas_d not in ["A", "B"]:
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
                }
            )

    if not registros:
        return pd.DataFrame()

    df_sug = pd.DataFrame(registros)
    df_sug = df_sug.sort_values(
        by=["Existencia origen", "Clave"], ascending=[False, True]
    )
    return df_sug


def get_base_c_sinmov(
    data,
    origin_inv_col: str,
    class_cols,
):
    """
    Devuelve TODOS los SKU del almacén origen que están en:
    - Clasificación C o Sin movimiento
    - Existencia > 0
    """
    origin_class_col = class_cols[origin_inv_col]
    df = data.copy()
    df["Existencia_origen"] = pd.to_numeric(
        df[origin_inv_col], errors="coerce"
    ).fillna(0).astype(int)
    df["Clasif_origen"] = df[origin_class_col].astype(str).str.strip()

    mask_c = df["Clasif_origen"] == "C"
    mask_sin = df["Clasif_origen"].str.lower().str.contains("sin")

    df = df[(df["Existencia_origen"] > 0) & (mask_c | mask_sin)]

    df = df[["Codigo", "Clave", "Descripcion", "Existencia_origen", "Clasif_origen"]]
    df = df.sort_values(by=["Clasif_origen", "Clave"])
    return df


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
            # Tablas de sugeridos (normales e inversos)
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
st.title("🏭 Balance de Inventario - El Cedro")

# ---------------------------------------------------------------------
# BOTÓN DE AYUDA (explicación detallada)
# ---------------------------------------------------------------------
with st.expander("❓", expanded=False):
    st.markdown("""
### 🧠 ¿Qué hace esta herramienta?

Esta página analiza automáticamente el archivo **Balance.csv**, que contiene todos los SKU con su existencia y clasificación en cada sucursal.  
Con esa información, la herramienta puede:

✔️ Contar SKU por clasificación (A, B, C y Sin movimiento)  
✔️ Detectar dónde hay faltantes importantes (A/B en 0)  
✔️ Ver todos los SKU de cada almacén  
✔️ Mostrar productos lentos (C / Sin mov)  
✔️ Sugerir movimientos entre almacenes  
✔️ Descargar Excel completos para trabajo operativo

---

## 🧭 **1. Selección de almacenes**

### 🏬 **Almacén que estoy revisando (origen)**  
Aquí eliges *desde qué sucursal quieres analizar el inventario*.  
Ejemplo: “Estoy revisando Adelitas”.

### 🔁 **Almacenes con los que puedo mover mercancía**  
Estos son los almacenes que **sí están disponibles para enviar o recibir mercancía**.  
Se usan para generar sugerencias.

---

## 📊 **2. Tarjetas amarillas (KPI)**

Resumen de inventario del almacén seleccionado.

| KPI | ¿Qué significa? |
|-----|------------------|
| SKU en origen | Total de productos con clasificación en ese almacén |
| SKU activos | A + B + C (todos) + Sin mov (solo > 0) |
| SKU (A/B en 0) | Productos importantes agotados en origen |
| Clasificación A | Total de SKU A (independiente de su existencia) |
| Clasificación B | Total de SKU B (independiente de su existencia) |
| Clasificación C | Total de SKU C (independiente de su existencia) |
| Sin movimiento | Solo los SKU con existencia **> 0** |

👉 Los SKU con existencia **0 o negativa** aún se cuentan como A/B/C, porque siguen clasificados, aunque estén agotados.

---

## 🔍 **3. Buscador de artículos**

Sirve para buscar por:

- Código  
- Clave  
- Descripción  

Si no escribes nada, se muestra una tabla completa.

#### 🟦 Colores por clasificación:

| Color | Significado |
|-------|-------------|
| 🟢 Verde | A o B con existencia > 0 |
| 🔵 Azul | A o B con existencia = 0 |
| 🟠 Naranja | C con existencia > 0 |
| 🔴 Rojo | Sin movimiento con existencia > 0 |

---

## ♻️ **4. Lógica del archivo de sugeridos**

### ⚠️ Importante:
**Los KPI NO coinciden con la tabla de sugeridos — y está bien.**  
No están midiendo lo mismo.

| KPI | Tabla de sugeridos (CSV) |
|-----|---------------------------|
| Cuenta TODOS los SKU A/B | Solo los A/B que tienen destino C/Sin mov |
| Un SKU aparece 1 vez | Puede aparecer varias veces si tiene varios destinos |
| No depende de destino | Depende del almacén destino y su clasificación |

#### 🧠 Ejemplo real:
Un SKU A en Adelitas puede aparecer así en el Excel de sugeridos:

| Clave | Origen    | Destino     |
|-------|-----------|-------------|
| TRU-123 | Adelitas | Matriz      |
| TRU-123 | Adelitas | San Agustín |
| TRU-123 | Adelitas | Berriozabal |

➡️ En los KPI cuenta **1 SKU A**  
➡️ ¡Pero en el Excel aparecen **3 filas**!

---

## 📦 **5. ¿Cómo genera la tabla de sugeridos?**

### 👍 **Sugeridos normales (reposición al origen):**
| Origen | Destino |
|--------|---------|
| A o B (independiente de su existencia) | C o Sin movimiento con existencia > 0 |

**SI CUMPLE ESO → aparece en el Excel.**  
Si no tiene destino válido → NO aparece.

---

## 📥 **6. ¿Qué pasa al descargar el Excel?**

El Excel SIEMPRE contiene **TODOS los SKU filtrados**, aunque en pantalla veas solo una parte.  
Esto permite trabajar bien desde Excel sin perder información.

---

## 🧾 **7. Consejos de uso**

✔ Puedes filtrar en Excel por “Almacén destino”  
✔ Puedes agrupar por “Clave” para saber cuántas sucursales podrían enviar  
✔ Puedes hacer un inventario de “pendientes por solicitar”  

📌 **RECOMENDADO:**  
Al abrir en Excel → Menú **Datos > Quitar duplicados** → Para ver cuántos SKU únicos tienes por categoría.

---

Si tienes dudas, vuelve a abrir esta ayuda.  
¡Gracias por usar el sistema! 🚀
""")

# ---------------------------------------------------------------------
# CONTROLES (ORIGEN + DESTINOS EN LA MISMA FILA)
# ---------------------------------------------------------------------
umbral_bajos_ab = 0  # dummy para firma de funciones

col_f1, col_f2 = st.columns([1.3, 2.0])

with col_f1:
    origin_options = ["Todos"] + warehouses
    default_origin = "Matriz" if "Matriz" in warehouses else origin_options[1]
    origin_index = origin_options.index(default_origin)

    origen_sel = st.selectbox(
        "Almacén que estoy revisando (origen)",
        origin_options,
        index=origin_index,
    )

with col_f2:
    if origen_sel == "Todos":
        opciones_destino = warehouses
    else:
        opciones_destino = [w for w in warehouses if w != origen_sel]

    destinos_sel = st.multiselect(
        "Almacenes de donde puedo solicitar / enviar",
        opciones_destino,
        default=opciones_destino,
    )

# ---------------------------------------------------------------------
# PANEL INVENTARIO
# ---------------------------------------------------------------------
st.markdown("---")
st.header("📊 KPI´s")

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

# KPIs
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
            <div class="metric-label">SKU (A/B con existencia 0)</div>
            <div class="metric-value">{format_int(kpis_to_show['bajos_ab'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Clasificación SKU A</div>
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
# TABS
# ---------------------------------------------------------------------
tab_buscar, tab_sugeridos, tab_inversos = st.tabs(
    ["🔎 Buscador", "📦 Sugeridos de traslado", "♻️ Sugeridos inversos"]
)

# =====================================================================
# TAB BUSCADOR
# =====================================================================
with tab_buscar:
    st.markdown("#### Buscador de artículos")
    busqueda = st.text_input(
        "Buscar por Código, Clave o Descripción (máximo 500 filas en pantalla):",
        value="",
    )

    df_buscar = data.copy()

    if busqueda.strip():
        txt = busqueda.strip().lower()
        mask = (
            df_buscar["Clave"].astype(str).str.lower().str.contains(txt)
            | df_buscar["Descripcion"].astype(str).str.lower().str.contains(txt)
            | df_buscar["Codigo"].astype(str).str.lower().str.contains(txt)
        )
        df_buscar = df_buscar[mask]

    # Limitar a 500 filas en pantalla para que sea más rápido
    total_rows = len(df_buscar)
    view_limit = 500
    if total_rows > view_limit:
        st.markdown(
            f"Se encontraron **{total_rows} artículos**. "
            f"Mostrando solo los primeros **{view_limit}** para mejorar el rendimiento."
        )
        df_view = df_buscar.head(view_limit)
    else:
        df_view = df_buscar

    cols_show = ["Codigo", "Clave", "Descripcion"]
    for inv in inv_cols:
        cols_show.append(inv)
        cls = class_cols[inv]
        if cls is not None:
            cols_show.append(cls)

    df_view = df_view[cols_show].copy()

    for col in inv_cols:
        df_view[col] = pd.to_numeric(
            df_view[col], errors="coerce"
        ).fillna(0).astype(int)

    if len(df_view) <= 1500:
        styler_buscar = style_class_colors(
            df_view, pairs_for_buscar=[(c, class_cols[c]) for c in inv_cols]
        )
        styler_buscar = styler_buscar.format(format_int, subset=inv_cols)
        st.dataframe(styler_buscar, use_container_width=True, height=480, hide_index=True)
    else:
        df_buscar_display = df_view.copy()
        for col in inv_cols:
            df_buscar_display[col] = df_buscar_display[col].apply(format_int)

        st.info("Tabla muy grande: se muestra sin colores para optimizar el rendimiento.")
        st.dataframe(df_buscar_display, use_container_width=True, height=480, hide_index=True)

# =====================================================================
# TAB SUGERIDOS
# =====================================================================
with tab_sugeridos:
    st.markdown("#### Sugeridos de traslado (reposiciones hacia el origen)")

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
            total_rows = len(df_sug)
            view_limit = 500
            if total_rows > view_limit:
                st.markdown(
                    f"Se encontraron **{total_rows} artículos**. "
                    f"Mostrando solo los primeros **{view_limit}** en la tabla para que cargue más rápido."
                )
                df_view = df_sug.head(view_limit)
            else:
                st.markdown(f"Se muestran **{total_rows} artículos**.")
                df_view = df_sug

            # CSV SIEMPRE CON TODAS LAS FILAS
            csv_sug = df_sug.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Descargar sugeridos (CSV - tabla completa)",
                data=csv_sug,
                file_name=f"sugeridos_{origin_inv_col}.csv",
                mime="text/csv",
            )

            styler_sug = style_class_colors(df_view)
            styler_sug = styler_sug.format(
                format_int,
                subset=[
                    "Existencia origen",
                    "Existencia destino",
                ],
            )
            st.dataframe(styler_sug, use_container_width=True, height=520, hide_index=True)

# =====================================================================
# TAB SUGERIDOS INVERSOS
# =====================================================================
with tab_inversos:
    st.markdown("#### Sugeridos inversos (salidas desde C / Sin movimiento hacia A/B)")

    if origen_sel == "Todos":
        st.info("Selecciona un **almacén específico** como origen para ver sugerencias inversas.")
    elif not destinos_sel:
        st.info("Selecciona al menos **un almacén destino** para calcular sugerencias inversas.")
    else:
        origin_inv_col = origen_sel

        df_base_cs = get_base_c_sinmov(
            data,
            origin_inv_col=origin_inv_col,
            class_cols=class_cols,
        )

        df_inv = build_reverse_suggestions(
            data,
            inv_cols,
            class_cols,
            origin_inv_col=origin_inv_col,
            dest_inv_cols=destinos_sel,
            umbral_bajos_ab=umbral_bajos_ab,
        )

        if df_base_cs.empty:
            st.warning(
                f"No hay SKU en clasificación C o Sin movimiento con existencia > 0 en **{origin_inv_col}**."
            )
        else:
            # CSV con TODOS los C / Sin mov > 0 del origen
            csv_base = df_base_cs.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Descargar TODOS los SKU C / Sin mov del origen (CSV)",
                data=csv_base,
                file_name=f"c_sinmov_{origin_inv_col}.csv",
                mime="text/csv",
            )

            if df_inv.empty:
                st.info(
                    "No se encontraron destinos A/B para proponer traslados desde estos C / Sin movimiento."
                )
            else:
                total_rows = len(df_inv)
                view_limit = 500
                if total_rows > view_limit:
                    st.markdown(
                        f"Se encontraron **{total_rows} artículos con sugerencia inversa**. "
                        f"Mostrando solo los primeros **{view_limit}** en la tabla."
                    )
                    df_view = df_inv.head(view_limit)
                else:
                    st.markdown(
                        f"Se muestran **{total_rows} artículos con sugerencia inversa**."
                    )
                    df_view = df_inv

                # CSV con TODAS las sugerencias inversas
                csv_inv = df_inv.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "⬇️ Descargar sugeridos inversos (CSV - tabla completa)",
                    data=csv_inv,
                    file_name=f"sugeridos_inversos_{origin_inv_col}.csv",
                    mime="text/csv",
                )

                styler_inv = style_class_colors(df_view)
                styler_inv = styler_inv.format(
                    format_int,
                    subset=[
                        "Existencia origen",
                        "Existencia destino",
                    ],
                )
                st.dataframe(
                    styler_inv,
                    use_container_width=True,
                    height=520,
                    hide_index=True,
                )
