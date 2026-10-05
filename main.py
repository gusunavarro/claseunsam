import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shutil
from pathlib import Path
from datetime import date, datetime

# ==========================================================
# CONFIGURACIÓN DE CARPETAS
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
BACKUP_DIR = DATA_DIR / "backups"
BACKUP_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "stock.csv"

COLUMNS = [
    "id", "fecha", "lote", "etapa", "sustrato_kg", "bolsas",
    "peso_cosechado_kg", "precio_kg", "cliente", "notas"
]

EC5_META_COLS = {
    "ec5_uuid", "ec5_parent_uuid", "created_at", "uploaded_at",
    "created_by", "uploaded_by", "title", "ec5_branch_owner_uuid",
    "ec5_branch_uuid"
}


# ==========================================================
# FUNCIONES DE STOCK
# ==========================================================
def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        df = pd.DataFrame(columns=COLUMNS)
        df["fecha"] = pd.to_datetime(df["fecha"])
        return df
    return pd.read_csv(DATA_FILE, parse_dates=["fecha"])


def _atomic_save(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _backup() -> None:
    if not DATA_FILE.exists():
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(DATA_FILE, BACKUP_DIR / f"stock_{ts}.csv")


def save_data(df: pd.DataFrame, backup: bool = True) -> None:
    if backup:
        _backup()
    _atomic_save(df, DATA_FILE)


def add_record(df: pd.DataFrame, record: dict) -> pd.DataFrame:
    record["id"] = int(df["id"].max() + 1) if not df.empty and pd.notna(df["id"].max()) else 1
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    save_data(df)
    return df


def update_record(df: pd.DataFrame, record_id: int, updates: dict) -> pd.DataFrame:
    idx = df.index[df["id"] == record_id]
    if len(idx) == 0:
        return df
    for k, v in updates.items():
        df.loc[idx, k] = v
    save_data(df)
    return df


def delete_record(df: pd.DataFrame, record_id: int) -> pd.DataFrame:
    df = df[df["id"] != record_id].reset_index(drop=True)
    save_data(df)
    return df


def kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total_kg": 0, "eficiencia": 0, "ingresos": 0, "lotes_activos": 0}
    cosechas = df[df["etapa"] == "Cosecha"]
    total_kg = cosechas["peso_cosechado_kg"].sum()
    ingresos = (cosechas["peso_cosechado_kg"] * cosechas["precio_kg"]).sum()
    sustrato = cosechas["sustrato_kg"].sum()
    eficiencia = (total_kg / sustrato * 100) if sustrato > 0 else 0
    return {
        "total_kg": round(total_kg, 2),
        "eficiencia": round(eficiencia, 2),
        "ingresos": round(ingresos, 2),
        "lotes_activos": df[df["etapa"].isin(
            ["Inoculación", "Incubación", "Fructificación"]
        )]["lote"].nunique(),
    }


# ==========================================================
# GRÁFICOS DE STOCK
# ==========================================================
def chart_cosecha_tiempo(df):
    d = df[df["etapa"] == "Cosecha"].sort_values("fecha")
    if d.empty:
        return None
    return px.line(d, x="fecha", y="peso_cosechado_kg", markers=True,
                   title="Cosecha en el tiempo (kg)",
                   labels={"peso_cosechado_kg": "Kg cosechados", "fecha": "Fecha"})


def chart_etapas(df):
    if df.empty:
        return None
    conteo = df["etapa"].value_counts().reset_index()
    conteo.columns = ["etapa", "cantidad"]
    return px.pie(conteo, names="etapa", values="cantidad",
                  title="Distribución por etapa", hole=0.4)


def chart_eficiencia_por_lote(df):
    d = df[df["etapa"] == "Cosecha"].copy()
    if d.empty:
        return None
    agg = d.groupby("lote").agg(
        sustrato=("sustrato_kg", "sum"),
        cosecha=("peso_cosechado_kg", "sum")
    ).reset_index()
    agg["eficiencia_%"] = (agg["cosecha"] / agg["sustrato"] * 100).round(2)
    return px.bar(agg, x="lote", y="eficiencia_%",
                  title="Eficiencia biológica por lote (BE %)",
                  color="eficiencia_%", color_continuous_scale="Greens")


def chart_ingresos_mes(df):
    d = df[df["etapa"] == "Cosecha"].copy()
    if d.empty:
        return None
    d["mes"] = d["fecha"].dt.to_period("M").astype(str)
    d["ingreso"] = d["peso_cosechado_kg"] * d["precio_kg"]
    agg = d.groupby("mes")["ingreso"].sum().reset_index()
    return px.bar(agg, x="mes", y="ingreso", title="Ingresos por mes ($)")


# ==========================================================
# FUNCIONES DE ENCUESTAS
# ==========================================================
def load_ec5_csv(uploaded_file) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        for sep in (",", ";", "\t"):
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc, sep=sep)
                if df.shape[1] > 1:
                    return df
            except Exception as e:
                last_err = e
    raise ValueError(f"No se pudo leer el CSV. Último error: {last_err}")


def save_upload(uploaded_file, df: pd.DataFrame) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = Path(uploaded_file.name).stem.replace(" ", "_")
    path = UPLOAD_DIR / f"{safe_name}_{ts}.csv"
    df.to_csv(path, index=False)
    return path


def list_uploads():
    return sorted(UPLOAD_DIR.glob("*.csv"), reverse=True)


def clean_ec5(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["created_at", "uploaded_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.replace(
        to_replace=["", "NA", "N/A", "null", "None", "none", "-", " "],
        value=np.nan
    )
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            non_null = df[col].notna().sum()
            if non_null > 0 and converted.notna().sum() / non_null > 0.7:
                df[col] = converted
    return df


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.replace(".", "_").strip() for c in df.columns]
    return df


def classify_columns(df: pd.DataFrame) -> dict:
    meta = set(c for c in df.columns if c.lower().strip() in EC5_META_COLS)
    numericas, categoricas, fechas, texto = [], [], [], []
    for col in df.columns:
        if col in meta:
            continue
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            fechas.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            if s.nunique(dropna=True) <= 10:
                categoricas.append(col)
            else:
                numericas.append(col)
        else:
            nunique = s.nunique(dropna=True)
            total = s.notna().sum()
            if total > 0 and nunique / total > 0.5 and nunique > 15:
                texto.append(col)
            else:
                categoricas.append(col)
    return {"numericas": numericas, "categoricas": categoricas,
            "fechas": fechas, "texto": texto}


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "columna": col,
            "tipo": str(s.dtype),
            "no_nulos": int(s.notna().sum()),
            "%_nulos": round(s.isna().mean() * 100, 1),
            "únicos": int(s.nunique(dropna=True)),
            "ejemplo": str(s.dropna().iloc[0])[:40] if s.notna().any() else "",
        })
    return pd.DataFrame(rows)


def bar_categorical(df, col, top=15):
    counts = df[col].value_counts(dropna=False).head(top).reset_index()
    counts.columns = [col, "cantidad"]
    fig = px.bar(counts, x="cantidad", y=col, orientation="h",
                 title=f"Distribución de «{col}» (top {top})", text="cantidad")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def pie_categorical(df, col, top=10):
    counts = df[col].value_counts(dropna=False).head(top).reset_index()
    counts.columns = [col, "cantidad"]
    return px.pie(counts, names=col, values="cantidad",
                  title=f"Proporción de «{col}»", hole=0.4)


def histogram_numeric(df, col, bins=20):
    return px.histogram(df, x=col, nbins=bins,
                        title=f"Distribución de «{col}»", marginal="box")


def box_by_category(df, num_col, cat_col, top=10):
    top_cats = df[cat_col].value_counts().head(top).index
    d = df[df[cat_col].isin(top_cats)]
    fig = px.box(d, x=cat_col, y=num_col, color=cat_col,
                 title=f"«{num_col}» por «{cat_col}»")
    fig.update_layout(showlegend=False)
    return fig


def timeline(df, date_col, freq="D"):
    d = df.dropna(subset=[date_col]).copy()
    d["periodo"] = d[date_col].dt.to_period(freq).astype(str)
    agg = d.groupby("periodo").size().reset_index(name="registros")
    return px.line(agg, x="periodo", y="registros", markers=True,
                   title=f"Registros en el tiempo ({freq})")


def crosstab_heatmap(df, row_col, col_col, top=10):
    r = df[row_col].value_counts().head(top).index
    c = df[col_col].value_counts().head(top).index
    ct = pd.crosstab(df[row_col], df[col_col]).loc[r, c]
    return px.imshow(ct, text_auto=True, aspect="auto",
                     title=f"Cruce: {row_col} × {col_col}",
                     color_continuous_scale="Blues")


# ==========================================================
# APP
# ==========================================================
st.set_page_config(page_title="Stock Pleurotus + Encuestas",
                   page_icon="🍄", layout="wide")
st.title("🍄 Gestión de Stock · Cultivo de Pleurotus")

if "df" not in st.session_state:
    st.session_state.df = load_data()

df = st.session_state.df

# ---- Sidebar ----
with st.sidebar:
    st.header("➕ Nuevo registro de stock")
    with st.form("form_alta", clear_on_submit=True):
        fecha = st.date_input("Fecha", value=date.today())
        lote = st.text_input("Lote", placeholder="L-2025-01")
        etapa = st.selectbox("Etapa",
            ["Inoculación", "Incubación", "Fructificación", "Cosecha", "Descarte"])
        sustrato = st.number_input("Sustrato (kg)", min_value=0.0, step=0.5)
        bolsas = st.number_input("Bolsas", min_value=0, step=1)
        peso = st.number_input("Peso cosechado (kg)", min_value=0.0, step=0.1)
        precio = st.number_input("Precio por kg ($)", min_value=0.0, step=10.0)
        cliente = st.text_input("Cliente")
        notas = st.text_area("Notas")

        if st.form_submit_button("Guardar", use_container_width=True):
            st.session_state.df = add_record(st.session_state.df, {
                "fecha": fecha, "lote": lote, "etapa": etapa,
                "sustrato_kg": sustrato, "bolsas": bolsas,
                "peso_cosechado_kg": peso, "precio_kg": precio,
                "cliente": cliente, "notas": notas,
            })
            st.success("Registro guardado ✅")
            st.rerun()

df = st.session_state.df

# ---- KPIs ----
k = kpis(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("🍄 Total cosechado", f"{k['total_kg']} kg")
c2.metric("📈 Eficiencia biológica", f"{k['eficiencia']} %")
c3.metric("💰 Ingresos", f"${k['ingresos']:,}")
c4.metric("🧺 Lotes activos", k["lotes_activos"])

st.divider()

# ---- Tabs ----
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dashboard", "📋 Datos", "✏️ Administrar", "📝 Encuestas"])

# ============ TAB 1: DASHBOARD ============
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fig = chart_cosecha_tiempo(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sin cosechas registradas aún.")
    with col2:
        fig = chart_etapas(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = chart_eficiencia_por_lote(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = chart_ingresos_mes(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

# ============ TAB 2: DATOS ============
with tab2:
    st.subheader("Registros")
    filtro = st.multiselect("Filtrar por etapa",
        options=df["etapa"].unique().tolist() if not df.empty else [],
        default=df["etapa"].unique().tolist() if not df.empty else [])
    df_show = df[df["etapa"].isin(filtro)] if filtro else df
    st.dataframe(df_show, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Descargar CSV",
        df_show.to_csv(index=False).encode("utf-8"),
        "stock_pleurotus.csv", "text/csv")

# ============ TAB 3: ADMINISTRAR ============
with tab3:
    st.subheader("✏️ Editar / 🗑️ Eliminar registros")
    if df.empty:
        st.info("No hay registros todavía.")
    else:
        id_sel = st.selectbox("Seleccionar ID",
            options=df["id"].tolist(),
            format_func=lambda i: (
                f"ID {i} — {df.loc[df['id']==i, 'lote'].values[0]} "
                f"| {df.loc[df['id']==i, 'etapa'].values[0]} "
                f"| {df.loc[df['id']==i, 'fecha'].dt.strftime('%Y-%m-%d').values[0]}"),
            key="edit_id")

        row = df[df["id"] == id_sel].iloc[0]
        etapas_opts = ["Inoculación", "Incubación", "Fructificación", "Cosecha", "Descarte"]

        with st.form("form_edit"):
            col1, col2 = st.columns(2)
            fecha = col1.date_input("Fecha", value=row["fecha"].date())
            lote = col2.text_input("Lote", value=row["lote"])
            etapa = col1.selectbox("Etapa", etapas_opts,
                index=etapas_opts.index(row["etapa"]) if row["etapa"] in etapas_opts else 0)
            sustrato = col2.number_input("Sustrato (kg)",
                value=float(row["sustrato_kg"]), min_value=0.0, step=0.5)
            bolsas = col1.number_input("Bolsas",
                value=int(row["bolsas"]), min_value=0, step=1)
            peso = col2.number_input("Peso cosechado (kg)",
                value=float(row["peso_cosechado_kg"]), min_value=0.0, step=0.1)
            precio = col1.number_input("Precio por kg ($)",
                value=float(row["precio_kg"]), min_value=0.0, step=10.0)
            cliente = col2.text_input("Cliente",
                value=str(row["cliente"]) if pd.notna(row["cliente"]) else "")
            notas = st.text_area("Notas",
                value=str(row["notas"]) if pd.notna(row["notas"]) else "")

            cc1, cc2 = st.columns(2)
            guardar = cc1.form_submit_button("💾 Guardar", use_container_width=True, type="primary")
            eliminar = cc2.form_submit_button("🗑️ Eliminar", use_container_width=True)

            if guardar:
                st.session_state.df = update_record(st.session_state.df, id_sel, {
                    "fecha": fecha, "lote": lote, "etapa": etapa,
                    "sustrato_kg": sustrato, "bolsas": bolsas,
                    "peso_cosechado_kg": peso, "precio_kg": precio,
                    "cliente": cliente, "notas": notas,
                })
                st.success("Actualizado ✅")
                st.rerun()
            if eliminar:
                st.session_state.df = delete_record(st.session_state.df, id_sel)
                st.success(f"ID {id_sel} eliminado")
                st.rerun()

    with st.expander("📦 Backups"):
        backups = sorted(BACKUP_DIR.glob("*"), reverse=True)
        if not backups:
            st.caption("Sin backups aún.")
        else:
            for b in backups[:10]:
                st.write(f"• `{b.name}` — {b.stat().st_size} bytes")

# ============ TAB 4: ENCUESTAS ============
with tab4:
    st.header("📝 Análisis de Encuestas (EpiCollect5)")
    uploaded = st.file_uploader("Subí el CSV de EpiCollect5", type=["csv"], key="ec5")

    previous = list_uploads()
    archivo_prev = "—"
    if previous:
        with st.expander("📂 Archivos ya subidos"):
            archivo_prev = st.selectbox("Elegir uno",
                options=["—"] + [p.name for p in previous], key="prev_sel")

    df_survey = None
    if uploaded is not None:
        try:
            df_raw = load_ec5_csv(uploaded)
            save_upload(uploaded, df_raw)
            df_survey = flatten_columns(clean_ec5(df_raw))
            st.success(f"✅ {df_raw.shape[0]} filas × {df_raw.shape[1]} columnas")
        except Exception as e:
            st.error(f"Error: {e}")
    elif previous and archivo_prev != "—":
        path = next(p for p in previous if p.name == archivo_prev)
        df_survey = flatten_columns(clean_ec5(pd.read_csv(path)))
        st.info(f"📄 {archivo_prev}")

    if df_survey is not None and not df_survey.empty:
        if "created_at" in df_survey.columns and df_survey["created_at"].notna().any():
            fmin = df_survey["created_at"].min().date()
            fmax = df_survey["created_at"].max().date()
            if fmin < fmax:
                rango = st.date_input("Rango de fechas", value=(fmin, fmax),
                    min_value=fmin, max_value=fmax, key="rango")
                if len(rango) == 2:
                    mask = ((df_survey["created_at"].dt.date >= rango[0]) &
                            (df_survey["created_at"].dt.date <= rango[1]))
                    df_survey = df_survey[mask]
                    st.caption(f"Registros: {len(df_survey)}")

        st.subheader("🔍 Resumen")
        st.dataframe(summary_stats(df_survey), use_container_width=True, hide_index=True)

        tipos = classify_columns(df_survey)

        with st.expander("👀 Datos crudos"):
            st.dataframe(df_survey.head(50), use_container_width=True)

        st.divider()
        modo = st.radio("Análisis",
            ["📊 Vista general", "📈 Variable única", "🔀 Cruce",
             "🕒 Serie temporal", "🗺️ Mapa"], horizontal=True, key="modo")

        if modo == "📊 Vista general":
            c1, c2 = st.columns(2)
            with c1:
                if "created_at" in df_survey.columns and df_survey["created_at"].notna().any():
                    fl = st.selectbox("Agrupación",
                        [("Diaria","D"),("Semanal","W"),("Mensual","M")],
                        format_func=lambda x: x[0], key="fv")
                    st.plotly_chart(timeline(df_survey, "created_at", fl[1]),
                                    use_container_width=True)
            with c2:
                for c in tipos["categoricas"][:3]:
                    st.plotly_chart(bar_categorical(df_survey, c, top=8),
                                    use_container_width=True)

        elif modo == "📈 Variable única":
            tv = st.selectbox("Tipo", ["Categórica", "Numérica", "Fecha"], key="tv")
            if tv == "Categórica" and tipos["categoricas"]:
                col = st.selectbox("Variable", tipos["categoricas"], key="cv")
                g = st.radio("Gráfico", ["Barras","Torta"], horizontal=True, key="gv")
                fig = bar_categorical(df_survey, col) if g == "Barras" else pie_categorical(df_survey, col)
                st.plotly_chart(fig, use_container_width=True)
            elif tv == "Numérica" and tipos["numericas"]:
                col = st.selectbox("Variable", tipos["numericas"], key="nv")
                b = st.slider("Bins", 5, 60, 20, key="bs")
                st.plotly_chart(histogram_numeric(df_survey, col, b), use_container_width=True)
            elif tv == "Fecha" and tipos["fechas"]:
                col = st.selectbox("Variable", tipos["fechas"], key="fv2")
                st.plotly_chart(timeline(df_survey, col), use_container_width=True)
            else:
                st.warning("No hay variables de ese tipo.")

        elif modo == "🔀 Cruce":
            tc = st.radio("Combinación", ["Cat × Cat","Cat × Num"], horizontal=True, key="tc")
            if tc == "Cat × Cat" and len(tipos["categoricas"]) >= 2:
                c1, c2 = st.columns(2)
                fi = c1.selectbox("Filas", tipos["categoricas"], key="cf")
                co = c2.selectbox("Columnas", tipos["categoricas"],
                    index=min(1, len(tipos["categoricas"])-1), key="cc")
                if fi != co:
                    st.plotly_chart(crosstab_heatmap(df_survey, fi, co), use_container_width=True)
                else:
                    st.info("Elegí variables distintas.")
            elif tc == "Cat × Num" and tipos["categoricas"] and tipos["numericas"]:
                c1, c2 = st.columns(2)
                cat = c1.selectbox("Categórica", tipos["categoricas"], key="xc")
                num = c2.selectbox("Numérica", tipos["numericas"], key="xn")
                st.plotly_chart(box_by_category(df_survey, num, cat), use_container_width=True)
            else:
                st.warning("Faltan variables.")

        elif modo == "🕒 Serie temporal":
            opc = list(tipos["fechas"])
            if "created_at" in df_survey.columns and "created_at" not in opc:
                opc.append("created_at")
            if not opc:
                st.warning("Sin columnas de fecha.")
            else:
                fc = st.selectbox("Fecha", opc, key="sfc")
                fl = st.selectbox("Agrupación",
                    [("Diaria","D"),("Semanal","W"),("Mensual","M")],
                    format_func=lambda x: x[0], key="sfq")
                st.plotly_chart(timeline(df_survey, fc, fl[1]), use_container_width=True)
                if tipos["categoricas"]:
                    cat = st.selectbox("Desglosar", ["—"] + tipos["categoricas"], key="sc")
                    if cat != "—":
                        d = df_survey.dropna(subset=[fc]).copy()
                        d["periodo"] = d[fc].dt.to_period(fl[1]).astype(str)
                        agg = d.groupby(["periodo", cat]).size().reset_index(name="n")
                        fig2 = px.line(agg, x="periodo", y="n", color=cat, markers=True)
                        st.plotly_chart(fig2, use_container_width=True)

        elif modo == "🗺️ Mapa":
            lat_c = [c for c in df_survey.columns if "latitude" in c.lower()]
            lon_c = [c for c in df_survey.columns if "longitude" in c.lower()]
            if not lat_c or not lon_c:
                st.info("Sin columnas de lat/lon.")
            else:
                d = df_survey.dropna(subset=[lat_c[0], lon_c[0]])
                if d.empty:
                    st.info("Sin coordenadas válidas.")
                else:
                    fig = px.scatter_mapbox(d, lat=lat_c[0], lon=lon_c[0],
                        zoom=10, height=500, mapbox_style="open-street-map")
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.download_button("⬇️ Descargar CSV limpio",
            df_survey.to_csv(index=False).encode("utf-8"),
            "encuestas_limpio.csv", "text/csv")
