import requests
import pandas as pd
import time
import re

# ─── 1. Descarga y construcción por versión ───────────────────────────────────
VERSIONES_URLS = {
    "H0": "https://comtradeapi.un.org/files/v1/app/reference/H0.json",
    "H1": "https://comtradeapi.un.org/files/v1/app/reference/H1.json",
    "H2": "https://comtradeapi.un.org/files/v1/app/reference/H2.json",
    "H3": "https://comtradeapi.un.org/files/v1/app/reference/H3.json",
    "H4": "https://comtradeapi.un.org/files/v1/app/reference/H4.json",
    "H5": "https://comtradeapi.un.org/files/v1/app/reference/H5.json",
    "H6": "https://comtradeapi.un.org/files/v1/app/reference/H6.json",
}

def limpiar_texto(texto, codigo):
    texto = re.sub(rf"^{re.escape(str(codigo))}\s*[-–]+\s*", "", texto).strip()
    texto = re.sub(r"^[-–\s]+", "", texto).strip()
    return texto

def construir_df_version(resultados, version):
    indice = {item["id"]: item for item in resultados}
    rows = []
    for item in resultados:
        codigo = str(item["id"]).strip()
        if not codigo.isdigit() or len(codigo) != 6:
            continue
        texto_propio  = limpiar_texto(item["text"], codigo)
        parent_id     = item.get("parent", "")
        partida_item  = indice.get(parent_id, {})
        partida_id    = str(partida_item.get("id", parent_id))
        desc_partida  = limpiar_texto(partida_item.get("text", ""), partida_id)
        capitulo_id   = str(partida_item.get("parent", ""))
        capitulo_item = indice.get(capitulo_id, {})
        desc_capitulo = limpiar_texto(capitulo_item.get("text", ""), capitulo_id)
        desc_completa = f"{desc_partida} | {texto_propio}" if desc_partida and desc_partida != texto_propio else texto_propio

        rows.append({
            "codigo_hs":       codigo,
            "capitulo":        codigo[:2],
            "partida":         codigo[:4],
            "desc_capitulo":   desc_capitulo,
            "desc_partida":    desc_partida,
            "desc_especifica": texto_propio,
            "desc_completa":   desc_completa,
        })
    return pd.DataFrame(rows)

def descargar_version(version, url):
    print(f"Descargando {version}...", end=" ")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    resultados = r.json().get("results", [])
    df = construir_df_version(resultados, version)
    print(f"→ {len(df):,} subpartidas")
    time.sleep(0.5)
    return df

# ─── 2. Descargar las 7 tablas ────────────────────────────────────────────────
print("=== Descargando 7 versiones HS ===\n")

hs_H0 = descargar_version("H0", VERSIONES_URLS["H0"])
hs_H1 = descargar_version("H1", VERSIONES_URLS["H1"])
hs_H2 = descargar_version("H2", VERSIONES_URLS["H2"])
hs_H3 = descargar_version("H3", VERSIONES_URLS["H3"])
hs_H4 = descargar_version("H4", VERSIONES_URLS["H4"])
hs_H5 = descargar_version("H5", VERSIONES_URLS["H5"])
hs_H6 = descargar_version("H6", VERSIONES_URLS["H6"])

dfs = {"H0": hs_H0, "H1": hs_H1, "H2": hs_H2, "H3": hs_H3,
       "H4": hs_H4, "H5": hs_H5, "H6": hs_H6}



# ─── 3. Tabla de cambios ──────────────────────────────────────────────────────
def construir_tabla_cambios(dfs):
    """
    Para cada código HS detecta:
    - En qué versiones existe
    - Si la descripción cambió entre versiones consecutivas
    - Tipo de evento: NUEVO, ELIMINADO, MODIFICADO, SIN_CAMBIO
    """
    orden = ["H0","H1","H2","H3","H4","H5","H6"]

    # Pivot: código × versión → desc_completa
    registros = []
    for v, df in dfs.items():
        for _, row in df.iterrows():
            registros.append({
                "codigo_hs": row["codigo_hs"],
                "version":   v,
                "desc":      row["desc_completa"],
                "capitulo":  row["capitulo"],
                "partida":   row["partida"],
            })

    df_largo = pd.DataFrame(registros)
    pivot = df_largo.pivot_table(
        index=["codigo_hs","capitulo","partida"],
        columns="version",
        values="desc",
        aggfunc="first"
    ).reset_index()
    pivot.columns.name = None

    # Detectar cambios entre versiones consecutivas
    cambios = []
    pares = list(zip(orden[:-1], orden[1:]))  # (H0,H1),(H1,H2),...,(H5,H6)

    for _, row in pivot.iterrows():
        codigo = row["codigo_hs"]
        for v_ant, v_act in pares:
            desc_ant = row.get(v_ant, None)
            desc_act = row.get(v_act, None)

            tiene_ant = pd.notna(desc_ant)
            tiene_act = pd.notna(desc_act)

            if not tiene_ant and tiene_act:
                evento = "NUEVO"
            elif tiene_ant and not tiene_act:
                evento = "ELIMINADO"
            elif tiene_ant and tiene_act and desc_ant != desc_act:
                evento = "MODIFICADO"
            elif tiene_ant and tiene_act and desc_ant == desc_act:
                evento = "SIN_CAMBIO"
            else:
                continue  # no existía en ninguna de las dos

            if evento != "SIN_CAMBIO":  # solo guardar cambios reales
                cambios.append({
                    "codigo_hs":  codigo,
                    "capitulo":   row["capitulo"],
                    "partida":    row["partida"],
                    "de_version": v_ant,
                    "a_version":  v_act,
                    "evento":     evento,
                })

    df_cambios = pd.DataFrame(cambios)
    return df_cambios, pivot

print("\n=== Construyendo tabla de cambios ===")
hs_cambios, hs_pivot = construir_tabla_cambios(dfs)

# ─── 5. Guardar todo ──────────────────────────────────────────────────────────
import pyarrow as pa
import pyarrow.parquet as pq

print("\n=== Guardando archivos ===")

str_cols_hs = ["codigo_hs", "capitulo", "partida", "desc_capitulo", "desc_partida", "desc_especifica", "desc_completa"]

for v, df in dfs.items():
    fname = f"hs_{v}.parquet"
    df_out = df.copy()
    df_out["codigo_hs"] = df_out["codigo_hs"].astype(str).str.zfill(6)
    df_out["capitulo"]  = df_out["codigo_hs"].str[:2]
    df_out["partida"]   = df_out["codigo_hs"].str[:4]
    schema = pa.schema([(c, pa.string()) for c in str_cols_hs])
    pq.write_table(pa.Table.from_pandas(df_out[str_cols_hs], schema=schema, preserve_index=False), fname)
    print(f"  {fname} → {len(df_out):,} filas")

# hs_cambios
str_cols_cambios = ["codigo_hs", "capitulo", "partida", "de_version", "a_version", "evento"]
hs_cambios_out = hs_cambios.copy()
hs_cambios_out["codigo_hs"] = hs_cambios_out["codigo_hs"].astype(str).str.zfill(6)
hs_cambios_out["capitulo"]  = hs_cambios_out["codigo_hs"].str[:2]
hs_cambios_out["partida"]   = hs_cambios_out["codigo_hs"].str[:4]
schema_cambios = pa.schema([(c, pa.string()) for c in str_cols_cambios])
pq.write_table(pa.Table.from_pandas(hs_cambios_out[str_cols_cambios], schema=schema_cambios, preserve_index=False), "hs_cambios.parquet")
print(f"  hs_cambios.parquet → {len(hs_cambios_out):,} filas")

# hs_pivot (codigo_hs, capitulo, partida + columnas H0..H6 como str)
hs_pivot_out = hs_pivot.copy()
hs_pivot_out["codigo_hs"] = hs_pivot_out["codigo_hs"].astype(str).str.zfill(6)
hs_pivot_out["capitulo"]  = hs_pivot_out["codigo_hs"].str[:2]
hs_pivot_out["partida"]   = hs_pivot_out["codigo_hs"].str[:4]
version_cols = [c for c in hs_pivot_out.columns if c in ["H0","H1","H2","H3","H4","H5","H6"]]
str_cols_pivot = ["codigo_hs", "capitulo", "partida"] + version_cols
for c in version_cols:
    hs_pivot_out[c] = hs_pivot_out[c].astype(str).replace("nan", None)
schema_pivot = pa.schema([(c, pa.string()) for c in str_cols_pivot])
pq.write_table(pa.Table.from_pandas(hs_pivot_out[str_cols_pivot], schema=schema_pivot, preserve_index=False), "hs_pivot.parquet")
print(f"  hs_pivot.parquet   → {len(hs_pivot_out):,} filas")