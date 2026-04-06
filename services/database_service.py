import requests as http_requests
from config import TURSO_DATABASE_URL, TURSO_AUTH_TOKEN


def _get_http_url():
    """Convert libsql:// URL to https:// for the HTTP API."""
    url = TURSO_DATABASE_URL
    if url.startswith("libsql://"):
        url = url.replace("libsql://", "https://")
    return url


def _execute_sql(sql: str, args: list = None):
    """Execute a SQL statement via Turso HTTP API."""
    url = _get_http_url()
    headers = {
        "Authorization": f"Bearer {TURSO_AUTH_TOKEN}",
        "Content-Type": "application/json",
    }

    # Build the request body for Turso HTTP API v2 (pipeline)
    request_body = {
        "requests": [
            {"type": "execute", "stmt": {"sql": sql, "args": _format_args(args)}},
            {"type": "close"},
        ]
    }

    response = http_requests.post(
        f"{url}/v2/pipeline", headers=headers, json=request_body
    )

    if response.status_code != 200:
        raise Exception(f"Turso API error: {response.status_code} - {response.text}")

    data = response.json()
    results = data.get("results", [])

    if results and results[0].get("type") == "error":
        error = results[0].get("error", {})
        raise Exception(f"SQL error: {error.get('message', 'Unknown error')}")

    return results[0] if results else None


def _format_args(args):
    """Format arguments for Turso HTTP API."""
    if not args:
        return []
    formatted = []
    for arg in args:
        if arg is None:
            formatted.append({"type": "null", "value": None})
        elif isinstance(arg, int):
            formatted.append({"type": "integer", "value": str(arg)})
        elif isinstance(arg, float):
            formatted.append({"type": "float", "value": str(arg)})
        else:
            formatted.append({"type": "text", "value": str(arg)})
    return formatted


def init_db():
    """Initialize the oportunidades table if it doesn't exist."""
    _execute_sql("""
        CREATE TABLE IF NOT EXISTS oportunidades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            numero_doc TEXT NOT NULL,
            nombre_oportunidad TEXT NOT NULL,
            proponentes_cargos TEXT NOT NULL,
            fecha TEXT NOT NULL,
            origen_oportunidad TEXT NOT NULL,
            situacion_actual TEXT NOT NULL,
            hallazgos TEXT NOT NULL,
            fuente_hallazgos TEXT NOT NULL,
            publico_objetivo TEXT NOT NULL,
            problema_principal TEXT NOT NULL,
            sub_problemas TEXT NOT NULL,
            impacto_esperado TEXT NOT NULL,
            importancia_problema TEXT NOT NULL,
            anexos TEXT DEFAULT '',
            documento_url TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("[DB] Tabla 'oportunidades' inicializada correctamente.")


def insertar_oportunidad(datos: dict) -> None:
    """Insert a complete oportunidad record into the database."""
    _execute_sql(
        """
        INSERT INTO oportunidades (
            numero_doc, nombre_oportunidad, proponentes_cargos, fecha,
            origen_oportunidad, situacion_actual, hallazgos, fuente_hallazgos,
            publico_objetivo, problema_principal, sub_problemas,
            impacto_esperado, importancia_problema, anexos, documento_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            datos.get("numero_doc", ""),
            datos.get("nombre_oportunidad", ""),
            datos.get("proponentes_cargos", ""),
            datos.get("fecha", ""),
            datos.get("origen_oportunidad", ""),
            datos.get("situacion_actual", ""),
            datos.get("hallazgos", ""),
            datos.get("fuente_hallazgos", ""),
            datos.get("publico_objetivo", ""),
            datos.get("problema_principal", ""),
            datos.get("sub_problemas", ""),
            datos.get("impacto_esperado", ""),
            datos.get("importancia_problema", ""),
            datos.get("anexos", ""),
            datos.get("documento_url", ""),
        ],
    )
    print(f"[DB] Oportunidad insertada: {datos.get('nombre_oportunidad')}")


def obtener_oportunidades() -> list:
    """Fetch all oportunidades from the database."""
    result = _execute_sql("SELECT * FROM oportunidades ORDER BY id DESC")

    if not result or result.get("type") != "ok":
        return []

    response = result.get("response", {}).get("result", {})
    cols = [col.get("name") for col in response.get("cols", [])]
    rows_data = response.get("rows", [])

    records = []
    for row in rows_data:
        values = [cell.get("value") for cell in row]
        records.append(dict(zip(cols, values)))

    return records
