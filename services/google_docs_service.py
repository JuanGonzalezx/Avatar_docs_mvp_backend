import os
import uuid
from datetime import datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from config import GOOGLE_DOCS_TEMPLATE_ID, GOOGLE_DRIVE_FOLDER_ID

SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
]

_docs_service = None
_drive_service = None


def _get_services():
    """Lazily initialize Google API services using OAuth 2.0 (user account)."""
    global _docs_service, _drive_service

    if _docs_service and _drive_service:
        return _docs_service, _drive_service

    creds = None
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    token_path = os.path.join(base_dir, "token.json")
    secret_path = os.path.join(base_dir, "client_secret.json")

    # 1. Try to load existing token
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    # 2. If no valid creds, refresh or re-authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("[DOCS] Refrescando token de Google...")
            creds.refresh(Request())
        else:
            print("[DOCS] Iniciando flujo de autenticación OAuth...")
            print("[DOCS] Se abrirá una ventana en tu navegador para iniciar sesión con Google.")
            flow = InstalledAppFlow.from_client_secrets_file(secret_path, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the token for future runs
        with open(token_path, "w") as token_file:
            token_file.write(creds.to_json())
        print(f"[DOCS] Token guardado en: {token_path}")

    _docs_service = build("docs", "v1", credentials=creds)
    _drive_service = build("drive", "v3", credentials=creds)

    return _docs_service, _drive_service


def llenar_documento_google(datos_usuario: dict) -> str:
    """
    Copy the Google Docs template and fill it with the user's data.

    Auto-generates: numero_doc, fecha.
    Replaces all {{variable}} placeholders in the document.

    Returns the URL of the newly created document.
    """
    docs_service, drive_service = _get_services()

    # --- Auto-generate fields ---
    numero_doc = f"DOC-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    fecha = datetime.now().strftime("%Y-%m-%d")

    datos_usuario["numero_doc"] = numero_doc
    datos_usuario["fecha"] = fecha
    datos_usuario["tipo_oportunidad"] = "NO CREADA AUN"

    # --- 1. Copy the template via Drive ---
    titulo = f"Oportunidad_{datos_usuario.get('nombre_oportunidad', 'Sin_Nombre')}_{numero_doc}"
    copy_body = {"name": titulo}
    if GOOGLE_DRIVE_FOLDER_ID:
        copy_body["parents"] = [GOOGLE_DRIVE_FOLDER_ID]

    copied_file = (
        drive_service.files()
        .copy(fileId=GOOGLE_DOCS_TEMPLATE_ID, body=copy_body)
        .execute()
    )
    new_doc_id = copied_file["id"]
    print(f"[DOCS] Documento copiado: {new_doc_id} → {titulo}")

    # --- 2. Build replacement requests ---
    variables = [
        "numero_doc",
        "nombre_oportunidad",
        "proponentes_cargos",
        "fecha",
        "origen_oportunidad",
        "situacion_actual",
        "hallazgos",
        "fuente_hallazgos",
        "publico_objetivo",
        "problema_principal",
        "sub_problemas",
        "impacto_esperado",
        "importancia_problema",
        "anexos",
        "tipo_oportunidad",
    ]

    requests = []
    for var in variables:
        value = datos_usuario.get(var, "")
        if value:
            requests.append(
                {
                    "replaceAllText": {
                        "containsText": {
                            "text": "{{" + var + "}}",
                            "matchCase": True,
                        },
                        "replaceText": value,
                    }
                }
            )

    # --- 3. Execute batch update ---
    if requests:
        docs_service.documents().batchUpdate(
            documentId=new_doc_id, body={"requests": requests}
        ).execute()
        print(f"[DOCS] {len(requests)} variables reemplazadas en el documento.")

    doc_url = f"https://docs.google.com/document/d/{new_doc_id}/edit"
    print(f"[DOCS] Documento listo: {doc_url}")

    return doc_url
