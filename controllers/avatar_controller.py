import uuid
import requests as http_requests
from flask import Blueprint, request, jsonify

from config import HEYGEN_API_KEY, LIVEAVATAR_AVATAR_ID
from agent.state import crear_estado_inicial, estado_completo
from agent.graph import graph

avatar_bp = Blueprint("avatar", __name__, url_prefix="/api")

# In-memory session store: {session_id: AgentState}
sessions: dict = {}


@avatar_bp.route("/get-access-token", methods=["POST"])
def get_access_token():
    """Proxy endpoint to get LiveAvatar session token."""
    try:
        response = http_requests.post(
            "https://api.liveavatar.com/v1/sessions/token",
            headers={
                "X-Api-Key": HEYGEN_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "mode": "FULL",
                "avatar_id": LIVEAVATAR_AVATAR_ID,
                "avatar_persona": {},
            },
        )
        data = response.json()
        if response.status_code == 200 and data.get("data"):
            token = data["data"].get("session_token", "")
            return jsonify({"token": token}), 200
        return jsonify({"error": "Failed to get token", "details": data}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@avatar_bp.route("/chat", methods=["POST"])
def chat():
    """
    Process a chat message through the LangGraph agent.
    Expects: {"mensaje": "...", "session_id": "..."}
    Returns: {"respuesta": "...", "completado": bool, "documento_url": "..." | null}
    """
    body = request.get_json()
    if not body or "mensaje" not in body:
        return jsonify({"error": "Campo 'mensaje' es requerido"}), 400

    mensaje = body["mensaje"]
    session_id = body.get("session_id", str(uuid.uuid4()))

    # Get or create session state
    if session_id not in sessions:
        sessions[session_id] = crear_estado_inicial()

    state = sessions[session_id]

    # Don't process if document already generated
    if state.get("documento_generado"):
        return jsonify({
            "respuesta": state.get("respuesta", "El documento ya fue generado."),
            "completado": True,
            "documento_url": state.get("documento_url"),
            "session_id": session_id,
        })

    # Add user message to history
    historial = list(state.get("historial_chat", []))
    historial.append({"role": "user", "content": mensaje})
    state["historial_chat"] = historial

    # Run the LangGraph
    try:
        result = graph.invoke(state)
        # Update session state
        sessions[session_id] = result

        return jsonify({
            "respuesta": result.get("respuesta", ""),
            "completado": result.get("documento_generado", False),
            "documento_url": result.get("documento_url"),
            "session_id": session_id,
        })
    except Exception as e:
        print(f"[ERROR] Error en /chat: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@avatar_bp.route("/session/reset", methods=["POST"])
def reset_session():
    """Reset a session to start a new conversation."""
    body = request.get_json() or {}
    session_id = body.get("session_id")
    if session_id and session_id in sessions:
        del sessions[session_id]
    return jsonify({"message": "Session reset", "session_id": session_id})
