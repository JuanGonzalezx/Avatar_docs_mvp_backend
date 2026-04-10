import json
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

from config import GEMINI_API_KEY
from agent.state import AgentState, campos_faltantes, CAMPOS_REQUERIDOS


# --- Pydantic schema for structured output ---
class RespuestaAgente(BaseModel):
    """Structured output from the Gemini LLM."""
    respuesta: str = Field(description="Mensaje conversacional para el usuario")
    nombre_oportunidad: Optional[str] = Field(default=None, description="Nombre de la oportunidad/iniciativa")
    proponentes_cargos: Optional[str] = Field(default=None, description="Proponentes y sus cargos")
    origen_oportunidad: Optional[str] = Field(default=None, description="De dónde nace la oportunidad")
    situacion_actual: Optional[str] = Field(default=None, description="Situación actual, qué está pasando")
    hallazgos: Optional[str] = Field(default=None, description="Hallazgos principales encontrados")
    fuente_hallazgos: Optional[str] = Field(default=None, description="Fuentes o referencias de los hallazgos")
    publico_objetivo: Optional[str] = Field(default=None, description="Cliente, mercado o usuario objetivo")
    problema_principal: Optional[str] = Field(default=None, description="Problema principal identificado")
    sub_problemas: Optional[str] = Field(default=None, description="Sub-problemas derivados")
    impacto_esperado: Optional[str] = Field(default=None, description="Impacto esperado de la solución")
    importancia_problema: Optional[str] = Field(default=None, description="Importancia estratégica del problema")
    anexos: Optional[str] = Field(default=None, description="Descripción de anexos mencionados por el usuario")


# --- System prompt ---
SYSTEM_PROMPT = """Eres un asistente virtual amigable de la Fundación Luker, diseñado para recopilar información sobre oportunidades de forma conversacional y natural.

Tu objetivo es llenar TODAS estas variables: nombre_oportunidad, proponentes_cargos, origen_oportunidad, situacion_actual, hallazgos, fuente_hallazgos, publico_objetivo, problema_principal, sub_problemas, impacto_esperado, importancia_problema, anexos.

REGLAS ESTRICTAS:
1. Revisa la lista de "VARIABLES QUE AÚN FALTAN" en la parte inferior. Tu respuesta debe SIEMPRE ser una pregunta conversacional y natural para averiguar las siguientes 1 o 2 variables faltantes de la lista.
2. Sigue este orden lógico para preguntar:
   - Paso 1: Pregunta el nombre de la oportunidad/iniciativa y quiénes son los proponentes con sus cargos.
   - Paso 2: Pregunta de dónde nace la oportunidad y qué está pasando actualmente (situación).
   - Paso 3: Pregunta qué hallazgos encontraron y de qué fuentes/referencias obtuvieron esa información.
   - Paso 4: Pregunta hacia qué público objetivo va dirigido, cuál es el problema principal, y si hay sub-problemas.
   - Paso 5: Pregunta cuál es el impacto esperado, la importancia estratégica, y si hay anexos que adjuntar.
3. SIEMPRE DEBES HACER UNA PREGUNTA AL USUARIO, a menos que la lista de VARIABLES QUE AÚN FALTAN esté completamente vacía.
4. NUNCA respondas solamente "Anotado. Un momento por favor" a menos que literalmente ya no falte NINGUNA variable. MIENTRAS FALTEN VARIABLES, SIEMPRE haz la siguiente pregunta para continuar la conversación.
5. NO suenes robótico ni hagas listas de preguntas. NUNCA resumas, repitas ni confirmes la información que el usuario dio en turnos anteriores. Haz directamente la siguiente pregunta. NUNCA repitas saludos después del primer turno.
6. Sólo si la lista de "VARIABLES QUE AÚN FALTAN" indica "¡TODAS las variables están completas!", tu ÚNICA respuesta debe ser exclusivamente: "Anotado. Un momento por favor."
7. Cuando extraigas información del mensaje del usuario, devuélvela en los campos correspondientes del JSON. Si el usuario responde "ninguno", "no hay" o similar, llena ese campo con "No aplica". ¡Nunca inventes datos!

ESTADO ACTUAL DE VARIABLES RECOPILADAS:
{estado_actual}

VARIABLES QUE AÚN FALTAN:
{campos_faltantes}
"""


def _build_llm():
    """Build the Gemini LLM with structured output."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=GEMINI_API_KEY,
        temperature=0.7,
    )
    return llm.with_structured_output(RespuestaAgente)


def procesar_mensaje(state: AgentState) -> dict:
    """
    Main processing node: sends the conversation to Gemini and extracts variables.
    Uses structured output to guarantee valid JSON responses.
    """
    llm = _build_llm()

    # Build current state summary for the system prompt
    estado_actual = {}
    for campo in CAMPOS_REQUERIDOS:
        valor = state.get(campo)
        if valor:
            estado_actual[campo] = valor

    faltantes = campos_faltantes(state)

    system_msg = SYSTEM_PROMPT.format(
        estado_actual=json.dumps(estado_actual, ensure_ascii=False, indent=2) if estado_actual else "Ninguna variable recopilada aún.",
        campos_faltantes=", ".join(faltantes) if faltantes else "¡TODAS las variables están completas!",
    )

    # Build message history for the LLM
    messages = [SystemMessage(content=system_msg)]
    for msg in state.get("historial_chat", []):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg['content']))

    # Call LLM with structured output (with safety net)
    fallback_msg = "Disculpa, tuve un pequeño problema técnico procesando eso. ¿Podrías repetirme tu última respuesta?"
    try:
        result: RespuestaAgente = llm.invoke(messages)
    except Exception as e:
        print(f"[WARN] Error al invocar Gemini: {e}")
        historial = list(state.get("historial_chat", []))
        historial.append({"role": "assistant", "content": fallback_msg})
        return {"respuesta": fallback_msg, "historial_chat": historial}

    # Guard against Gemini returning None (safety filter or structured output failure)
    if not result:
        print("[WARN] Gemini devolvió None — pidiendo al usuario que repita.")
        historial = list(state.get("historial_chat", []))
        historial.append({"role": "assistant", "content": fallback_msg})
        return {"respuesta": fallback_msg, "historial_chat": historial}

    # Build state update — only update fields that the LLM extracted
    update = {"respuesta": result.respuesta}

    for campo in CAMPOS_REQUERIDOS:
        nuevo_valor = getattr(result, campo, None)
        if nuevo_valor and not state.get(campo):
            update[campo] = nuevo_valor
        elif nuevo_valor and state.get(campo):
            # Allow overwriting if user corrects information
            update[campo] = nuevo_valor

    # Update chat history
    historial = list(state.get("historial_chat", []))
    # The last user message was already added by the controller
    historial.append({"role": "assistant", "content": result.respuesta})
    update["historial_chat"] = historial

    return update
