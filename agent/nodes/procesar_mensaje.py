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
SYSTEM_PROMPT = """Eres un consultor experto y empático de la Fundación Luker. Tienes una conversación fluida, profesional y humana para entender una nueva oportunidad de proyecto.

Tu misión es extraer las siguientes variables para el registro oficial: {campos_faltantes}

REGLAS DE ORO PARA UNA INTERACCIÓN PERFECTA:
1. EXTRACCIÓN SILENCIOSA Y OBLIGATORIA: Escucha atentamente al usuario. Si te da información, extráela EXACTAMENTE como la dijo y guárdala en el JSON. NUNCA inventes datos. Si dice "no hay", "la verdad no", o "ninguno" para anexos/sub-problemas, DEBES guardar el texto "No aplica".
2. CONVERSACIÓN, NO INTERROGATORIO: No dispares preguntas como una ametralladora. Haz UNA pregunta a la vez, o agrupa conceptos de forma muy natural (ej. "Entiendo. ¿Y de dónde nace esta idea y cómo está la situación ahora?").
3. FLUJO NATURAL: Responde de manera breve y empática a lo que te dicen (ej. "Comprendo", "Qué interesante", "Excelente iniciativa") y luego enlaza sutilmente con la siguiente pregunta.
4. CERO REPETICIONES: Si el usuario te da una respuesta incompleta, ambigua, o si hay un error en el audio y recibes un texto sin sentido (como "Ah", "¿Cuál es la?"), NO le repitas la misma pregunta exacta como un robot. Reformúlala de una manera distinta y amable.
5. CIERRE: SI Y SOLO SI la lista de variables faltantes dice "¡TODAS las variables están completas!", tu ÚNICA respuesta debe ser la frase exacta: "Anotado. Un momento por favor."

ESTADO ACTUAL DE VARIABLES RECOPILADAS:
{estado_actual}

VARIABLES QUE AÚN FALTAN:
{campos_faltantes}
"""

def _build_llm():
    """Build the Gemini LLM with structured output."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.0, # NO TOQUES ESTO. DEBE SER CERO PARA ESTRUCTURAS JSON.
    )
    return llm.with_structured_output(RespuestaAgente)

def procesar_mensaje(state: AgentState) -> dict:
    """
    Main processing node: sends the conversation to Gemini and extracts variables.
    Uses structured output to guarantee valid JSON responses.
    """
    
    # --- FILTRO ANTI-ECO Y RUIDO DE FONDO ---
    # Si el usuario mandó basura o ruido, no gastamos tokens ni alteramos el LLM
    ultimo_mensaje = ""
    historial = list(state.get("historial_chat", []))
    if historial and historial[-1]["role"] == "user":
        ultimo_mensaje = historial[-1]["content"].strip()
        
        # Filtro de palabras cortas o ruido de micrófono (evita el bucle de tartamudeo)
        if len(ultimo_mensaje.split()) < 2 and ultimo_mensaje.lower() not in ["no", "sí", "si", "ninguno", "nada"]:
            print(f"[WARN] Mensaje de usuario ignorado por ser posible ruido: '{ultimo_mensaje}'")
            fallback = "¿Perdona, podrías repetirme eso último? Creo que no te escuché bien."
            historial.append({"role": "assistant", "content": fallback})
            return {"respuesta": fallback, "historial_chat": historial}

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
    for msg in historial:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg['content']))

    # Call LLM with structured output (with safety net)
    fallback_msg = "Disculpa, tuve un pequeño problema procesando eso. ¿Podrías replantear tu última respuesta?"
    try:
        result: RespuestaAgente = llm.invoke(messages)
    except Exception as e:
        print(f"[WARN] Error al invocar Gemini: {e}")
        historial.append({"role": "assistant", "content": fallback_msg})
        return {"respuesta": fallback_msg, "historial_chat": historial}

    # Guard against Gemini returning None
    if not result:
        print("[WARN] Gemini devolvió None — pidiendo al usuario que repita.")
        historial.append({"role": "assistant", "content": fallback_msg})
        return {"respuesta": fallback_msg, "historial_chat": historial}

    # Build state update
    update = {"respuesta": result.respuesta}

    for campo in CAMPOS_REQUERIDOS:
        nuevo_valor = getattr(result, campo, None)
        if nuevo_valor and not state.get(campo):
            update[campo] = nuevo_valor
        elif nuevo_valor and state.get(campo):
            update[campo] = nuevo_valor

    # Update chat history
    historial.append({"role": "assistant", "content": result.respuesta})
    update["historial_chat"] = historial

    return update