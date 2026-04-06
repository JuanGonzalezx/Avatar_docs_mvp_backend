from langgraph.graph import StateGraph, START, END

from agent.state import AgentState, estado_completo
from agent.nodes.procesar_mensaje import procesar_mensaje
from services.database_service import insertar_oportunidad
from services.google_docs_service import llenar_documento_google


# --- Nodes ---

def nodo_procesar(state: AgentState) -> dict:
    """Wrapper node for message processing."""
    return procesar_mensaje(state)


def nodo_generar_documento(state: AgentState) -> dict:
    """Generate Google Doc and save to database when all fields are complete."""
    datos = {
        "nombre_oportunidad": state["nombre_oportunidad"],
        "proponentes_cargos": state["proponentes_cargos"],
        "origen_oportunidad": state["origen_oportunidad"],
        "situacion_actual": state["situacion_actual"],
        "hallazgos": state["hallazgos"],
        "fuente_hallazgos": state["fuente_hallazgos"],
        "publico_objetivo": state["publico_objetivo"],
        "problema_principal": state["problema_principal"],
        "sub_problemas": state["sub_problemas"],
        "impacto_esperado": state["impacto_esperado"],
        "importancia_problema": state["importancia_problema"],
        "anexos": state.get("anexos", ""),
    }

    import traceback

    # Generate Google Doc
    doc_url = ""
    try:
        doc_url = llenar_documento_google(datos)
        print(f"[GRAPH] Documento creado exitosamente: {doc_url}")
    except Exception as e:
        print(f"[ERROR] Error al generar documento en Google Docs: {e}")
        traceback.print_exc()
        doc_url = f"Error: No se pudo generar el documento: {str(e)}"

    # Save to Turso
    try:
        datos["documento_url"] = doc_url
        insertar_oportunidad(datos)
        print("[GRAPH] Oportunidad guardada en BD exitosamente.")
    except Exception as e:
        print(f"[ERROR] Error al guardar en BD Turso: {e}")
        traceback.print_exc()

    return {
        "documento_generado": True,
        "documento_url": doc_url,
        "respuesta": "¡Perfecto! He recopilado toda la información y el documento de la oportunidad ha sido generado exitosamente. Puedes acceder a él haciendo clic en el botón de la pantalla.",
    }


# --- Conditional edge ---

def verificar_completitud(state: AgentState) -> str:
    """Route to document generation if all fields are complete, otherwise end."""
    from agent.state import campos_faltantes
    faltantes = campos_faltantes(state)
    print(f"\n[SEMÁFORO] Verificando completitud. Campos faltantes: {faltantes}")
    if len(faltantes) == 0 and not state.get("documento_generado"):
        print("[SEMÁFORO] ✅ Todos los campos completos → generando documento...")
        return "generar_documento"
    if state.get("documento_generado"):
        print("[SEMÁFORO] Documento ya fue generado previamente.")
    return END


# --- Graph builder ---

def build_graph():
    """Build and compile the LangGraph StateGraph."""
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("procesar_mensaje", nodo_procesar)
    builder.add_node("generar_documento", nodo_generar_documento)

    # Add edges
    builder.add_edge(START, "procesar_mensaje")
    builder.add_conditional_edges(
        "procesar_mensaje",
        verificar_completitud,
        {
            "generar_documento": "generar_documento",
            END: END,
        },
    )
    builder.add_edge("generar_documento", END)

    return builder.compile()


# Compile the graph once at module level
graph = build_graph()
