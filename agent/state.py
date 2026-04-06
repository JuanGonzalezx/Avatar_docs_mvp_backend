from typing import Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State for the opportunity documentation agent."""
    historial_chat: list              # List of {"role": "user"/"assistant", "content": "..."}
    nombre_oportunidad: Optional[str]
    proponentes_cargos: Optional[str]
    origen_oportunidad: Optional[str]
    situacion_actual: Optional[str]
    hallazgos: Optional[str]
    fuente_hallazgos: Optional[str]
    publico_objetivo: Optional[str]
    problema_principal: Optional[str]
    sub_problemas: Optional[str]
    impacto_esperado: Optional[str]
    importancia_problema: Optional[str]
    anexos: Optional[str]
    documento_generado: bool
    respuesta: str
    documento_url: Optional[str]


# Fields the agent must collect
CAMPOS_REQUERIDOS = [
    "nombre_oportunidad",
    "proponentes_cargos",
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
]


def campos_faltantes(state: AgentState) -> list[str]:
    """Return list of fields that are still None or empty."""
    return [c for c in CAMPOS_REQUERIDOS if not state.get(c)]


def estado_completo(state: AgentState) -> bool:
    """Check if all required fields have been filled."""
    return len(campos_faltantes(state)) == 0


def crear_estado_inicial() -> AgentState:
    """Create a fresh initial state."""
    return AgentState(
        historial_chat=[],
        nombre_oportunidad=None,
        proponentes_cargos=None,
        origen_oportunidad=None,
        situacion_actual=None,
        hallazgos=None,
        fuente_hallazgos=None,
        publico_objetivo=None,
        problema_principal=None,
        sub_problemas=None,
        impacto_esperado=None,
        importancia_problema=None,
        anexos=None,
        documento_generado=False,
        respuesta="",
        documento_url=None,
    )
