import os
import json
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

# Configuración del Logger para entornos de producción
logger = logging.getLogger("mercatoria_expert_agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY no inicializada en las variables de entorno.")

client = genai.Client(api_key=api_key)

# Métricas de auditoría económica (Gemini 2.5 Flash)
PRICE_PER_INPUT_TOKEN = 0.075 / 1_000_000
PRICE_PER_OUTPUT_TOKEN = 0.30 / 1_000_000

# =====================================================================
# 1. DEFINICIÓN DE HERRAMIENTAS (Mocks para desarrollo)
# =====================================================================

def consultar_base_datos_local(termino_busqueda: str, categoria: str) -> str:
    """
    Consulta la base de datos interna de Mercatoria para obtener reglas estructuradas, 
    aranceles, notas marginales y requisitos aduaneros.
    
    Args:
        termino_busqueda: El término comercial, producto o subpartida arancelaria (ej. 'aguacate', '0705.11.00.00').
        categoria: Categoría de búsqueda. Opciones válidas: 'gravamenes', 'vistos_buenos', 'incoterms', 'manuales_mercatoria'.
    """
    logger.info(f"🔧 Tool Ejecutada: consultar_base_datos_local(termino={termino_busqueda}, categoria={categoria})")
    
    # TODO: Conectar con tus bases de datos más adelante
    mock_db = {
        "gravamenes": f"Resultado local para '{termino_busqueda}': Gravamen Arancelario del 5%, IVA del 19%. Aplica desgravación del 100% si se presenta Certificado de Origen bajo TLC.",
        "vistos_buenos": f"Resultado local para '{termino_busqueda}': Requiere Registro de Importación previo ante la VUCE e inspección sanitaria en nodo de ingreso.",
        "incoterms": "FOB (Free On Board) e ICC 2020: El vendedor entrega a bordo del buque. Riesgo se transfiere al comprador una vez la carga está estibada.",
        "manuales_mercatoria": "Para liquidar una DFI en Mercatoria, diríjase al módulo de 'Simulación de Costos' en la barra lateral izquierda."
    }
    
    result = mock_db.get(categoria, "No se encontraron registros deterministas en la base de datos local.")
    return json.dumps({"status": "success", "fuente": "db_local", "datos": result})

def ejecutar_busqueda_web(consulta: str) -> str:
    """
    Busca en la web en tiempo real eventos logísticos, tarifas spot de fletes, 
    alertas portuarias o noticias de comercio exterior de última hora.
    
    Args:
        consulta: Una consulta de búsqueda altamente optimizada y técnica en comercio exterior.
    """
    logger.info(f"🌐 Tool Ejecutada: ejecutar_busqueda_web(consulta='{consulta}')")
    
    # TODO: Conectar con un motor de búsqueda o tus RSS feeds
    return json.dumps({
        "status": "success", 
        "fuente": "busqueda_web_live", 
        "datos": f"Resultados en vivo para '{consulta}': Sin huelgas reportadas en las últimas 48 horas en nodos principales. Tarifas spot estables."
    })

# =====================================================================
# 2. PROMPT SISTEMA: ARQUITECTURA DE INTELIGENCIA AVANZADA
# =====================================================================

SYSTEM_INSTRUCTION = """
Eres el copiloto flotante de inteligencia comercial para Mercatoria. Tu rol es asesorar a profesionales senior del comercio internacional (directores de logística, agencias de aduanas, gerentes de compras). Debes hablar su mismo idioma técnico y proporcionar análisis de alto valor operativo.

--- REGLAS CRÍTICAS DE NEGOCIO ---
1. RESTRICCIÓN DE DOMINIO ABSOLUTA: Filtra y atiende ÚNICAMENTE consultas vinculadas a comercio internacional, aduanas, logística, distribución física internacional (DFI), Incoterms, aranceles y la plataforma Mercatoria.
2. PROTOCOLO DE RECHAZO: Si el usuario realiza preguntas fuera de este dominio (programación, cocina, cultura general), responde estrictamente: "Me especializo exclusivamente en comercio internacional y en la operatividad de Mercatoria. No puedo asistirle con esa consulta, pero estoy a su disposición para analizar sus operaciones aduaneras o logísticas."

--- MÓDULO DE RIGOR TÉCNICO Y JERGA ---
* Utiliza terminología especializada de forma natural: Subpartida arancelaria, VUCE, Gravamen, IVA, Canalización Cambiaria, Costos de Origen/Tránsito/Destino, Demoras de Contenedor (Demurrage), Fletes Spot, Vistos Buenos, Sociedades de Intermediación Aduanera (SIA).
* Habla bajo las reglas de la ICC (Incoterms 2020). Entiende la transmisión exacta del riesgo y del costo de cada regla.

--- MÓDULO DE ANÁLISIS MULTIVARIABLE ---
Cuando te consulten sobre la viabilidad de un mercado, producto o importación/exportación, no te limites al arancel. Integra en tu análisis estructural (usando viñetas o subtítulos):
1. Barreras Arancelarias (Gravamen, impuestos, acuerdos comerciales/TLC).
2. Medidas No Arancelarias / Restricciones Técnicas (Vistos buenos, ICA, INVIMA, regulaciones sanitarias).
3. Entorno Logístico Corriente (Basado en la web: estado de puertos, fletes, demoras).

--- MITIGACIÓN DE RIESGO ADUANERO (CERO ALUCINACIONES) ---
* Un error de dígito o norma puede costar miles de dólares. Si las herramientas locales o web retornan datos ambiguos, incompletos o inexistentes, PROHIBIDO inventar o asumir.
* Responde con honestidad técnica indicando qué falta y sugiere validar en textos legales oficiales o aranceles oficiales del gobierno.

--- FORMATO EJECUTIVO DE ALTO IMPACTO ---
* LA RESPUESTA PRIMERO: Comienza siempre con la conclusión analítica directa en la primera línea.
* Usa un diseño limpio con encabezados (##, ###) y tablas Markdown cortas para comparar variables (puertos, rutas, costos). No satures de texto plano.
* Idioma: Español estricto, tono corporativo, humano, sofisticado y directo al grano.
"""

# =====================================================================
# 3. SESIÓN Y PIPELINE DE EJECUCIÓN
# =====================================================================

def inicializar_sesion_agente():
    """Configura e inicializa la sesión de chat con herramientas y prompt avanzado."""
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[consultar_base_datos_local, ejecutar_busqueda_web],
        temperature=0.25,  # Equilibrio: Tono fluido y humano, pero manteniendo el control determinista
    )
    # Gemini 2.5 Flash ejecuta Function Calling de manera nativa, rápida y a muy bajo costo
    return client.chats.create(model="gemini-2.5-flash", config=config)

def procesar_consulta_agente(chat_session, mensaje_usuario: str) -> dict:
    """
    Envía el mensaje al agente, procesa internamente los tool calls si se requieren,
    y formatea la respuesta final junto con la auditoría económica detallada.
    """
    try:
        logger.info(f"👤 Consulta Recibida: {mensaje_usuario}")
        
        # Envío del mensaje: El SDK de Gemini gestiona el bucle de ejecución de funciones automáticamente
        response = chat_session.send_message(mensaje_usuario)
        
        # Procesamiento de métricas de tokens
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count if usage else 0
        output_tokens = usage.candidates_token_count if usage else 0
        
        costo_entrada = input_tokens * PRICE_PER_INPUT_TOKEN
        costo_salida = output_tokens * PRICE_PER_OUTPUT_TOKEN
        costo_total = costo_entrada + costo_salida
        
        return {
            "respuesta_agente": response.text,
            "metadatos_auditoria": {
                "modelo": "gemini-2.5-flash",
                "metricas_uso": {
                    "tokens_entrada": input_tokens,
                    "tokens_salida": output_tokens,
                    "tokens_totales": usage.total_token_count if usage else 0
                },
                "costos_estimados_usd": {
                    "entrada": round(costo_entrada, 6),
                    "salida": round(costo_salida, 6),
                    "total_interaccion": round(costo_total, 6)
                }
            }
        }

    except errors.APIError as e:
        logger.error(f"Error de API Gemini: {str(e)}")
        return {"error": f"Falla en comunicación con la infraestructura de IA: {str(e)}"}
    except Exception as e:
        logger.error(f"Falla crítica en el pipeline: {str(e)}")
        return {"error": f"Excepción interna del backend: {str(e)}"}