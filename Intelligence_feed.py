import os
import re
import json
import logging
import feedparser
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

# Configuración del Logger para entornos de producción (el logger es seguro para backends)
logger = logging.getLogger("trade_intelligence_cron")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY no inicializada en las variables de entorno.")

client = genai.Client(api_key=api_key)

# Métricas de auditoría económica (gemini-2.5-flash-lite)
PRICE_PER_INPUT_TOKEN = 0.10 / 1_000_000
PRICE_PER_OUTPUT_TOKEN = 0.40 / 1_000_000

# Fuentes de Datos Estratégicas para Inteligencia en Comercio Exterior
COLOMBIA_FEEDS = [
    "https://www.valoraanalitik.com/feed/",
    "https://www.banrep.gov.co/es/noticias-rss",
    "https://www.larepublica.co/rss/economia",
    "https://www.larepublica.co/rss/infraestructura",
    "https://procolombia.co/sala-de-prensa/noticias"
]

INTERNATIONAL_FEEDS = [
    "https://www.wto.org/english/news_e/news_e.xml",
    "https://unctad.org/rss/news",
    "https://www.imf.org/en/News/RSS?category=Press+Releases",
    "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/news-events/rss.xml",
    "https://www.usitc.gov/press_room/news_release.xml",
    "https://www.cnbc.com/id/10000067/device/rss/rss.html",
    "https://www.federalreserve.gov/feeds/press_all.xml"
]

def _fetch_feed_layer(feed_urls, max_articles=8):
    """Capa de extracción perimetral con control de desbordamiento de búfer de tokens."""
    compiled_articles = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/rss+xml, application/xml, text/xml, text/html;q=0.9, */*;q=0.8"
    }

    for url in feed_urls:
        try:
            response = requests.get(url, headers=headers, timeout=8)
            if response.status_code != 200:
                continue
                
            feed = feedparser.parse(response.content)
            if not feed.entries:
                continue
                
            for entry in feed.entries[:max_articles]:
                article = {
                    "title": entry.get("title", "").strip(),
                    "snippet": entry.get("summary", entry.get("description", "")),
                    "url": entry.get("link", "")
                }
                
                if article["snippet"]:
                    article["snippet"] = re.sub('<[^<]+?>', '', article["snippet"]).strip()
                    
                if article["title"] and article["title"] not in [a["title"] for a in compiled_articles]:
                    compiled_articles.append(article)
        except Exception as e:
            logger.debug(f"Interrupción leve de lectura en origen {url}: {str(e)}")
            continue
            
    return compiled_articles

def _build_token_payload(articles):
    text_block = ""
    for idx, art in enumerate(articles):
        snippet = art['snippet'][:200] if art['snippet'] else "Sin metadatos descritos."
        text_block += f"[{idx}] TÍTULO: {art['title']}\nRESUMEN: {snippet}\nURL: {art['url']}\n\n"
    return text_block

def run_trade_pipeline():
    """
    Tubería de ejecución automatizada. 
    Retorna un diccionario limpio listo para serialización JSON HTTP completamente en español.
    """
    co_articles = _fetch_feed_layer(COLOMBIA_FEEDS)
    global_articles = _fetch_feed_layer(INTERNATIONAL_FEEDS)
    
    if not co_articles and not global_articles:
        logger.error("Extracción fallida: 0 artículos acumulados en el pool de feeds.")
        return {"error": "No se alcanzaron fuentes de noticias base durante este ciclo de ejecución."}
        
    co_text = _build_token_payload(co_articles) if co_articles else "Sin datos."
    global_text = _build_token_payload(global_articles) if global_articles else "Sin datos."
    
    prompt = f"""
    Actúa como el motor analítico de Comercio Exterior para Mercatoria.
    Tu tarea es procesar las siguientes fuentes noticiosas y estructurarlas para una actualización que ocurre cada 12 horas.

    --- NOVEDADES: COLOMBIA ---
    {co_text}
    
    --- NOVEDADES: INTERNACIONAL ---
    {global_text}
    
    CRITERIOS DE FILTRADO Y RESTRICCIÓN:
    1. BRIEFING DIARIO: Diseña un único título analítico para el ciclo actual de 12 horas y un análisis integrado enfocado en importaciones/exportaciones de Colombia. Genera listas estrictas de 3 elementos para 'puntos_clave', 'oportunidades' y 'riesgos'.
    2. NOTICIAS CLASIFICADAS: Selecciona e integra un máximo estricto de hasta 10 noticias (combinando Colombia e Internacional) que tengan el impacto más alto en aduanas, logística o balanza de pagos. No devuelvas más de 10 elementos bajo ninguna circunstancia. Categorías válidas: 'Mercados', 'Logística', 'Acuerdos', 'Regulación'. Impactos válidos: 'alta', 'media', 'baja'.
    3. ALERTAS COMERCIALES: Captura las regulaciones urgentes de la FDA, la DIAN, o alertas portuarias inmediatas. Prioridades: 'crítica', 'importante', 'informativa'.

    Idioma requerido de respuesta: Español.
    
    Responde exclusivamente bajo el siguiente esquema JSON:
    {{
      "briefing_diario": {{
        "titulo_macro": "",
        "analisis_ejecutivo": "",
        "puntos_clave": [],
        "oportunidades": [],
        "riesgos": []
      }},
      "noticias_clasificadas": [
        {{
          "titulo": "",
          "extracto": "",
          "categoria_tag": "",
          "impacto_tag": "",
          "url": ""
        }}
      ],
      "alertas_comerciales": [
        {{
          "alerta_titulo": "",
          "descripcion": "",
          "prioridad": "",
          "tags": []
        }}
      ]
    }}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                system_instruction="Debes generar absolutamente todo el contenido y las respuestas en idioma español."
            )
        )
        
        payload_data = json.loads(response.text)
        
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count
        
        cost_input = input_tokens * PRICE_PER_INPUT_TOKEN
        cost_output = output_tokens * PRICE_PER_OUTPUT_TOKEN
        total_predicted_cost = cost_input + cost_output
        
        payload_data["metadatos_pipeline"] = {
            "modelo_ejecucion": "gemini-2.5-flash-lite",
            "frecuencia_ciclo": "programado_cada_12_horas",
            "metricas": {
                "tokens_entrada": input_tokens,
                "tokens_salida": output_tokens,
                "total_tokens": usage.total_token_count
            },
            "costos_estimados_usd": {
                "entrada": round(cost_input, 6),
                "salida": round(cost_output, 6),
                "total_ejecucion": round(total_predicted_cost, 6)
            }
        }
        
        return payload_data

    except errors.APIError as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "billing" in error_msg or "429" in error_msg or "403" in error_msg:
            logger.error(f"Alerta de Facturación/Cuota en Gemini API: {str(e)}")
            return {"error": "Se ha agotado el saldo o la cuota de la API de Gemini. Por favor, verifica la facturación en Google Cloud/AI Studio."}
        
        logger.error(f"Error de la API de Gemini: {str(e)}")
        return {"error": f"Error de comunicación con la API: {str(e)}"}

    except json.JSONDecodeError:
        logger.error("Falla de decodificación: La respuesta del modelo no conservó la sintaxis estructural.")
        return {"error": "Estructura JSON malformada generada por el modelo subyacente."}
    except Exception as e:
        logger.error(f"Falla crítica del sistema: {str(e)}")
        return {"error": f"Excepción interna del pipeline: {str(e)}"}