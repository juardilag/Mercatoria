import os
import re
import json
import feedparser
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Configuración de Entorno e IA
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY no se encuentra en el archivo .env")

client = genai.Client(api_key=api_key)

# Métricas de control de costos para Gemini 2.5 Flash
PAID_PRICE_PER_INPUT_TOKEN = 0.075 / 1_000_000
PAID_PRICE_PER_OUTPUT_TOKEN = 0.30 / 1_000_000

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

def fetch_rss_news(feed_urls, max_articles=15):
    """Descarga vectores informativos inyectando encabezados corporativos para evitar bloqueos."""
    compiled_articles = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/rss+xml, application/xml, text/xml, text/html;q=0.9, */*;q=0.8",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8"
    }

    for url in feed_urls:
        try:
            response = requests.get(url, headers=headers, timeout=12)
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
                    
                # Filtro de duplicados por título analizado
                if article["title"] and article["title"] not in [a["title"] for a in compiled_articles]:
                    compiled_articles.append(article)
        except Exception:
            continue
            
    return compiled_articles

def format_articles_for_llm(articles):
    text_block = ""
    for idx, art in enumerate(articles):
        snippet = art['snippet'][:250] if art['snippet'] else "Sin descripción disponible en origen."
        text_block += f"[{idx}] TÍTULO: {art['title']}\nRESUMEN: {snippet}\nURL: {art['url']}\n\n"
    return text_block

def generate_intelligence_report():
    print("🛰️ Extrayendo Vectores Económicos de Colombia...")
    co_articles = fetch_rss_news(COLOMBIA_FEEDS)
    print(f"   -> {len(co_articles)} historias colombianas recuperadas.")
    
    print("🌍 Extrayendo Vectores Macroeconómicos Internacionales...")
    global_articles = fetch_rss_news(INTERNATIONAL_FEEDS)
    print(f"   -> {len(global_articles)} historias globales recuperadas.")
    
    if not co_articles and not global_articles:
        print("❌ Error Fatal: No se pudo recolectar información de ninguna fuente primaria.")
        return None, None
        
    co_text = format_articles_for_llm(co_articles) if co_articles else "No hay datos locales disponibles."
    global_text = format_articles_for_llm(global_articles) if global_articles else "No hay datos globales disponibles."
    
    print("🧠 Ejecutando Analizador de Comercio Exterior (Gemini 2.5 Flash)...")
    
    prompt = f"""
    Actúa como el Analista Principal de Comercio Exterior y Estrategia de Cadena de Suministro para Mercatoria. 
    Tu objetivo es destilar el flujo masivo de noticias crudas en tres productos de alta fidelidad: Briefing Diario, Noticias Clasificadas y Alertas de Comercio Exterior.

    --- ENTRADA DE DATOS: COLOMBIA ---
    {co_text}
    
    --- ENTRADA DE DATOS: INTERNACIONAL ---
    {global_text}
    
    INSTRUCCIONES DE DISEÑO Y FILTRADO:
    1. BRIEFING DIARIO: Genera un análisis ejecutivo macro de alto nivel (un eje conceptual unificado). Debe incluir un título analítico potente, un resumen estratégico profundo, y tres desgloses tácticos en listas de objetos: 'puntos_clave', 'oportunidades' y 'riesgos'.
    2. NOTICIAS: Extrae las noticias individuales con relevancia aduanera o comercial directa. Clasifícalas usando exactamente estos tags de categoría ('Mercados', 'Logística', 'Acuerdos', 'Regulación') y tags de impacto ('alta', 'media', 'baja').
    3. ALERTAS: Detecta cambios críticos y urgentes (ej. incrementos de demanda en nichos específicos, nuevas regulaciones aduaneras de la FDA/DIAN, volatilidad de commodities). Clasifica su nivel en: 'crítica', 'importante', 'informativa'. Incluye un array de palabras clave útiles ('tags') como 'café', 'textiles', 'Estados Unidos'.
    
    Idioma de respuesta corporativa: Español.
    
    Devuelve la respuesta estrictamente bajo el siguiente esquema JSON estructurado:
    {{
      "briefing_diario": {{
        "titulo_macro": "Título de impacto macroeconómico o geopolítico",
        "analisis_ejecutivo": "Texto analítico integrado de 4-5 líneas evaluando el escenario para exportadores colombianos.",
        "puntos_clave": ["Punto clave 1 con enfoque de comercio exterior", "Punto clave 2"],
        "oportunidades": ["Oportunidad comercial táctica o de sustitución identificada", "Oportunidad 2"],
        "riesgos": ["Riesgo regulatorio, logístico o cambiario detectado", "Riesgo 2"]
      }},
      "noticias_clasificadas": [
        {{
          "titulo": "Título de la noticia original o refinado profesionalmente",
          "extracto": "Resumen contextualizado del impacto comercial de la noticia en 2 líneas.",
          "categoria_tag": "Logística",
          "impacto_tag": "alta",
          "url": "url_origen"
        }}
      ],
      "alertas_comerciales": [
        {{
          "alerta_titulo": "Ej: Nuevas regulaciones de trazabilidad para frutas en EE. UU.",
          "descripcion": "Explicación breve de la ventana de acción o peligro aduanero.",
          "prioridad": "crítica",
          "tags": ["Regulación", "frutas", "Estados Unidos"]
        }}
      ]
    }}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2  # Temperatura baja para maximizar precisión en la clasificación de datos
            )
        )
        return response.text, response
    except Exception as e:
        print(f"❌ Error en la capa de procesamiento del Modelo AI: {e}")
        return None, None

if __name__ == "__main__":
    final_json, raw_response = generate_intelligence_report()
    
    if final_json and raw_response:
        print("\n📦 --- JSON ESTRUCTURADO PARA FRONTEND ---")
        print(final_json)
        
        # Procesamiento de analíticas de ejecución
        usage = raw_response.usage_metadata
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count
        
        costo_entrada = input_tokens * PAID_PRICE_PER_INPUT_TOKEN
        costo_salida = output_tokens * PAID_PRICE_PER_OUTPUT_TOKEN
        valor_ahorrado = costo_entrada + costo_salida
        
        print("\n📊 --- MÉTRICAS DE OPERACIÓN DEL AGENTE ---")
        print(f"• Motor de Inferencia: gemini-2.5-flash")
        print(f"• Volumen de Entrada (Tokens): {input_tokens}")
        print(f"• Volumen de Salida (Tokens): {output_tokens}")
        print(f"• Costo de Operación en Free Tier: $0.00 USD")
        print(f"• Valor de Carga Equivalente Ahorrado: ${valor_ahorrado:.6f} USD")
        print("-------------------------------------------------------")