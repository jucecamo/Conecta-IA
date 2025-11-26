#!/usr/bin/env python
# coding: utf-8

# In[19]:


from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import json
import os

# ----------------------------------------
# Para la IA local
# ----------------------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ---------------------------------------------------------
#   Cargar bases de conocimiento
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")


def cargar_json(nombre_archivo):
    ruta = os.path.join(DATA_DIR, nombre_archivo)
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


TRASTORNOS = cargar_json("trastornos.json")["trastornos"]
EPISODIOS = cargar_json("episodios.json")["episodios"]
ESTADOS = cargar_json("estados.json")["estados"]

# ---------------------------------------------------------
#   Cargar modelo local ligero
# ---------------------------------------------------------

MODEL_NAME = "distilgpt2"  # modelo muy ligero para pruebas
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1  # CPU
)

# ---------------------------------------------------------
#   Crear instancia de la API
# ---------------------------------------------------------

app = FastAPI(
    title="ConectaIA - Backend MVP con IA",
    description="API para análisis básico de episodios emocionales con respuestas empáticas generadas por IA.",
    version="0.3"
)

# ---------------------------------------------------------
# Configurar CORS
# ---------------------------------------------------------
origins = [
    "*",  # o poner el dominio de tu frontend, ej: "http://localhost:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],   # permite GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)

# ---------------------------------------------------------
#   Modelos de entrada
# ---------------------------------------------------------

class EpisodioInput(BaseModel):
    sintomas: list[str] 


class EstadoInput(BaseModel):
    estado: str


# ---------------------------------------------------------
#   Funciones de match
# ---------------------------------------------------------

def buscar_trastorno_por_sintomas(sintomas_usuario):
    sintomas_usuario = [s.lower().strip() for s in sintomas_usuario]
    mejor = None
    mejor_score = 0
    for trast in TRASTORNOS:
        sintomas_t = [s.lower() for s in trast["sintomas_clave"]]
        matches = len(set(sintomas_usuario) & set(sintomas_t))
        total = len(sintomas_t)
        score = matches / total if total > 0 else 0
        if score > mejor_score:
            mejor_score = score
            mejor = trast
    if mejor_score == 0:
        return None
    return mejor


def buscar_estado(entrada):
    entrada = entrada.lower().strip()
    mejor = None
    mejor_score = 0
    for estado in ESTADOS:
        e_texto = estado["estado"].lower()
        if entrada in e_texto or e_texto in entrada:
            return estado
        palabras_usuario = set(entrada.split())
        palabras_estado = set(e_texto.split())
        score = len(palabras_usuario & palabras_estado)
        if score > mejor_score:
            mejor_score = score
            mejor = estado
    return mejor

# ---------------------------------------------------------
#   Función para generar respuesta natural empática
# ---------------------------------------------------------

def generar_respuesta(prompt_text, max_length=150):
    salida = generator(prompt_text, max_length=max_length, do_sample=True, temperature=0.7)
    return salida[0]['generated_text']

# ---------------------------------------------------------
#   ENDPOINT 1 — Episodio → Diagnóstico → IA empática
# ---------------------------------------------------------

@app.post("/episodio/")
def analizar_episodio(data: EpisodioInput):
    # Convertimos la cadena en lista separando por comas
    sintomas_lista = [s.strip() for s in data.sintomas if s.strip()]
    trastorno = buscar_trastorno_por_sintomas(sintomas_lista)

    if trastorno is None:
        return {
            "mensaje": "No pude identificar un episodio con esos síntomas.",
            "advertencia": "Esta herramienta NO reemplaza atención médica profesional."
        }

    # Construir respuesta limpia y formateada
    respuesta = {
        "posible_caso": trastorno["nombre"],
        "descripcion": trastorno["descripcion"],
        "primeros_auxilios": trastorno["primeros_auxilios"],
        "recomendaciones_convivencia": trastorno["recomendaciones_convivencia"],
        "senales_alarma": trastorno["senales_alarma"],
        "respuesta_empatica": (
            f"Detectamos que podrías estar experimentando un episodio de '{trastorno['nombre']}'. "
            f"{trastorno['descripcion']} "
            f"Primeros auxilios: {trastorno['primeros_auxilios']} "
            f"Recomendaciones de convivencia: {trastorno['recomendaciones_convivencia']} "
            f"Señales de alarma a vigilar: {', '.join(trastorno['senales_alarma'])}."
        ),
        "advertencia": "Esta herramienta NO es un diagnóstico médico."
    }

    return respuesta


# ---------------------------------------------------------
#   ENDPOINT 2 — Estado alterado → IA empática
# ---------------------------------------------------------

@app.post("/estado/")
def recomendaciones_estado(data: EstadoInput):
    estado = buscar_estado(data.estado)

    if estado is None:
        return {
            "mensaje": "Aún no tengo recomendaciones para ese estado.",
            "advertencia": "Esta herramienta no reemplaza atención profesional."
        }

    # Respuesta empática resumida
    respuesta_empatica = (
        f"{estado['mensaje_empatico']} "
        f"{estado['recomendacion']} "
        f"{estado['sugerencia_extra']}"
    )

    return {
        "estado_detectado": estado["estado"],
        "": respuesta_empatica.strip(),
        "advertencia": estado["advertencia"]
    }

# ---------------------------------------------------------
# Root
# ---------------------------------------------------------

@app.get("/")
def root():
    return {"message": "Backend ConectaIA con IA local funcionando correctamente."}

@app.get("/sintomas/")
def listar_sintomas():
    sintomas = []
    for trastorno in TRASTORNOS:
        sintomas.extend(trastorno["sintomas_clave"])
    # Eliminamos duplicados y ordenamos
    sintomas_unicos = sorted(list(set(sintomas)))
    return {"sintomas": sintomas_unicos}

@app.get("/estados/")
def listar_estados():
    estados_unicos = sorted([estado["estado"] for estado in ESTADOS])
    return {"estados": estados_unicos}


# In[ ]:




