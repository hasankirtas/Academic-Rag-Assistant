"""
LLM Service for RAG System (Cloud-based by default)

Provides text generation using Hugging Face Inference API (Llama 3.1).
"""

from typing import List, Dict, Any, Optional
import time
import logging
import huggingface_hub as hf
import src.utils.config_parser as config_module

from src.utils.logger import setup_logger
from src.utils.config_parser import CONFIG

logger = setup_logger(__name__)


class LLMService:
    """
    Service for text generation using Hugging Face Inference API (Llama 3.1).
    """

    def __init__(self, model_name: Optional[str] = None):
        # Access CONFIG dynamically so test patches are honored
        llm_config = config_module.CONFIG.get("llm", {})

        self.model_name = model_name or llm_config.get(
            "model_name", "meta-llama/Meta-Llama-3.1-8B-Instruct"
        )
        self.temperature = llm_config.get("temperature", 0.1)
        self.max_tokens = llm_config.get("max_tokens", 512)
        self.language = llm_config.get("language", "de-turkish")
        self.system_prompt = llm_config.get(
            "system_prompt",
            ""
        )

        api_token = llm_config.get("api_token", None)
        if not api_token:
            raise ValueError("HF API Token is missing")

        # Use module attribute so patch('huggingface_hub.InferenceClient', ...) works
        self.client = hf.InferenceClient(model=self.model_name, token=api_token)
        logger.info(f"Initialized cloud LLM service with {self.model_name}")

    def _build_prompt(self, query: str, contexts: List[Dict]) -> str:
        """
        Build a professional instruction prompt for German academic content,
        integrating persona, instructions, context, and language handling.
        
        Args:
            query: User query
            contexts: List of context dictionaries with 'text' and 'hybrid_score'
        
        Returns:
            Formatted instruction prompt string
        """

        # 1. Prepare context string
        context_str = ""
        if contexts:
            # Limit total context to top 5 most relevant
            sorted_contexts = sorted(contexts, key=lambda x: x.get('hybrid_score', 0), reverse=True)[:5]
            for i, ctx in enumerate(sorted_contexts, 1):
                text = ctx.get("text", "")[:2000]  # truncate very long texts
                context_str += f"{i}. (Relevanz: {ctx.get('hybrid_score', 0):.3f})\n{text}\n\n"

        # 2. Professional instruction template (avoid few-shot examples when no contexts to keep prompt minimal)
        base_prompt = f"""
            Persona:

            Du bist ein hochqualifizierter akademischer Assistent, spezialisiert auf deutsche Wirtschafts- und Managementthemen. Du verstehst komplexe akademische Inhalte und kannst sie klar erklären.

            Instruction:

            Beantworte die folgende Frage präzise, basierend hauptsächlich auf den bereitgestellten Dokumentenauszügen. Stelle sicher, dass alle Informationen korrekt und gut begründet sind. Keine Annahmen außerhalb der Dokumente treffen.

            Context:

            Die folgenden Dokumentenauszüge stehen zur Verfügung. Sie enthalten relevante Definitionen, Beispiele, Formeln und Konzepte. Nutze diese Auszüge als Grundlage deiner Antwort.

            {context_str if context_str else "Keine Dokumente verfügbar."}

            Context Processing Rules:

            - Falls der bereitgestellte Kontext sehr umfangreich ist (>2000 Wörter), priorisiere die relevantesten 3-5 Abschnitte für die Antwort.
            - Nutze Dokument-Referenzen: "[Quelle: Dokument X, Seite Y]" wenn verfügbar.
            - Falls Informationen widersprüchlich sind, erwähne dies explizit: "Die Quellen zeigen unterschiedliche Ansätze..."

            Format:

            - Nummerierte Absätze, wenn mehrere Punkte erklärt werden
            - Definitionen fett markieren
            - Beispiele kursiv schreiben
            - Formeln innerhalb von LaTeX-Syntax darstellen, wenn vorhanden

            Citation Requirements:

            - Direkte Fakten: Immer mit Dokumenten-Referenz versehen
            - Beispiele: Als solche markieren und unterscheiden von faktischen Inhalten
            - Formeln: Mit Quelle angeben, falls aus Dokumenten stammend
            - Eigene Erklärungen: Klar als "ergänzende Erklärung" kennzeichnen

            Response Length Guidelines:

            - Standard-Fragen: Maximum 250 Wörter
            - Komplexe Themen: Maximum 400 Wörter
            - Falls eine Frage sehr breit ist, fokussiere auf die wichtigsten 3 Kernpunkte
            - Nutze prägnante Sätze und vermeide Wiederholungen

            Audience:

            Die Antwort richtet sich an Studierende im ersten oder zweiten Semester eines Universitätsstudiums, die die Konzepte verstehen sollen.

            Tone:

            Formell, akademisch, klar und präzise, aber leicht verständlich für Studierende.

            Language Handling:

            - Detectiere die Sprache der Benutzerfrage automatisch.
            - Wenn die Benutzerfrage auf Türkisch gestellt wird, antworte auf Türkisch.
            - Wenn die Benutzerfrage auf Deutsch gestellt wird, antworte auf Deutsch.
            - Akademische Begriffe (Fachtermini) auf Deutsch belassen. Bei Türkisch: beim ersten Auftreten **Türkische Übersetzung (Almanca)**, danach nur Deutsch.

            Fallback Procedures:

            - Falls deutsche Fachbegriffe unklar sind: Nutze alternative deutsche Begriffe oder kurze Definitionen
            - Falls Dokument-Kontext unvollständig ist: "Diese Frage erfordert zusätzliche Fachliteratur für eine vollständige Antwort."
            - Falls technische Formeln fehlen: "Für mathematische Details siehe entsprechende Fachliteratur."

            Ethics & Safety:

            - Keine Inhalte erstellen, die beleidigend, diskriminierend oder politisch extremistisch sind.
            - Keine medizinischen, rechtlichen oder finanziellen Ratschläge geben, die riskant sein könnten.
            - Keine Annahmen oder Spekulationen über nicht im Kontext enthaltene Informationen treffen.
            - Keine persönlichen Daten verarbeiten oder weitergeben.
            - Wenn die Frage außerhalb des akademischen Kontexts liegt, antworte neutral: "Ich kann diese Anfrage nicht beantworten."

            Reliability & Enrichment:

            - Die Antwort sollte **hauptsächlich** auf den bereitgestellten Dokumentenauszügen basieren.
            - Du darfst nur kleine zusätzliche Erklärungen oder Beispiele hinzufügen, die die dokumentierten Informationen unterstützen.
            - Spekulationen oder Annahmen außerhalb des Kontexts vermeiden.

            Confidence Indicators:

            - Bei sicheren, dokumentierten Informationen: Normale Antwort
            - Bei teilweise belegten Informationen: "Basierend auf den verfügbaren Quellen..."
            - Bei unsicheren Interpretationen: "Die Dokumente deuten darauf hin..."
            - Falls Kontext unvollständig: "Für eine vollständige Antwort wären zusätzliche Informationen nötig."

            """

        if contexts:
            examples_block = f"""
            Few-Shot Examples:

            Example 1:

            Question: Was ist das Bruttoinlandsprodukt (BIP)?

            Context: Das BIP misst den Gesamtwert aller Güter und Dienstleistungen, die in einem Land in einem bestimmten Zeitraum produziert werden. Es wird häufig zur Messung der wirtschaftlichen Leistung genutzt.

            Answer: **Antwort:**

            1. **Definition:** Das Bruttoinlandsprodukt (BIP) ist der Gesamtwert aller produzierten Güter und Dienstleistungen innerhalb eines Landes in einem bestimmten Zeitraum.

            2. **Erklärung:** Es dient als Maß für die wirtschaftliche Leistung eines Landes und hilft, Wirtschaftswachstum oder -rückgang zu bewerten.

            3. **Beispiel:** Wenn ein Land in einem Jahr Autos, Maschinen und Dienstleistungen im Wert von 500 Milliarden Euro produziert, entspricht dies dem BIP dieses Jahres.
            """
        else:
            examples_block = ""

        tail = f"""
            User Question:

            {query}

            Output Indicator:

            Beginne die Antwort mit "**Antwort:**" und beantworte die Frage Schritt für Schritt, wobei du die Dokumentenauszüge als Referenz nutzt. Befolge die Language Handling Regeln: Türkçe → Antwort Türkçe mit akademischen Termen, Almanca → Antwort Deutsch.
            """

        instruction_prompt = (base_prompt + examples_block + tail)
        return instruction_prompt.strip()

    def _generate_answer(self, prompt: str) -> str:
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in conversational generation: {str(e)}")
            return "Entschuldigung, es gab einen Fehler bei der Antwortgenerierung."

    def generate_response(self, query: str, contexts: List[Dict] = None) -> Dict[str, Any]:
        start = time.time()
        prompt = self._build_prompt(query, contexts or [])
        answer = self._generate_answer(prompt)
        elapsed = (time.time() - start) * 1000

        return {
            "answer": answer,
            "prompt": prompt,
            "generation_time_ms": elapsed,
            "model_name": self.model_name,
            "contexts_used": len(contexts) if contexts else 0,
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "language": self.language,
        }


def create_llm_service(model_name: Optional[str] = None) -> LLMService:
    service = LLMService(model_name=model_name)
    logger.info("LLMService created successfully")
    return service


if __name__ == "__main__":
    llm = create_llm_service()
    query = "Was ist das Bruttoinlandsprodukt?"
    contexts = [
        {"text": "Das Bruttoinlandsprodukt misst den Wert aller Güter und Dienstleistungen.", "hybrid_score": 0.85}
    ]

    response = llm.generate_response(query, contexts)
    print("Answer:", response["answer"])
    print("Generation time:", response["generation_time_ms"], "ms")
    print("Model info:", llm.get_model_info())