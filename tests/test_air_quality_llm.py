from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import air_quality_llm


class AirQualityPromptTests(unittest.TestCase):
    def test_chat_prompt_does_not_force_a_fixed_template(self) -> None:
        with patch("air_quality_llm.build_dynamic_context", return_value="DATOS DE PRUEBA"):
            prompt = air_quality_llm.build_user_prompt("¿Cuál es el NO2 actual en Patraix?")

        self.assertIn("¿Cuál es el NO2 actual en Patraix?", prompt)
        self.assertIn("No uses una plantilla fija", prompt)
        self.assertIn("No anadas una seccion `Datos relevantes` por rutina", prompt)
        self.assertIn("No anadas predicciones a preguntas sobre la situacion actual", prompt)
        self.assertIn("nunca afirmes que falta un dato", prompt)
        self.assertNotIn("usa el titulo `### Datos relevantes`", prompt)

    def test_chat_role_reinforces_the_question_scope(self) -> None:
        with (
            patch("air_quality_llm.build_user_prompt", return_value="PROMPT"),
            patch("air_quality_llm.call_mistral_api", return_value="RESPUESTA") as call,
        ):
            response = air_quality_llm.ask_air_quality("pregunta")

        self.assertEqual(response, "RESPUESTA")
        role = call.call_args.args[1]
        self.assertIn("alcance exacto", role)
        self.assertIn("sin repetir una estructura fija", role)


if __name__ == "__main__":
    unittest.main()
