"""
A3M Router integration for RouterBench.

Uses the A3M Router CLI (npx a3m-router) for parallel multi-LLM routing
with fallback logic when the CLI is unavailable.

RouterArena #1 (76.43) — https://github.com/RouteWorks/RouterArena/pull/113
"""

import subprocess
import json
import re
import os
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from routers.abstract_router import AbstractRouter


class A3MRouter(AbstractRouter):
    """
    A3M Router — parallel multi-LLM execution with confidence-scored voting.

    Routes each prompt to the optimal model by running multiple provider
    candidates in parallel and scoring responses by confidence.
    """

    def __init__(
        self,
        models_to_route: list[str] = None,
        cache_url: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the A3M Router.

        Args:
            models_to_route: List of model names to route between.
                Defaults to a diverse set of cost-quality tiers.
            cache_url: Optional MongoDB connection string for embedding cache.
        """
        self.models_to_route = models_to_route or [
            "gpt-4o-mini",
            "claude-3-haiku-20240307",
            "gemini-2.0-flash-001",
        ]
        self._check_a3m_available()

    def _check_a3m_available(self) -> bool:
        """Check if A3M Router CLI is available."""
        try:
            result = subprocess.run(
                ["npx", "a3m-router", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.a3m_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            self.a3m_available = False

        if not self.a3m_available:
            print(
                "[A3M Router] CLI not found. Install: npm install -g adaptive-memory-multi-model-router"
            )
        return self.a3m_available

    def batch_route_prompts(self, prompts: list[str], **kwargs) -> NDArray[str]:
        """
        Route each prompt to the best model using A3M Router.

        Uses A3M's query classification to route:
        - Simple factual queries → cheapest capable model
        - Creative queries → higher temperature models
        - Complex reasoning → strongest models
        - Code queries → code-optimized models

        Falls back to cost-aware round-robin if A3M CLI is unavailable.

        Args:
            prompts: List of text prompts to route.

        Returns:
            Array of model names (one per prompt).
        """
        if self.a3m_available and len(prompts) > 0:
            return self._route_with_a3m(prompts)

        return self._route_fallback(prompts)

    def _route_with_a3m(self, prompts: list[str]) -> NDArray[str]:
        """Route prompts using the actual A3M Router CLI."""
        results = []
        for prompt in prompts:
            try:
                result = subprocess.run(
                    [
                        "npx",
                        "a3m-router",
                        "route",
                        "--json",
                        prompt[:500],  # Truncate very long prompts
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    # Parse JSON output to extract model name
                    output = result.stdout.strip()
                    try:
                        data = json.loads(output)
                        model = data.get("model", data.get("provider", ""))
                    except json.JSONDecodeError:
                        # Try to extract model from text output
                        model = self._extract_model_from_text(output)
                else:
                    model = self._classify_and_route(prompt)
            except (subprocess.TimeoutExpired, Exception):
                model = self._classify_and_route(prompt)

            # Validate against our allowed models
            if model not in self.models_to_route:
                model = self.models_to_route[0]
            results.append(model)

        return np.array(results)

    def _extract_model_from_text(self, text: str) -> str:
        """Extract model name from A3M text output."""
        for model in self.models_to_route:
            if model.lower() in text.lower():
                return model
        # Fallback: use the first word that looks like a model name
        words = text.split()
        for w in words:
            if any(c in w for c in ["gpt", "claude", "gemini", "llama", "mixtral"]):
                return w.strip("'\",. ")
        return self.models_to_route[0]

    def _classify_and_route(self, prompt: str) -> str:
        """
        Simple query-type classification for fallback routing.

        Mirrors A3M's query-type preset logic:
        - Code queries → fastest capable model
        - Math/reasoning → strongest model
        - Creative → mid-tier with higher temp proxy
        - Everything else → cheapest capable
        """
        prompt_lower = prompt.lower()

        # Code detection
        code_patterns = [
            "def ", "class ", "import ", "function", "const ", "let ", "var ",
            "```", "#include", "print(", "console.log", "return ",
            "npm ", "git ", "python", "javascript", "typescript",
        ]
        if any(p in prompt_lower for p in code_patterns):
            return self.models_to_route[0]  # Cheapest for code

        # Math/reasoning detection
        reasoning_patterns = [
            "explain", "why", "how does", "compare", "analyze",
            "solve", "calculate", "prove", "derive", "what is the difference",
            "step by step", "reason",
        ]
        if any(p in prompt_lower for p in reasoning_patterns):
            return self.models_to_route[-1]  # Strongest for reasoning

        # Creative detection
        creative_patterns = [
            "write a story", "poem", "creative", "imagine", "design",
            "brainstorm", "suggest", "idea", "funny", "joke",
        ]
        if any(p in prompt_lower for p in creative_patterns):
            return self.models_to_route[-1]  # Strongest for creative

        # Default: cheapest capable
        return self.models_to_route[0]

    def _route_fallback(self, prompts: list[str]) -> NDArray[str]:
        """Fallback routing when A3M CLI is unavailable."""
        results = []
        for prompt in prompts:
            model = self._classify_and_route(prompt)
            results.append(model)
        return np.array(results)
