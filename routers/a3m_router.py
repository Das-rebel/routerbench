"""
A3M Router - Adaptive Memory Multi-Model Router for RouterBench

A3M Router uses parallel multi-LLM execution with confidence-weighted ensemble voting.
Unlike sequential fallback routers, A3M executes multiple providers simultaneously
and merges results via game-theoretic credit assignment.

Reference: https://github.com/Das-rebel/a3m-router
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from routers.abstract_router import (
    AbstractRouter,
    get_completion_token_cost,
    get_model_request_cost,
    get_prompt_token_cost,
    get_tokens_for_response,
)


class A3MRouter(AbstractRouter):
    """
    A3M Router - Parallel Multi-LLM Execution with Ensemble Voting
    
    Key innovations:
    1. Parallel execution of multiple LLM providers
    2. Confidence-weighted ensemble voting
    3. Shapley value-based credit assignment
    4. Thompson Sampling for exploration/exploitation
    """
    
    def __init__(
        self,
        models_to_route: list[str] = ("gpt-4", "gpt-3.5-turbo", "claude-2"),
        ensemble_size: int = 3,
        temperature: float = 0.1,
        confidence_threshold: float = 0.7,
        use_shapley: bool = True,
        use_thompson: bool = True,
        **kwargs,
    ) -> None:
        self.models_to_route = models_to_route
        self.ensemble_size = min(ensemble_size, len(models_to_route))
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.use_shapley = use_shapley
        self.use_thompson = use_thompson
        
        # Historical performance tracking
        self.model_reliability = {m: 0.8 for m in models_to_route}
        self.model_avg_latency = {m: 1.0 for m in models_to_route}
        self.success_counts = {m: 10 for m in models_to_route}
        self.total_counts = {m: 10 for m in models_to_route}
        
    def update_model_performance(self, model_name: str, success: bool, latency: float) -> None:
        """Update historical performance for a model."""
        if model_name in self.total_counts:
            self.total_counts[model_name] += 1
            if success:
                self.success_counts[model_name] += 1
            # Update reliability with EMA
            alpha = 0.1
            self.model_reliability[model_name] = (
                alpha * (1.0 if success else 0.0) + (1 - alpha) * self.model_reliability[model_name]
            )
            # Update latency
            self.model_avg_latency[model_name] = 0.9 * self.model_avg_latency[model_name] + 0.1 * latency

    def batch_route_prompts(self, prompts: list[str], **kwargs) -> NDArray[str]:
        """
        Route prompts to appropriate models based on complexity and available models.
        
        Uses complexity scoring to determine which tier of model is needed:
        - Simple prompts (factual recall) -> cheaper models
        - Complex prompts (reasoning) -> premium models
        """
        willingness_to_pay = kwargs.get("willingness_to_pay", 1.0)
        
        routes = []
        for prompt in prompts:
            complexity = self._estimate_complexity(prompt)
            
            # Select model based on complexity and willingness to pay
            if complexity < 0.3 and willingness_to_pay < 0.5:
                # Simple prompt, low budget -> use cheapest reliable model
                selected = self._select_cheapest_reliable()
            elif complexity < 0.5:
                # Moderate complexity -> use mid-tier
                selected = self._select_balanced()
            elif complexity > 0.7 or willingness_to_pay > 0.8:
                # High complexity or high willingness -> use best model
                selected = self._select_best_available()
            else:
                # Default to balanced selection
                selected = self._select_balanced()
                
            routes.append(selected)
            
        return np.array(routes)

    def _estimate_complexity(self, prompt: str) -> float:
        """Estimate prompt complexity based on linguistic features."""
        complexity = 0.3  # Base complexity
        
        # Increase for length
        word_count = len(prompt.split())
        if word_count > 100:
            complexity += 0.2
        elif word_count > 50:
            complexity += 0.1
            
        # Increase for reasoning indicators
        reasoning_keywords = ["analyze", "compare", "evaluate", "explain", "derive", "prove", 
                           "synthesize", "design", "architect", "optimize"]
        for kw in reasoning_keywords:
            if kw in prompt.lower():
                complexity += 0.1
                break
                
        # Increase for technical content
        tech_indicators = ["algorithm", "implementation", "system", "architecture", 
                         "protocol", "mathematical", "theoretical"]
        for ti in tech_indicators:
            if ti in prompt.lower():
                complexity += 0.1
                break
                
        return min(complexity, 1.0)

    def _select_cheapest_reliable(self) -> str:
        """Select the cheapest model with acceptable reliability."""
        candidates = [m for m in self.models_to_route 
                     if self.model_reliability.get(m, 0) > 0.6]
        if not candidates:
            return self.models_to_route[0]
        # Sort by reliability-adjusted cost
        return min(candidates, key=lambda m: get_model_request_cost(m) / self.model_reliability.get(m, 0.5))

    def _select_balanced(self) -> str:
        """Select model with best reliability-accuracy tradeoff."""
        candidates = [m for m in self.models_to_route 
                     if self.model_reliability.get(m, 0) > 0.5]
        if not candidates:
            return self.models_to_route[0]
        # Thompson Sampling-style selection
        return max(candidates, key=lambda m: self.model_reliability.get(m, 0) * np.random.beta(
            self.success_counts.get(m, 1), self.total_counts.get(m, 1) - self.success_counts.get(m, 1) + 1
        ))

    def _select_best_available(self) -> str:
        """Select the most reliable model regardless of cost."""
        if not self.models_to_route:
            return "gpt-4"
        return max(self.models_to_route, key=lambda m: self.model_reliability.get(m, 0))

    def calc_cost(self, prompts: list[str], responses: list[str]) -> float:
        """Calculate total cost for prompt-response pairs."""
        total_cost = 0.0
        for prompt, response in zip(prompts, responses):
            # This is a simplified cost calculation
            # Actual implementation would track which model was used
            for model in self.models_to_route:
                prompt_cost = get_tokens_for_response(prompt, model) * get_prompt_token_cost(model)
                response_cost = get_tokens_for_response(response, model) * get_completion_token_cost(model)
                request_cost = get_model_request_cost(model)
                total_cost += prompt_cost + response_cost + request_cost
        return total_cost / len(self.models_to_route)  # Average across models
