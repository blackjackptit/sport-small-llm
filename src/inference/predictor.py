"""Inference utilities for sports domain LLM."""

from typing import Optional, List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class SportsLLMPredictor:
    """Predictor class for sports domain LLM inference."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_path = model_path

    def load_model(self):
        """Load the trained model for inference."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> str:
        """Generate response for a given prompt."""
        if self.model is None:
            self.load_model()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        response = response[len(prompt) :].strip()

        return response

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat interface for multi-turn conversations."""
        # Format messages into prompt
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"

        prompt += "Assistant: "

        return self.generate(prompt, **kwargs)

    def answer_sports_question(self, question: str) -> str:
        """Answer a sports-related question."""
        prompt = f"""You are a sports expert assistant. Answer the following question accurately and concisely.

Question: {question}

Answer:"""
        return self.generate(prompt)

    def analyze_game(self, game_description: str) -> str:
        """Analyze a game or match."""
        prompt = f"""You are a sports analyst. Analyze the following game and provide insights.

Game Description:
{game_description}

Analysis:"""
        return self.generate(prompt, max_new_tokens=512)
