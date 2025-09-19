# tests/test_llm_service.py

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import time
from src.services.llm_service import LLMService, create_llm_service

class MockInferenceClient:
    """Mock HuggingFace InferenceClient for testing"""
    
    def __init__(self, model=None, token=None):
        self.model = model
        self.token = token
    
    def chat_completion(self, messages, max_tokens, temperature):
        """Mock chat completion response"""
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        
        # Simulate different responses based on input
        user_message = messages[1]["content"] if len(messages) > 1 else ""
        
        if "BIP" in user_message or "Bruttoinlandsprodukt" in user_message:
            mock_message.content = "**Antwort:** Das Bruttoinlandsprodukt (BIP) ist der Gesamtwert aller produzierten Güter und Dienstleistungen."
        elif "GSYH" in user_message:  # Turkish
            mock_message.content = "**Cevap:** Gayri Safi Yurt İçi Hasıla (GSYH) bir ülkede üretilen tüm mal ve hizmetlerin toplam değeridir."
        elif "error" in user_message.lower():
            raise Exception("Mocked API error")
        else:
            mock_message.content = "**Antwort:** Test response generated successfully."
        
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        return mock_response


class TestLLMService:
    """Test cases for LLMService"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            "llm": {
                "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "temperature": 0.1,
                "max_tokens": 512,
                "language": "de-turkish",
                "system_prompt": "Du bist ein akademischer Assistent.",
                "api_token": "test_token_12345"
            }
        }
    
    @pytest.fixture
    def sample_contexts(self):
        """Sample context data for testing"""
        return [
            {
                "text": "Das Bruttoinlandsprodukt (BIP) misst den Gesamtwert aller Güter und Dienstleistungen, die in einem Land produziert werden.",
                "hybrid_score": 0.85
            },
            {
                "text": "Wirtschaftswachstum wird durch die prozentuale Veränderung des BIP gemessen.",
                "hybrid_score": 0.72
            },
            {
                "text": "İktisat bilimi kaynak dağılımını inceler.",
                "hybrid_score": 0.68
            }
        ]
    
    @patch('huggingface_hub.InferenceClient', MockInferenceClient)
    @patch('src.utils.config_parser.CONFIG')
    def test_llm_service_initialization(self, mock_config_parser, mock_config):
        """Test LLMService initialization with proper config"""
        mock_config_parser.get.return_value = mock_config["llm"]
        
        # Test with default model
        llm_service = LLMService()
        
        assert llm_service.model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct"
        assert llm_service.temperature == 0.1
        assert llm_service.max_tokens == 512
        assert llm_service.language == "de-turkish"
        assert llm_service.client is not None
    
    @patch('huggingface_hub.InferenceClient', MockInferenceClient)
    @patch('src.utils.config_parser.CONFIG')
    def test_llm_service_custom_model(self, mock_config_parser, mock_config):
        """Test LLMService with custom model name"""
        mock_config_parser.get.return_value = mock_config["llm"]
        
        custom_model = "custom/model-name"
        llm_service = LLMService(model_name=custom_model)
        
        assert llm_service.model_name == custom_model
    
    @patch('src.utils.config_parser.CONFIG')
    def test_llm_service_missing_api_token(self, mock_config_parser):
        """Test that missing API token raises ValueError"""
        mock_config_parser.get.return_value = {"temperature": 0.1}  # No api_token
        
        with pytest.raises(ValueError, match="HF API Token is missing"):
            LLMService()
    
    @patch('huggingface_hub.InferenceClient', MockInferenceClient)
    @patch('src.utils.config_parser.CONFIG')
    def test_build_prompt_with_contexts(self, mock_config_parser, mock_config, sample_contexts):
        """Test prompt building with context data"""
        mock_config_parser.get.return_value = mock_config["llm"]
        llm_service = LLMService()
        
        query = "Was ist das BIP?"
        prompt = llm_service._build_prompt(query, sample_contexts)
        
        # Check that prompt contains key elements
        assert "Persona:" in prompt
        assert "Context:" in prompt
        assert "User Question:" in prompt
        assert query in prompt
        assert "Das Bruttoinlandsprodukt" in prompt  # From context
        assert "Relevanz: 0.85" in prompt  # Hybrid score
    
    @patch('huggingface_hub.InferenceClient', MockInferenceClient)
    @patch('src.utils.config_parser.CONFIG')
    def test_build_prompt_no_contexts(self, mock_config_parser, mock_config):
        """Test prompt building without context data"""
        mock_config_parser.get.return_value = mock_config["llm"]
        llm_service = LLMService()
        
        query = "Was ist Wirtschaft?"
        prompt = llm_service._build_prompt(query, [])
        
        assert "Keine Dokumente verfügbar" in prompt
        assert query in prompt
    
    @patch('huggingface_hub.InferenceClient', MockInferenceClient)
    @patch('src.utils.config_parser.CONFIG')
    def test_generate_response_successful(self, mock_config_parser, mock_config, sample_contexts):
        """Test successful response generation"""
        mock_config_parser.get.return_value = mock_config["llm"]
        llm_service = LLMService()
        
        query = "Was ist das BIP?"
        response = llm_service.generate_response(query, sample_contexts)
        
        # Check response structure
        assert isinstance(response, dict)
        assert "answer" in response
        assert "prompt" in response
        assert "generation_time_ms" in response
        assert "model_name" in response
        assert "contexts_used" in response
        
        # Check response values
        assert "Bruttoinlandsprodukt" in response["answer"]
        assert response["model_name"] == "meta-llama/Meta-Llama-3.1-8B-Instruct"
        assert response["contexts_used"] == 3
        assert response["generation_time_ms"] >= 0
    
    @patch('huggingface_hub.InferenceClient', MockInferenceClient)
    @patch('src.utils.config_parser.CONFIG')
    def test_generate_response_no_contexts(self, mock_config_parser, mock_config):
        """Test response generation without contexts"""
        mock_config_parser.get.return_value = mock_config["llm"]
        llm_service = LLMService()
        
        query = "Grundlagen der Wirtschaft?"
        response = llm_service.generate_response(query)
        
        assert response["contexts_used"] == 0
        assert "Test response generated successfully" in response["answer"]
    
    @patch('huggingface_hub.InferenceClient')
    @patch('src.utils.config_parser.CONFIG')
    def test_api_error_handling(self, mock_config_parser, mock_client_class, mock_config):
        """Test handling of API errors"""
        mock_config_parser.get.return_value = mock_config["llm"]
        
        # Mock client that raises exception
        mock_client = Mock()
        mock_client.chat_completion.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client
        
        llm_service = LLMService()
        response = llm_service.generate_response("test query")
        
        assert "Entschuldigung, es gab einen Fehler" in response["answer"]
    
    @patch('huggingface_hub.InferenceClient', MockInferenceClient)
    @patch('src.utils.config_parser.CONFIG')
    def test_get_model_info(self, mock_config_parser, mock_config):
        """Test model info retrieval"""
        mock_config_parser.get.return_value = mock_config["llm"]
        llm_service = LLMService()
        
        info = llm_service.get_model_info()
        
        assert info["model_name"] == "meta-llama/Meta-Llama-3.1-8B-Instruct"
        assert info["temperature"] == 0.1
        assert info["max_tokens"] == 512
        assert info["language"] == "de-turkish"
    
    @patch('huggingface_hub.InferenceClient', MockInferenceClient)
    @patch('src.utils.config_parser.CONFIG')
    def test_context_sorting_and_limiting(self, mock_config_parser, mock_config):
        """Test that contexts are properly sorted by score and limited to top 5"""
        mock_config_parser.get.return_value = mock_config["llm"]
        llm_service = LLMService()
        
        # Create 7 contexts with different scores
        many_contexts = [
            {"text": f"Context {i}", "hybrid_score": i * 0.1}
            for i in range(7, 0, -1)  # Scores: 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
        ]
        
        query = "Test query"
        prompt = llm_service._build_prompt(query, many_contexts)
        
        # Should contain top 5 contexts (scores 0.7-0.3)
        assert "Relevanz: 0.700" in prompt
        assert "Relevanz: 0.300" in prompt
        # Should not contain lowest scores
        assert "Relevanz: 0.200" not in prompt
        assert "Relevanz: 0.100" not in prompt
    
    @patch('huggingface_hub.InferenceClient', MockInferenceClient)
    @patch('src.utils.config_parser.CONFIG')
    def test_performance_timing(self, mock_config_parser, mock_config):
        """Test that generation timing is recorded"""
        mock_config_parser.get.return_value = mock_config["llm"]
        llm_service = LLMService()
        
        start_time = time.time()
        response = llm_service.generate_response("Performance test")
        end_time = time.time()
        
        # Generation time should be reasonable
        assert 0 <= response["generation_time_ms"] <= (end_time - start_time) * 1000 + 100


class TestCreateLLMService:
    """Test the factory function"""
    
    @patch('huggingface_hub.InferenceClient', MockInferenceClient)
    @patch('src.utils.config_parser.CONFIG')
    def test_create_llm_service_default(self, mock_config_parser):
        """Test factory function with default parameters"""
        mock_config = {
            "model_name": "test-model",
            "temperature": 0.2,
            "max_tokens": 256,
            "language": "de-turkish",
            "system_prompt": "Test prompt",
            "api_token": "test_token"
        }
        mock_config_parser.get.return_value = mock_config
        
        service = create_llm_service()
        
        assert isinstance(service, LLMService)
        assert service.model_name == "test-model"
    
    @patch('huggingface_hub.InferenceClient', MockInferenceClient)
    @patch('src.utils.config_parser.CONFIG')
    def test_create_llm_service_custom_model(self, mock_config_parser):
        """Test factory function with custom model"""
        mock_config = {
            "model_name": "default-model",
            "temperature": 0.2,
            "max_tokens": 256,
            "language": "de-turkish",
            "system_prompt": "Test prompt",
            "api_token": "test_token"
        }
        mock_config_parser.get.return_value = mock_config
        
        custom_model = "custom/test-model"
        service = create_llm_service(model_name=custom_model)
        
        assert service.model_name == custom_model


# Integration-style tests (can be run with real API if needed)
class TestLLMServiceIntegration:
    """Integration tests - can be skipped for unit testing"""
    
    @pytest.mark.skipif(not (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")), reason="Requires real HF API token")
    def test_real_api_call(self):
        """Test with real API - skip by default"""
        # This would test against real HuggingFace API
        # Only run when you want to test the actual integration
        pass


if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__ + "::TestLLMService::test_llm_service_initialization", "-v"])