from lcb_runner.lm_styles import LMStyle, LanguageModel
from openai import OpenAI
from pymilvus import MilvusClient


def build_runner(args, model: LanguageModel, llm_client: OpenAI, cipher_endpoint: str):
    if model.model_style == LMStyle.OpenAIChat:
        from lcb_runner.runner.oai_runner import OpenAIRunner

        return OpenAIRunner(args, model, llm_client, cipher_endpoint)
    if model.model_style in [LMStyle.OpenAIReason, LMStyle.OpenAIReasonPreview]:
        from lcb_runner.runner.oai_runner import OpenAIRunner

        return OpenAIRunner(args, model, llm_client, cipher_endpoint)
    if model.model_style in [LMStyle.Gemini, LMStyle.GeminiThinking]:
        from lcb_runner.runner.gemini_runner import GeminiRunner

        return GeminiRunner(args, model, llm_client, cipher_endpoint)
    if model.model_style in [LMStyle.Claude3, LMStyle.Claude3Thinking]:
        from lcb_runner.runner.claude3_runner import Claude3Runner

        return Claude3Runner(args, model, llm_client, cipher_endpoint)
    if model.model_style == LMStyle.Claude:
        from lcb_runner.runner.claude_runner import ClaudeRunner

        return ClaudeRunner(args, model, llm_client, cipher_endpoint)
    if model.model_style == LMStyle.MistralWeb:
        from lcb_runner.runner.mistral_runner import MistralRunner

        return MistralRunner(args, model, llm_client, cipher_endpoint)
    if model.model_style == LMStyle.CohereCommand:
        from lcb_runner.runner.cohere_runner import CohereRunner

        return CohereRunner(args, model, llm_client, cipher_endpoint)
    if model.model_style == LMStyle.DeepSeekAPI:
        from lcb_runner.runner.deepseek_runner import DeepSeekRunner

        return DeepSeekRunner(args, model, llm_client, cipher_endpoint)
    if model.model_style == LMStyle.DeepSeekAPI:
        from lcb_runner.runner.deepseek_runner import DeepSeekRunner

        return DeepSeekRunner(args, model, llm_client, cipher_endpoint)
    if model.model_style == LMStyle.Grok:
        from lcb_runner.runner.grok_runner import GrokRunner

        return GrokRunner(args, model, llm_client, cipher_endpoint)
    if model.model_style == LMStyle.TogetherAI:
        from lcb_runner.runner.together_runner import TogetherAIRunner

        return TogetherAIRunner(args, model, llm_client, cipher_endpoint)
    if "/fireworks/" in model.model_name:
        from lcb_runner.runner.fireworks_runner import FireWorksRunner

        return FireWorksRunner(args, model, llm_client, cipher_endpoint)
    elif model.model_style in []:
        raise NotImplementedError(
            f"Runner for language model style {model.model_style} not implemented yet"
        )
    else:
        from lcb_runner.runner.vllm_runner import VLLMRunner

        return VLLMRunner(args, model, llm_client, cipher_endpoint)
