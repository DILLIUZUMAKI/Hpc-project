import json
import urllib.error
from unittest.mock import MagicMock, patch
import pytest
from hpc_optimizer.llm import LLMClient, LLMConfig, LLMError


def test_chat_request_and_json_response_are_compatible():
    model_content=json.dumps({"code":"int main(void){return 0;}","explanation":"Synthetic API fixture"})
    body=json.dumps({"choices":[{"message":{"content":model_content},"finish_reason":"stop"}],"usage":{"total_tokens":12}}).encode()
    response=MagicMock(); response.__enter__.return_value.read.return_value=body
    opener=MagicMock(); opener.open.return_value=response
    with patch("hpc_optimizer.llm.urllib.request.build_opener",return_value=opener):
        result=LLMClient(LLMConfig("https://example.invalid/v1","test-model","secret")).generate({"system":"rules","user":"source"})
    request=opener.open.call_args.args[0]
    assert request.full_url=="https://example.invalid/v1/chat/completions"
    assert request.get_header("Authorization")=="Bearer secret"
    assert json.loads(request.data)["messages"][1]["content"]=="source"
    assert result["usage"]["total_tokens"]==12 and "secret" not in json.dumps(result)


def test_http_errors_do_not_echo_provider_body_or_secret():
    opener=MagicMock()
    opener.open.side_effect=urllib.error.HTTPError("https://secret.invalid",401,"SECRET_KEY",None,None)
    with patch("hpc_optimizer.llm.urllib.request.build_opener",return_value=opener):
        with pytest.raises(LLMError) as caught:
            LLMClient(LLMConfig("https://example.invalid/v1","test","SECRET_KEY")).generate({"system":"x","user":"y"})
    assert "401" in str(caught.value) and "SECRET_KEY" not in str(caught.value)
