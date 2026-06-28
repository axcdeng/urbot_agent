from app.agent.llm_client import LLMClient, LLMMetrics, extract_json_object


def test_metrics_aggregate_across_calls():
    client = LLMClient.__new__(LLMClient)  # avoid network/settings
    client.metrics = LLMMetrics()
    client._record({"usage": {"prompt_tokens": 25, "completion_tokens": 90}}, 3.0)
    client._record({"usage": {"prompt_tokens": 30, "completion_tokens": 60}}, 2.0)
    m = client.metrics
    assert m.calls == 2
    assert m.prompt_tokens == 55
    assert m.completion_tokens == 150
    assert m.total_tokens == 205
    assert m.elapsed_s == 5.0
    assert round(m.tps, 2) == 30.0  # 150 output tokens / 5.0 s


def test_metrics_tps_zero_when_no_time():
    assert LLMMetrics().tps == 0.0


def test_extract_json_strips_think_and_fences():
    assert extract_json_object("<think>plan</think>\n```json\n{\"a\": 1}\n```") == {"a": 1}
