from ids_tier2.explanation_producer import StubExplanationProducer


def test_stub_producer_records_published_explanations():
    producer = StubExplanationProducer()
    event = {"explanation_id": "e-1", "alert_id": "a-1"}
    producer.publish(event)
    assert producer.published == [event]


def test_stub_producer_close_marks_closed():
    producer = StubExplanationProducer()
    assert producer.closed is False
    producer.close()
    assert producer.closed is True
