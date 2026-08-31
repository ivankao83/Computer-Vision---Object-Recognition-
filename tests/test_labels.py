from edge_object_recognition.labels import label_for


def test_known_label_is_returned() -> None:
    assert label_for(15) == "person"


def test_unknown_label_gets_stable_fallback() -> None:
    assert label_for(999) == "class_999"

