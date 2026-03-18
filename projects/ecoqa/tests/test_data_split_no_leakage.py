from collections import defaultdict

from projects.ecoqa.data.generate_qa_pairs_from_yaml import _signature_key, _split_examples


def _example(
    *,
    source_id: str,
    question: str,
    ground_truth_sql: str,
    answer: str,
    source_yaml: str = "demo_qa.yaml",
    table_name: str = "demo_table",
) -> dict:
    return {
        "id": "",
        "source_id": source_id,
        "table_name": table_name,
        "question": question,
        "ground_truth_sql": ground_truth_sql,
        "answer": answer,
        "requires_calculator": False,
        "source_yaml": source_yaml,
    }


def test_split_examples_keeps_duplicate_signatures_in_same_split():
    examples = [
        _example(
            source_id="1",
            question="Q_dup",
            ground_truth_sql="SELECT x FROM demo_table WHERE y=1 LIMIT 1",
            answer='{"items":[{"name":"result","value":1}]}',
        ),
        _example(
            source_id="2",
            question="Q_dup",
            ground_truth_sql="SELECT x FROM demo_table WHERE y=1 LIMIT 1",
            answer='{"items":[{"name":"result","value":1}]}',
        ),
        _example(
            source_id="3",
            question="Q_a",
            ground_truth_sql="SELECT a FROM demo_table WHERE y=2 LIMIT 1",
            answer='{"items":[{"name":"result","value":2}]}',
        ),
        _example(
            source_id="4",
            question="Q_b",
            ground_truth_sql="SELECT b FROM demo_table WHERE y=3 LIMIT 1",
            answer='{"items":[{"name":"result","value":3}]}',
        ),
        _example(
            source_id="5",
            question="Q_c",
            ground_truth_sql="SELECT c FROM demo_table WHERE y=4 LIMIT 1",
            answer='{"items":[{"name":"result","value":4}]}',
        ),
    ]

    train, val, test = _split_examples(examples, train_ratio=0.6, val_ratio=0.2, seed=42)
    assert len(train) + len(val) + len(test) == len(examples)

    split_by_signature: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for split_name, rows in (("train", train), ("val", val), ("test", test)):
        for row in rows:
            split_by_signature[_signature_key(row)].add(split_name)

    assert all(len(splits) == 1 for splits in split_by_signature.values())
