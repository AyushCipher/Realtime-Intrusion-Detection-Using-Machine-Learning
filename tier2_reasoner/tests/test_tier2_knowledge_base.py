from ids_tier2.knowledge_base import KNOWLEDGE_BASE, categories_covered


def test_knowledge_base_is_non_empty():
    assert len(KNOWLEDGE_BASE) > 0


def test_every_entry_has_a_technique_id_shaped_like_attck():
    for entry in KNOWLEDGE_BASE:
        assert entry.technique_id.startswith("T")
        assert entry.technique_id[1:5].split(".")[0].isdigit()


def test_technique_ids_are_unique():
    ids = [e.technique_id for e in KNOWLEDGE_BASE]
    assert len(ids) == len(set(ids))


def test_categories_covered_matches_this_projects_attack_families():
    # These should line up with ids_ml.data.ATTACK_CATEGORY_MAP's family
    # vocabulary (checked by name here rather than importing ids_ml, per
    # this module's no-code-dependency boundary).
    covered = categories_covered()
    for expected in ("Botnet", "Brute Force", "DoS/DDoS", "PortScan", "Web Attack"):
        assert expected in covered
