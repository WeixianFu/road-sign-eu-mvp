from pathlib import Path

from roadsigns.ontology import Ontology


def test_semantic_distinctions_and_review_conflicts():
    ontology = Ontology(Path(__file__).resolve().parents[1] / "configs/ontology.csv")

    def target(source):
        return ontology.names[ontology.mapping[source]]

    assert len(ontology.names) == 153
    assert target(133) == target(270)
    assert target(77) != target(85)
    assert target(19) == target(100)
    assert target(100) != target(44)
    assert target(146) != target(28)
    assert target(191) != target(296)
    assert target(106) == target(395)
    assert target(106) != target(317) != target(341)
    assert len({target(i) for i in [9, 96, 333]}) == 3
    assert target(159) != target(316)
    assert target(92) != target(25)
    assert target(58) != target(63)
    assert ontology.review_ids == {80, 150, 260, 271, 304}
    assert 304 not in ontology.mapping
    assert ontology.rows[304]["action"] == "review"
    assert set(ontology.mapping) | ontology.review_ids | {
        int(r["source_id"]) for r in ontology.rows if r["action"] == "exclude"
    } == set(range(401))
