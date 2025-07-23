def test_basic():
    assert 1 + 1 == 2
    print("✅ Тест работает!")


def test_import():
    from src.wb_ranking_model import WBRankingModel

    model = WBRankingModel()
    assert model is not None
    print("✅ Код импортируется!")
