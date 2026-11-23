from aiss import ConversionConfig, convert_batch


def test_batch_stable_paths_order_and_error_isolation(tmp_path):
    source = tmp_path / "lines.json"
    source.write_text('{"size":[10,10],"polylines":[[[2,2],[8,8]]]}')
    missing = tmp_path / "missing.png"
    config = ConversionConfig(columns=12, max_proposals=0, seed=11)
    first = convert_batch([source, missing], config=config, workers=2)
    second = convert_batch([missing, source], config=config, workers=1)
    assert [item.path for item in first] == [source, missing]
    assert first[0].result.text == second[1].result.text
    assert first[0].seed == second[1].seed == first[0].result.config.seed
    assert first[1].seed == second[0].seed != first[0].seed
    assert first[1].result is None and "FileNotFoundError" in first[1].error
