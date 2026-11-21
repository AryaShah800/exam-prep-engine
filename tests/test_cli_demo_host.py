from exam_prep import cli


def test_demo_defaults_to_loopback(monkeypatch):
    seen = {}
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.setattr(cli, "_demo", lambda port, host="127.0.0.1": seen.update(port=port, host=host))
    cli.main(["demo"])
    assert seen == {"port": 8000, "host": "127.0.0.1"}


def test_demo_host_and_port_for_containers(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "_demo", lambda port, host="127.0.0.1": seen.update(port=port, host=host))
    cli.main(["demo", "--host", "0.0.0.0", "--port", "7860"])
    assert seen == {"port": 7860, "host": "0.0.0.0"}
