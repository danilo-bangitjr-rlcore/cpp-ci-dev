from lib_sql.hello import hello


def test_hello_returns_string():
    assert hello() == "hello from lib_sql"
