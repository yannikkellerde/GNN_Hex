def hasattr(x:object, y:str)->bool:
    """Mokeypatch for graph-tool properties.

    Graph-tool Graphs throw KeyError instead of AttributeError when
    calling gettattr on them.

    Args:
        x: object
        y: attribute name
    Returns:
        if x has attribute y
    """
    try:
        getattr(x, y)
        return True
    except (AttributeError, KeyError):
        return False