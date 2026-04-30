import sys

if __name__ == "src":
    sys.modules.setdefault("runspace.src", sys.modules[__name__])
elif __name__ == "runspace.src":
    sys.modules.setdefault("src", sys.modules[__name__])
