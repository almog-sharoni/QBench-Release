import timm; print([m for m in timm.list_models() if 'mobilevit' in m.lower()])
