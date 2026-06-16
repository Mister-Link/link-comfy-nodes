from __future__ import annotations


class _AnyType(str):
    def __ne__(self, _other):
        return False


_any = _AnyType("*")


def _parse_value(s: str):
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


class StringToListNode:
    CATEGORY = "utils"
    RETURN_TYPES = (_any,)
    RETURN_NAMES = ("list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "128, 128, 256, 96", "multiline": False}),
                "delimiter": ("STRING", {"default": ","}),
            }
        }

    def run(self, text: str, delimiter: str):
        items = [_parse_value(p) for p in text.split(delimiter) if p.strip()]
        return (items,)
