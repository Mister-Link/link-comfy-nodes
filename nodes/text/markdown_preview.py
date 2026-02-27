from __future__ import annotations


class PreviewAsMarkdown:
    CATEGORY = "utils"
    RETURN_TYPES = ()
    FUNCTION = "preview_markdown"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def __init__(self):
        self.min_height = 180

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (
                    "STRING",
                    {"default": "", "multiline": True, "forceInput": True},
                ),
            },
        }

    def preview_markdown(self, source: str):
        return {
            "ui": {"markdown": [source]},
        }
