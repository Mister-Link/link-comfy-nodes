from __future__ import annotations


class AdvancedStringConcat:
    CATEGORY = "utils"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "concat_strings"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": (
                    "STRING",
                    {"default": "%1 %2", "multiline": True},
                ),
            },
            "optional": {
                "string1": ("STRING", {"default": "", "forceInput": True}),
                "string2": ("STRING", {"default": "", "forceInput": True}),
                "string3": ("STRING", {"default": "", "forceInput": True}),
                "string4": ("STRING", {"default": "", "forceInput": True}),
                "string5": ("STRING", {"default": "", "forceInput": True}),
                "string6": ("STRING", {"default": "", "forceInput": True}),
                "string7": ("STRING", {"default": "", "forceInput": True}),
                "string8": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    def concat_strings(self, template: str, **kwargs):
        result = template
        for i in range(1, 9):
            string_key = f"string{i}"
            string_value = kwargs.get(string_key, "")
            if string_value:
                result = result.replace(f"%{i}", string_value)
        return (result,)
