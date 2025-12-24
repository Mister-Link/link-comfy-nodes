"""String manipulation nodes."""


class AdvancedStringConcat:
    """Concatenate strings using template placeholders like %1, %2, etc."""

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
        """Replace %1, %2, etc. in template with corresponding string inputs."""
        result = template

        # Replace placeholders %1 through %8
        for i in range(1, 9):
            string_key = f"string{i}"
            string_value = kwargs.get(string_key, "")
            if string_value:
                result = result.replace(f"%{i}", string_value)

        return (result,)


NODE_CLASS_MAPPINGS = {
    "AdvancedStringConcat": AdvancedStringConcat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedStringConcat": "Advanced String Concat",
}
