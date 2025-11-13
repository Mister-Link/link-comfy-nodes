class ColorParserNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {
                    "multiline": False,
                    "default": "3883558"
                }),
            }
        }
    RETURN_TYPES = ("INT", "STRING", "STRING")
    RETURN_NAMES = ("24-bit", "hex", "rgb")
    FUNCTION = "parse_color"
    CATEGORY = "utils"
    def parse_color(self, value):
        value = str(value).strip()
        try:
            if value.startswith("#"):
                value_int = int(value[1:], 16)
            elif all(c in "0123456789ABCDEFabcdef" for c in value) and len(value) <= 6:
                value_int = int(value, 16)
            else:
                value_int = int(value)
                if not (0 <= value_int <= 16777215):
                    raise ValueError("Out of 24-bit range")
        except Exception:
            return (0, "Invalid", "0, 0, 0")
        r = (value_int >> 16) & 0xFF
        g = (value_int >> 8) & 0xFF
        b = value_int & 0xFF
        hex_str = f"#{value_int:06X}"
        rgb_str = f"{r}, {g}, {b}"
        return (value_int, hex_str, rgb_str)
NODE_CLASS_MAPPINGS = {
    "Hex or 24 Bit Color": ColorParserNode,
}
