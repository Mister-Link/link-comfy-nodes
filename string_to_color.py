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

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("24-bit", "hex", "red", "green", "blue")

    FUNCTION = "parse_color"
    CATEGORY = "utils"

    def parse_color(self, value):
        # Remove whitespace
        value = str(value).strip()

        try:
            if value.startswith("#"):
                # Hex with #
                value_int = int(value[1:], 16)
            elif all(c in "0123456789ABCDEFabcdef" for c in value) and len(value) <= 6:
                # Hex without #
                value_int = int(value, 16)
            else:
                # Try interpreting as decimal 24-bit integer
                value_int = int(value)
                if not (0 <= value_int <= 16777215):
                    raise ValueError("Out of 24-bit range")
        except Exception:
            return ("Invalid", "Invalid", 0, 0, 0)

        # Convert to RGB
        r = (value_int >> 16) & 0xFF
        g = (value_int >> 8) & 0xFF
        b = value_int & 0xFF

        hex_str = f"#{value_int:06X}"
        dec_str = str(value_int)

        return (dec_str, hex_str, r, g, b)


NODE_CLASS_MAPPINGS = {
    "Hex or 24 Bit Color": ColorParserNode,
}
