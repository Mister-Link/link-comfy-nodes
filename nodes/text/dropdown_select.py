from __future__ import annotations

import json


class DropdownSelectNode:
    """Editable dropdown: manage the option list and pick one, output is that string.

    The "options" and "selected" widgets are hidden on the frontend in favor of
    an inline list editor with add/remove rows and a real <select> dropdown
    (see web/dropdownSelect.js). "options" stores the list as JSON.
    """

    CATEGORY = "utils"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("value", "index")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "options": (
                    "STRING",
                    {"default": json.dumps(["Option A", "Option B", "Option C"])},
                ),
                "selected": ("STRING", {"default": "Option A"}),
            }
        }

    def run(self, options: str, selected: str):
        items = self._parse_options(options)
        if selected in items:
            return (selected, items.index(selected))
        if items:
            return (items[0], 0)
        return (selected, -1)

    @staticmethod
    def _parse_options(text: str) -> list[str]:
        text = text.strip()
        if text.startswith("["):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, list):
                return [str(item).strip() for item in data if str(item).strip()]
        return [item.strip() for item in text.split(",") if item.strip()]
