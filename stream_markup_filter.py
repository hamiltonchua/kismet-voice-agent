"""Streaming filter for hidden assistant markup blocks.

Suppresses <thinking>...</thinking> blocks always and optionally
<canvas ...>...</canvas> blocks while handling token-split tags.
"""


class StreamMarkupFilter:
    def __init__(self, canvas_enabled: bool):
        self.canvas_enabled = canvas_enabled
        self.in_canvas_block = False
        self.in_thinking_block = False
        self.pending = ""

    @staticmethod
    def _carry_suffix(text: str, marker: str) -> str:
        if not text:
            return ""
        keep = min(len(text), max(0, len(marker) - 1))
        return text[-keep:]

    @staticmethod
    def _is_full_or_prefix(candidate: str, marker: str) -> bool:
        return candidate.startswith(marker) or marker.startswith(candidate)

    def feed(self, token: str) -> str:
        if not token:
            return ""

        text = self.pending + token
        self.pending = ""
        out: list[str] = []
        i = 0

        while i < len(text):
            if self.in_thinking_block:
                close = text.find("</thinking>", i)
                if close == -1:
                    self.pending = self._carry_suffix(text[i:], "</thinking>")
                    return "".join(out)
                i = close + len("</thinking>")
                self.in_thinking_block = False
                continue

            if self.canvas_enabled and self.in_canvas_block:
                close = text.find("</canvas>", i)
                if close == -1:
                    self.pending = self._carry_suffix(text[i:], "</canvas>")
                    return "".join(out)
                i = close + len("</canvas>")
                self.in_canvas_block = False
                continue

            next_lt = text.find("<", i)
            if next_lt == -1:
                out.append(text[i:])
                i = len(text)
                continue

            if next_lt > i:
                out.append(text[i:next_lt])
                i = next_lt

            candidate = text[i:]

            if self._is_full_or_prefix(candidate, "<thinking>"):
                if candidate.startswith("<thinking>"):
                    self.in_thinking_block = True
                    i += len("<thinking>")
                    continue
                self.pending = candidate
                return "".join(out)

            if self.canvas_enabled and self._is_full_or_prefix(candidate, "<canvas"):
                if candidate.startswith("<canvas"):
                    close = text.find(">", i)
                    if close == -1:
                        self.pending = candidate
                        return "".join(out)
                    self.in_canvas_block = True
                    i = close + 1
                    continue
                self.pending = candidate
                return "".join(out)

            out.append("<")
            i += 1

        return "".join(out)

    def flush(self) -> str:
        if self.in_thinking_block or self.in_canvas_block:
            self.pending = ""
            return ""

        out = self.pending
        self.pending = ""
        return out
