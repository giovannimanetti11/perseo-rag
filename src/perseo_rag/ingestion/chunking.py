from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextChunker:
    max_chars: int = 1200
    min_break_ratio: float = 0.6

    def __post_init__(self) -> None:
        if self.max_chars < 1:
            raise ValueError("max_chars must be positive")

        if not 0 < self.min_break_ratio <= 1:
            raise ValueError("min_break_ratio must be between 0 and 1")

    def split(self, content: str) -> tuple[str, ...]:
        if not content:
            return ()

        chunks: list[str] = []
        start = 0

        while start < len(content):
            hard_end = min(start + self.max_chars, len(content))
            end = self._find_break(content, start, hard_end)
            chunk = content[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start = end
            while start < len(content) and content[start].isspace():
                start += 1

        return tuple(chunks)

    def _find_break(self, content: str, start: int, hard_end: int) -> int:
        if hard_end == len(content):
            return hard_end

        minimum = start + int(self.max_chars * self.min_break_ratio)
        for separator in ("\n\n", "\n", " "):
            position = content.rfind(separator, minimum, hard_end)
            if position >= minimum:
                return position

        return hard_end
