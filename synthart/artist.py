from dataclasses import dataclass, field
from typing import List


@dataclass
class SyntheticArtist:
    """Represents a synthetic artist/system style prompt."""

    name: str
    system_prompt: str
    style_tags: List[str] = field(default_factory=list)

    def build_image_prompt(self, content_prompt: str) -> str:
        """Combine artist style and content into a single prompt for image gen."""
        tags = f"\nStyle tags: {', '.join(self.style_tags)}" if self.style_tags else ""
        return (
            f"Artist style: {self.system_prompt}{tags}\n\n"
            f"Subject: {content_prompt}\n\n"
            f"Render the subject strictly in the artist's style."
        )


