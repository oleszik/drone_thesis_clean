from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SitlTransport:
    connection_url: str

    @property
    def transport_kind(self) -> str:
        return "sitl"
