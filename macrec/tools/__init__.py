from macrec.tools.base import Tool
from macrec.tools.summarize import TextSummarizer
from macrec.tools.wikipedia import Wikipedia

TOOL_MAP: dict[str, type] = {
    'summarize': TextSummarizer,
    'wikipedia': Wikipedia,
}