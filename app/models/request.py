from pydantic import BaseModel
from typing import Optional, List


class SearchRequest(BaseModel):
    query: str
    kb_ids: Optional[List[str]] = None
    fd_ids: Optional[List[str]] = None
    topk: Optional[int] = None
    use_rerank: Optional[bool] = True