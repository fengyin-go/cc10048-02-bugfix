"""通用接口契约；模块字段由领域服务持有，不再为每个模块复制一份模型类。"""
from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class PageResult(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int = 1
    size: int = 20


class ActionResult(BaseModel):
    ok: bool
    message: str
    entry: dict[str, Any] | None = None


class EntryPayload(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)
    remark: str | None = None


class ActionPayload(BaseModel):
    action: str
    remark: str | None = None


class AllocationPlanPayload(BaseModel):
    """初始化样本分配批次：分流比例 + 计划曝光数。"""
    split_ratio: Any
    planned_exposures: int = Field(default=100, ge=1)


class ExposureReportPayload(BaseModel):
    """曝光上报：曝光编号必填；同编号重复上报幂等。"""
    exposure_no: str
    planned_exposures: int = Field(default=100, ge=1)


class RatioChangePayload(BaseModel):
    """修改分流比例：旧批次封存，按新口径开启新批次。"""
    split_ratio: Any
    planned_exposures: int | None = Field(default=None, ge=1)
