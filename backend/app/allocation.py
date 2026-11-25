"""实验分流（样本分配）共享口径。

实验效果入口（ExperimentService.compute_effect）与链式评估
（chain_experiment_evaluate）共用同一套规则，避免各自解释「样本比例」：

- parse_split_ratio：统一解析「30%」「0.30」「30」等录入口径为对照比例（0~1）；
- plan_group_quotas：按计划曝光数和对照比例，用最大余额法得出两组名额上限；
- AllocationRegistry：按批次保存分配口径，同一曝光编号重复上报只占一个名额，
  修改分流比例会开启新批次，历史批次保留当时的口径与分组计数，不重新计数。

口径约定：样本比例表示对照组占比，剩余部分归实验组。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CONTROL = "control"
TREATMENT = "treatment"
GROUP_LABELS = {CONTROL: "对照组", TREATMENT: "实验组"}


class AllocationError(ValueError):
    """分流分配领域错误（如名额耗尽、无开放批次）。"""


def parse_split_ratio(raw: Any, default: float = 0.5) -> float:
    """把录入的分流比例统一解析为对照组占比（0~1 的小数）。

    支持：
    - "30%"、"30 %" → 0.3
    - "0.30"、0.3   → 0.3
    - "30"、30      → 0.3（大于 1 的数值按百分数处理）
    - 空值/无法解析  → default
    """
    if raw is None:
        return default
    text = str(raw).strip()
    if not text:
        return default
    percent = "%" in text
    text = text.replace("%", "").strip()
    try:
        value = float(text)
    except ValueError:
        return default
    if percent or value > 1:
        value = value / 100.0
    return max(0.0, min(1.0, value))


def plan_group_quotas(planned_exposures: int, control_ratio: float) -> dict[str, int]:
    """按实际口径计算名额上限和分组数量。

    对照组名额 = round_half_up(计划曝光 × 对照比例)，其余名额归实验组，
    保证两组名额之和恰好等于计划曝光数（最大余额法，余额分给对照组）。
    """
    planned = max(0, int(planned_exposures))
    ratio = max(0.0, min(1.0, float(control_ratio)))
    exact = planned * ratio
    n_control = int(exact + 0.5)
    n_control = max(0, min(planned, n_control))
    n_treatment = planned - n_control
    return {CONTROL: n_control, TREATMENT: n_treatment}


@dataclass
class AllocationBatch:
    """一个分配批次：一次生效的分流口径及其名额占用情况。"""

    batch_no: int
    control_ratio: float
    planned_exposures: int
    quota: dict[str, int]
    status: str = "open"
    # 曝光编号 -> 曝光记录，重复上报幂等
    exposures: dict[str, dict[str, Any]] = field(default_factory=dict)
    group_counts: dict[str, int] = field(default_factory=lambda: {CONTROL: 0, TREATMENT: 0})

    @property
    def split_condition(self) -> str:
        """该批次采用的分流条件（实际口径的可读描述）。"""
        return (
            f"对照组占比 {self.control_ratio:.2%}、实验组占比 {1 - self.control_ratio:.2%}"
            f"（{GROUP_LABELS[CONTROL]}名额 {self.quota[CONTROL]} / "
            f"{GROUP_LABELS[TREATMENT]}名额 {self.quota[TREATMENT]}）"
        )

    def exposure_range(self) -> dict[str, int]:
        """本批次实际曝光范围：已占用名额的最小编号序号到最大编号序号。"""
        if not self.exposures:
            return {"min_seq": 0, "max_seq": 0, "count": 0}
        seqs = [item["seq"] for item in self.exposures.values()]
        return {"min_seq": min(seqs), "max_seq": max(seqs), "count": len(seqs)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_no": self.batch_no,
            "status": self.status,
            "control_ratio": self.control_ratio,
            "treatment_ratio": round(1 - self.control_ratio, 6),
            "planned_exposures": self.planned_exposures,
            "quota": dict(self.quota),
            "group_counts": dict(self.group_counts),
            "exposure_range": self.exposure_range(),
            "split_condition": self.split_condition,
        }


class AllocationRegistry:
    """按实验维护分配批次与曝光名额（进程内单例，内存仓库风格与 store 一致）。"""

    def __init__(self) -> None:
        self._experiments: dict[int, list[AllocationBatch]] = {}
        self._seq = 0

    def reset(self) -> None:
        self._experiments.clear()
        self._seq = 0

    def batches(self, experiment_id: int) -> list[AllocationBatch]:
        return self._experiments.get(experiment_id, [])

    def open_or_init(
        self, experiment_id: int, ratio_raw: Any, planned_exposures: int
    ) -> AllocationBatch:
        """有批次则返回当前开放批次，否则按口径初始化。"""
        batches = self._experiments.setdefault(experiment_id, [])
        if batches:
            return self._open_batch(batches)
        return self._create_batch(experiment_id, ratio_raw, planned_exposures)

    def change_ratio(
        self, experiment_id: int, ratio_raw: Any, planned_exposures: int | None = None
    ) -> AllocationBatch:
        """修改分流比例：封存旧批次并按新口径开启新批次。

        旧批次的口径、名额与分组计数原样保留，不按新比例重新计数。
        """
        batches = self._experiments.setdefault(experiment_id, [])
        for batch in batches:
            batch.status = "closed"
        if planned_exposures is None:
            planned_exposures = batches[-1].planned_exposures if batches else 100
        return self._create_batch(experiment_id, ratio_raw, planned_exposures)

    def report_exposure(self, experiment_id: int, exposure_no: str, ratio_raw: Any = None,
                        planned_exposures: int = 100) -> dict[str, Any]:
        """上报一次曝光并占用一个名额；同一曝光编号重复上报只占一个名额。"""
        batches = self._experiments.setdefault(experiment_id, [])
        if not batches:
            self._create_batch(
                experiment_id,
                0.5 if ratio_raw is None else ratio_raw,
                planned_exposures,
            )
        batch = self._open_batch(batches)
        key = str(exposure_no)
        if key in batch.exposures:
            existing = dict(batch.exposures[key])
            existing["duplicate"] = True
            existing["message"] = "同一曝光编号重复上报，沿用原分组且不新增名额"
            return existing

        used = batch.group_counts[CONTROL] + batch.group_counts[TREATMENT]
        if used >= batch.planned_exposures:
            raise AllocationError(
                f"批次 {batch.batch_no} 的 {batch.planned_exposures} 个名额已用完，"
                "请调整分流比例开启新批次后再上报"
            )
        # 确定性分配：先填满对照组名额，再进入实验组；名额上限来自实际口径
        if batch.group_counts[CONTROL] < batch.quota[CONTROL]:
            group = CONTROL
        else:
            group = TREATMENT
        self._seq += 1
        record = {
            "exposure_no": key,
            "group": group,
            "group_label": GROUP_LABELS[group],
            "batch_no": batch.batch_no,
            "control_ratio": batch.control_ratio,
            "seq": self._seq,
            "duplicate": False,
        }
        batch.exposures[key] = record
        batch.group_counts[group] += 1
        return dict(record)

    def totals(self, experiment_id: int) -> dict[str, Any]:
        """汇总该实验全部批次的计划名额与实际分组计数（各批次口径分别保留）。"""
        batches = self.batches(experiment_id)
        quota = {CONTROL: 0, TREATMENT: 0}
        counts = {CONTROL: 0, TREATMENT: 0}
        planned = 0
        for batch in batches:
            planned += batch.planned_exposures
            for group in (CONTROL, TREATMENT):
                quota[group] += batch.quota[group]
                counts[group] += batch.group_counts[group]
        return {
            "planned_exposures": planned,
            "quota": quota,
            "group_counts": counts,
            "unique_exposures": counts[CONTROL] + counts[TREATMENT],
        }

    def snapshot(self, experiment_id: int) -> dict[str, Any]:
        """输出可回查的分配快照：批次口径、名额、实际曝光范围与分组计数。"""
        batches = self.batches(experiment_id)
        return {
            "experiment_id": experiment_id,
            "batches": [batch.to_dict() for batch in batches],
            "totals": self.totals(experiment_id),
        }

    def plan_snapshot(self, experiment_id: int, ratio_raw: Any, planned_exposures: int) -> dict[str, Any]:
        """实验尚未产生批次时，按当前录入口径即时计算一份计划快照（不落库）。"""
        ratio = parse_split_ratio(ratio_raw)
        quota = plan_group_quotas(planned_exposures, ratio)
        return {
            "experiment_id": experiment_id,
            "batches": [],
            "planned_only": True,
            "control_ratio": ratio,
            "planned_exposures": planned_exposures,
            "quota": quota,
            "group_counts": {CONTROL: 0, TREATMENT: 0},
            "split_condition": (
                f"对照组占比 {ratio:.2%}、实验组占比 {1 - ratio:.2%}"
                f"（{GROUP_LABELS[CONTROL]}名额 {quota[CONTROL]} / "
                f"{GROUP_LABELS[TREATMENT]}名额 {quota[TREATMENT]}）"
            ),
        }

    # ---- 内部工具 ----
    def _create_batch(
        self, experiment_id: int, ratio_raw: Any, planned_exposures: int
    ) -> AllocationBatch:
        batches = self._experiments.setdefault(experiment_id, [])
        ratio = parse_split_ratio(ratio_raw)
        planned = max(0, int(planned_exposures))
        batch = AllocationBatch(
            batch_no=len(batches) + 1,
            control_ratio=ratio,
            planned_exposures=planned,
            quota=plan_group_quotas(planned, ratio),
        )
        batches.append(batch)
        return batch

    @staticmethod
    def _open_batch(batches: list[AllocationBatch]) -> AllocationBatch:
        for batch in batches:
            if batch.status == "open":
                return batch
        raise AllocationError("当前没有开放的分配批次，请先调整分流比例开启新批次")


# 进程内单例，与 app.store.store 的内存仓库风格一致
allocation_registry = AllocationRegistry()
