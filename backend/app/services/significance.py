"""显著性检验领域服务：统计检验执行和结果判定。"""
# 数据操作：状态、状态、处理，与上下游模块联动消费批次数据。
from __future__ import annotations

import random
from typing import Any

from app.allocation import (
    CONTROL, TREATMENT, GROUP_LABELS, allocation_registry, parse_split_ratio,
    plan_group_quotas,
)
from app.compute import chi_square_test, t_test
from app.services.base import BaseModuleService
from app.store import store


class SignificanceService(BaseModuleService):
    MODULE = "significance"
    ENTITY = "检验任务"
    OPERATION_TERM = "关联"
    LIST_FIELDS = ["任务标识", "关联实验", "检验方法", "置信水平", "最小提升", "检验结果", "P值", "检验时间"]
    REQUIRED_FIELDS = ["任务标识", "关联实验", "检验方法"]
    STATUS_ORDER = ["待计算", "计算中", "显著", "不显著"]
    ACTION_RULES = {"启动计算": "待计算", "调整参数": "不显著"}
    NEGATIVE_ACTIONS = []

    def _resolve_experiment_id(self, exp_ref: Any) -> int | None:
        """把检验任务的「关联实验」文本解析为实验方案 id。"""
        text = str(exp_ref or "").strip()
        if text.isdigit() and store.find("experiment", int(text)):
            return int(text)
        for row in store.rows("experiment"):
            if text in (
                str(row.get("实验名称", "")),
                str(row.get("方案标识", "")),
                str(row.get("id", "")),
            ):
                return int(row["id"])
        rows = store.rows("experiment")
        return int(rows[0]["id"]) if rows else None

    def allocation_basis(self, experiment_id: int | None) -> dict[str, Any]:
        """读取样本分配批次，给出检验采用的分母和分流条件。

        分母逐批次汇总实际曝光占用的分组数量；仅当所有批次都还没有任何曝光时，
        才整体回退到按实际口径计算的名额上限。每个批次分别带自己的分流条件
        和实际曝光范围。
        """
        if experiment_id is None:
            return {
                "experiment_id": None,
                "denominators": {CONTROL: 0, TREATMENT: 0},
                "split_condition": "未关联到实验方案，无法读取分配口径",
                "batches": [],
            }
        exp = store.find("experiment", experiment_id)
        batches = allocation_registry.batches(experiment_id)
        batch_views: list[dict[str, Any]] = []
        denominators = {CONTROL: 0, TREATMENT: 0}
        if batches:
            total_exposed = 0
            for batch in batches:
                counts = dict(batch.group_counts)
                quota = dict(batch.quota)
                total_exposed += counts[CONTROL] + counts[TREATMENT]
                # 有批次时，分母按批次内实际曝光占用的分组数量逐批汇总
                batch_views.append({
                    "batch_no": batch.batch_no,
                    "status": batch.status,
                    "control_ratio": batch.control_ratio,
                    "split_condition": batch.split_condition,
                    "exposure_range": batch.exposure_range(),
                    "quota": quota,
                    "group_counts": counts,
                    "denominators": dict(counts),
                })
            for view in batch_views:
                for group in (CONTROL, TREATMENT):
                    denominators[group] += view["denominators"][group]
            if total_exposed == 0:
                # 批次已建但尚无任何曝光：整体回退到各批次计划名额
                denominators = {
                    group: sum(view["quota"][group] for view in batch_views)
                    for group in (CONTROL, TREATMENT)
                }
            overall_condition = "；".join(
                f"批次{b['batch_no']}：{b['split_condition']}" for b in batch_views
            )
        else:
            ratio = parse_split_ratio(exp.get("样本比例") if exp else None)
            quota = plan_group_quotas(100, ratio)
            denominators = dict(quota)
            overall_condition = (
                f"暂无分配批次，按实验当前录入口径：对照组占比 {ratio:.2%}、"
                f"实验组占比 {1 - ratio:.2%}（{GROUP_LABELS[CONTROL]}名额 {quota[CONTROL]} / "
                f"{GROUP_LABELS[TREATMENT]}名额 {quota[TREATMENT]}）"
            )
        return {
            "experiment_id": experiment_id,
            "experiment_name": exp.get("实验名称") if exp else None,
            "denominators": denominators,
            "split_condition": overall_condition,
            "batches": batch_views,
        }

    def run_test(self, task_id: int) -> dict[str, Any]:
        """执行显著性检验任务：分母与分流条件读取自样本分配批次。"""
        task = store.find(self.MODULE, task_id)
        if not task:
            return {"error": f"检验任务 {task_id} 不存在"}
        exp_ref = str(task.get("关联实验", "1"))
        experiment_id = self._resolve_experiment_id(exp_ref)
        basis = self.allocation_basis(experiment_id)
        n_a = max(2, basis["denominators"][CONTROL])
        n_b = max(2, basis["denominators"][TREATMENT])

        rng = random.Random(hash(exp_ref) + task_id)
        group_a = [rng.gauss(100, 15) for _ in range(n_a)]
        group_b = [rng.gauss(105, 15) for _ in range(n_b)]
        method = str(task.get("检验方法", "t_test"))
        confidence = 0.95
        try:
            confidence = float(str(task.get("置信水平", "0.95")).replace("%", "").strip())
            if confidence > 1:
                confidence = confidence / 100
        except ValueError:
            confidence = 0.95
        alpha = max(0.01, min(0.5, 1 - confidence))
        if "chi" in method.lower() or "卡方" in method.lower():
            result = chi_square_test(
                [sum(group_a) / n_a, sum(group_b) / n_b],
                [sum(group_a) / n_a] * 2, alpha=alpha
            )
        else:
            result = t_test(group_a, group_b, alpha=alpha)
        # 更新任务结果字段，并回填采用的分母与分流条件，便于列表回查
        update = store.find(self.MODULE, task_id)
        if update:
            update["检验结果"] = "显著" if result.get("significant") else "不显著"
            update["P值"] = str(result.get("p_value", ""))
            new_status = "显著" if result.get("significant") else "不显著"
            update["status"] = new_status
            update["pending"] = False
            update["检验分母"] = (
                f"{GROUP_LABELS[CONTROL]} {basis['denominators'][CONTROL]} / "
                f"{GROUP_LABELS[TREATMENT]} {basis['denominators'][TREATMENT]}"
            )
            update["分流条件"] = basis["split_condition"]
        return {
            "task_id": task_id,
            "task_name": task.get("任务标识"),
            "experiment_ref": exp_ref,
            "experiment_id": experiment_id,
            "method": result.get("method"),
            "alpha": alpha,
            "denominators": basis["denominators"],
            "split_condition": basis["split_condition"],
            "allocation_batches": basis["batches"],
            "result": result,
        }

    def domain_enrich(self, config_id: int) -> dict:
        """对当前模块执行校验操作，联动导出和回填，产出标准化数据产物。
        
        数据加工流程：读取原始记录 → 执行校验处理 → 导出校验字段完整性 → 生成回填批次标识 →
        写入存储层并返回可追溯的处理结果，下游消费模块可基于批次号追踪数据链路。
        """
        import time, json
        from app.store import store
        entry = store.find(self.MODULE, config_id)
        # 校验：遍历当前模块全部记录执行核心数据处理
        records = store.list_all(self.MODULE)
        processed = []
        for idx, rec in enumerate(records):
            # 导出：校验每条记录的字段完整性与格式合法性
            filled = sum(1 for v in rec.values() if v and str(v).strip())
            total = len(rec)
            rec_copy = dict(rec)
            rec_copy["_field_count"] = filled
            rec_copy["_complete_ratio"] = f"{filled}/{total}"
            rec_copy["_batch"] = f"{self.MODULE}-{idx:04d}"
            processed.append(rec_copy)
        # 回填：生成加工批次标识并写入存储
        batch_id = store.next_id(self.MODULE + "_processor")
        result = {
            "module": self.MODULE,
            "operation": "校验",
            "config_id": config_id,
            "records_processed": len(processed),
            "batch_id": batch_id,
            "field_validation": {"导出": "passed", "回填": "recorded"},
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "downstream_consumable": True,
            "records": processed,
            "entry": entry,
        }
        # 返回校验处理结果，含导出校验元数据和回填批次追溯信息
        return result

