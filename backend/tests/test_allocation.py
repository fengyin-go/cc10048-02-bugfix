"""实验分流口径验收测试。

覆盖题面要求：
1. 「30%」与「0.30」在 100 次计划曝光下都得到 30 个对照名额、70 个实验名额；
2. 实验效果入口与链式评估共用同一套比例计算规则和样本分配批次；
3. 显著性检验读取批次的分组与实际曝光范围，给出采用的分母和分流条件；
4. 同一曝光编号重复上报只占一个名额；
5. 第 40 次曝光后改分流比例：旧批次保留旧口径不重新计数，第 41 次进入新批次，
   回查前 40 次分组计数仍可辨认。
"""
import pytest
from fastapi.testclient import TestClient

from app.allocation import (
    CONTROL, TREATMENT, allocation_registry, parse_split_ratio, plan_group_quotas,
)
from app.main import app
from app.services.experiment import ExperimentService
from app.services.significance import SignificanceService
from app.store import store

client = TestClient(app)


@pytest.fixture
def clean_registry():
    allocation_registry.reset()
    yield allocation_registry
    allocation_registry.reset()


@pytest.fixture
def experiment_id(clean_registry):
    """新建一个独立实验，避免污染种子数据。"""
    svc = ExperimentService()
    entry, missing = svc.create_entry({
        "方案标识": "验收实验-A",
        "实验名称": "验收实验A",
        "对照组": "对照组A",
        "实验组": "实验组A",
        "样本比例": "30%",
    })
    assert not missing
    return int(entry["id"])


class TestRatioParsing:
    """共用比例计算规则。"""

    @pytest.mark.parametrize("raw,expected", [
        ("30%", 0.30),
        ("0.30", 0.30),
        ("0.3", 0.30),
        (0.30, 0.30),
        ("30", 0.30),
        (30, 0.30),
        (" 30 % ", 0.30),
        ("50%", 0.50),
        ("0.5", 0.50),
    ])
    def test_parse_equivalent_inputs(self, raw, expected):
        assert parse_split_ratio(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["", None, "无法解析", "abc%"])
    def test_parse_invalid_falls_back_to_half(self, raw):
        assert parse_split_ratio(raw) == 0.5

    def test_ratio_is_clamped(self):
        assert parse_split_ratio("150%") == 1.0
        assert parse_split_ratio("-20%") == 0.0


class TestGroupQuotaPlanning:
    """名额上限和分组数量由实际口径得出。"""

    @pytest.mark.parametrize("raw", ["30%", "0.30", 0.30, "30"])
    def test_100_planned_exposures_30_70(self, raw):
        quota = plan_group_quotas(100, parse_split_ratio(raw))
        assert quota[CONTROL] == 30
        assert quota[TREATMENT] == 70
        assert quota[CONTROL] + quota[TREATMENT] == 100

    def test_half_split_no_longer_default_for_30_percent(self):
        # 修复点：30% 不能再被解释成 50/50
        quota = plan_group_quotas(100, parse_split_ratio("30%"))
        assert quota != {CONTROL: 50, TREATMENT: 50}

    def test_quota_sum_always_equals_planned(self):
        for planned in (1, 7, 33, 100, 250):
            quota = plan_group_quotas(planned, 0.3)
            assert quota[CONTROL] + quota[TREATMENT] == planned

    def test_zero_and_full_ratio(self):
        assert plan_group_quotas(100, 0.0) == {CONTROL: 0, TREATMENT: 100}
        assert plan_group_quotas(100, 1.0) == {CONTROL: 100, TREATMENT: 0}


class TestExposureDedupAndBatches:
    """样本分配批次与名额占用。"""

    def test_duplicate_exposure_consumes_single_slot(self, experiment_id):
        svc = ExperimentService()
        first = svc.report_exposure(experiment_id, "EXP-1")["exposure"]
        second = svc.report_exposure(experiment_id, "EXP-1")["exposure"]
        assert first["duplicate"] is False
        assert second["duplicate"] is True
        assert second["group"] == first["group"]
        assert second["batch_no"] == first["batch_no"]
        snapshot = svc.allocation_snapshot(experiment_id)
        totals = snapshot["totals"]
        assert totals["unique_exposures"] == 1
        assert totals["group_counts"][CONTROL] + totals["group_counts"][TREATMENT] == 1

    def test_groups_fill_against_quota(self, experiment_id):
        svc = ExperimentService()
        for i in range(1, 41):
            svc.report_exposure(experiment_id, f"EXP-{i}")
        snapshot = svc.allocation_snapshot(experiment_id)
        batch = snapshot["batches"][0]
        # 30% 批次先填满 30 个对照名额，再进入实验组
        assert batch["group_counts"] == {CONTROL: 30, TREATMENT: 10}
        assert batch["quota"] == {CONTROL: 30, TREATMENT: 70}
        rng = batch["exposure_range"]
        assert rng["min_seq"] >= 1 and rng["max_seq"] == rng["min_seq"] + 39

    def test_quota_exhaustion_raises_allocation_error(self, experiment_id):
        from app.allocation import AllocationError
        svc = ExperimentService()
        for i in range(100):
            svc.report_exposure(experiment_id, f"EXP-{i}")
        with pytest.raises(AllocationError):
            svc.report_exposure(experiment_id, "EXP-OVERFLOW")


class TestRatioChangeKeepsHistory:
    """第 40 次曝光后改比例：旧批次冻结、新批次接第 41 次。"""

    def _send_40_then_change_to_50(self, svc, experiment_id):
        for i in range(1, 41):
            svc.report_exposure(experiment_id, f"EXP-{i}")
        svc.change_split_ratio(experiment_id, "50%", planned_exposures=100)
        return svc.report_exposure(experiment_id, "EXP-41")

    def test_old_batch_keeps_30_percent_caliber(self, experiment_id):
        svc = ExperimentService()
        self._send_40_then_change_to_50(svc, experiment_id)
        snapshot = svc.allocation_snapshot(experiment_id)
        assert len(snapshot["batches"]) == 2
        old_batch, new_batch = snapshot["batches"]
        assert old_batch["status"] == "closed"
        assert old_batch["control_ratio"] == pytest.approx(0.30)
        assert old_batch["quota"] == {CONTROL: 30, TREATMENT: 70}
        # 回查前 40 次分组计数：旧阶段不能按新比例重新计数
        assert old_batch["group_counts"] == {CONTROL: 30, TREATMENT: 10}

    def test_41st_exposure_landed_in_new_batch(self, experiment_id):
        svc = ExperimentService()
        result = self._send_40_then_change_to_50(svc, experiment_id)
        exposure = result["exposure"]
        assert exposure["batch_no"] == 2
        assert exposure["duplicate"] is False
        snapshot = svc.allocation_snapshot(experiment_id)
        new_batch = snapshot["batches"][1]
        assert new_batch["status"] == "open"
        assert new_batch["control_ratio"] == pytest.approx(0.50)
        assert new_batch["quota"] == {CONTROL: 50, TREATMENT: 50}
        assert new_batch["group_counts"] == {CONTROL: 1, TREATMENT: 0}

    def test_ratio_change_does_not_retroactively_recount(self, experiment_id):
        svc = ExperimentService()
        self._send_40_then_change_to_50(svc, experiment_id)
        # 再上报若干次，旧批次计数必须纹丝不动
        for i in range(42, 51):
            svc.report_exposure(experiment_id, f"EXP-{i}")
        snapshot = svc.allocation_snapshot(experiment_id)
        old_batch, new_batch = snapshot["batches"]
        assert old_batch["group_counts"] == {CONTROL: 30, TREATMENT: 10}
        # 新批次共收到 10 次（41~50），全部计入新口径
        assert sum(new_batch["group_counts"].values()) == 10
        totals = snapshot["totals"]
        assert totals["unique_exposures"] == 50

    def test_each_batch_records_its_own_exposure_range(self, experiment_id):
        svc = ExperimentService()
        self._send_40_then_change_to_50(svc, experiment_id)
        old_batch, new_batch = svc.allocation_snapshot(experiment_id)["batches"]
        assert old_batch["exposure_range"]["count"] == 40
        assert new_batch["exposure_range"]["count"] == 1


class TestBothPanelsShareCaliber:
    """实验效果入口与链式评估必须读出一致的分组人数。"""

    def test_effect_entry_uses_quota_before_any_exposure(self, experiment_id):
        svc = ExperimentService()
        result = svc.compute_effect(experiment_id, planned_exposures=100)
        assert result["sample_sizes"]["control"] == 30
        assert result["sample_sizes"]["treatment"] == 70
        assert result["allocation"]["quota"] == {CONTROL: 30, TREATMENT: 70}

    def test_chain_panel_matches_effect_entry(self, experiment_id):
        from app.routers.chains import chain_experiment_evaluate
        svc = ExperimentService()
        for i in range(1, 41):
            svc.report_exposure(experiment_id, f"EXP-{i}")
        effect = svc.compute_effect(experiment_id)
        chain = chain_experiment_evaluate(experiment_id)
        # 两处入口的实际分组数量一致
        assert effect["allocation"]["group_counts"] == chain["allocation"]["group_counts"]
        assert effect["allocation"]["quota"] == chain["allocation"]["quota"]
        assert chain["descriptive_stats"]["control"]["n"] == effect["sample_sizes"]["control"]
        assert chain["descriptive_stats"]["experiment"]["n"] == effect["sample_sizes"]["treatment"]
        # 链式评估不再出现 50/50 的错误默认值
        assert chain["allocation"]["quota"] != {CONTROL: 50, TREATMENT: 50}

    def test_chain_panel_after_ratio_change_shows_both_batches(self, experiment_id):
        from app.routers.chains import chain_experiment_evaluate
        svc = ExperimentService()
        for i in range(1, 41):
            svc.report_exposure(experiment_id, f"EXP-{i}")
        svc.change_split_ratio(experiment_id, "0.50", planned_exposures=100)
        svc.report_exposure(experiment_id, "EXP-41")
        chain = chain_experiment_evaluate(experiment_id)
        batch_nos = [b["batch_no"] for b in chain["allocation"]["batches"]]
        assert batch_nos == [1, 2]
        # 实际占用：对照 30+1、实验 10+0
        assert chain["allocation"]["group_counts"] == {CONTROL: 31, TREATMENT: 10}
        # 名额上限是两批次口径之和（旧 30/70 + 新 50/50）
        assert chain["allocation"]["quota"] == {CONTROL: 80, TREATMENT: 120}


class TestSignificanceReadsBatches:
    """显著性检验的分母和分流条件来自样本分配批次。"""

    def test_denominators_and_condition_reported(self, experiment_id):
        svc = ExperimentService()
        for i in range(1, 41):
            svc.report_exposure(experiment_id, f"EXP-{i}")
        svc.change_split_ratio(experiment_id, "50%", planned_exposures=100)
        svc.report_exposure(experiment_id, "EXP-41")

        sig_svc = SignificanceService()
        task, missing = sig_svc.create_entry({
            "任务标识": "验收检验-1",
            "关联实验": str(experiment_id),
            "检验方法": "t_test",
            "置信水平": "0.95",
        })
        assert not missing
        result = sig_svc.run_test(int(task["id"]))
        # 分母取实际曝光分组计数：对照 31、实验 10
        assert result["denominators"] == {CONTROL: 31, TREATMENT: 10}
        # 分流条件逐批次给出
        assert "批次1" in result["split_condition"]
        assert "批次2" in result["split_condition"]
        assert "30.00%" in result["split_condition"]
        assert "50.00%" in result["split_condition"]
        assert len(result["allocation_batches"]) == 2
        for batch in result["allocation_batches"]:
            assert batch["exposure_range"]["count"] > 0
            assert batch["denominators"][CONTROL] == batch["group_counts"][CONTROL]

    def test_denominator_falls_back_to_quota_without_exposure(self, experiment_id):
        sig_svc = SignificanceService()
        task, _ = sig_svc.create_entry({
            "任务标识": "验收检验-2",
            "关联实验": str(experiment_id),
            "检验方法": "t_test",
        })
        result = sig_svc.run_test(int(task["id"]))
        # 无曝光时分母回退到计划名额 30/70，而不是随机数或 50/50
        assert result["denominators"] == {CONTROL: 30, TREATMENT: 70}


class TestAllocationEndpoints:
    """HTTP 层端到端验收。"""

    def test_full_40_41_scenario_over_http(self, experiment_id):
        # 初始化 30% 批次（两种录入等价）
        for raw in ("30%", "0.30"):
            resp = client.post(
                f"/api/domain/experiment/{experiment_id}/allocation",
                json={"split_ratio": raw, "planned_exposures": 100},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["quota"] == {CONTROL: 30, TREATMENT: 70}

        # 前 40 次曝光
        for i in range(1, 41):
            resp = client.post(
                f"/api/domain/experiment/{experiment_id}/exposures",
                json={"exposure_no": f"E{i}", "planned_exposures": 100},
            )
            assert resp.status_code == 200

        # 重复上报同一个编号，只占一个名额
        dup = client.post(
            f"/api/domain/experiment/{experiment_id}/exposures",
            json={"exposure_no": "E1", "planned_exposures": 100},
        ).json()
        assert dup["message"] == "重复曝光已忽略"
        assert dup["snapshot"]["totals"]["unique_exposures"] == 40

        # 调整为 50%
        change = client.post(
            f"/api/domain/experiment/{experiment_id}/ratio",
            json={"split_ratio": "50%", "planned_exposures": 100},
        ).json()
        assert change["new_batch"]["batch_no"] == 2
        assert change["new_batch"]["quota"] == {CONTROL: 50, TREATMENT: 50}

        # 第 41 次曝光被新批次接收
        e41 = client.post(
            f"/api/domain/experiment/{experiment_id}/exposures",
            json={"exposure_no": "E41", "planned_exposures": 100},
        ).json()
        assert e41["exposure"]["batch_no"] == 2

        # 两个面板核对
        effect = client.get(
            f"/api/domain/experiment/{experiment_id}/compute-effect"
        ).json()
        chain = client.get(
            f"/api/chains/experiment-evaluate/{experiment_id}"
        ).json()
        assert effect["allocation"]["group_counts"] == {CONTROL: 31, TREATMENT: 10}
        assert chain["allocation"]["group_counts"] == {CONTROL: 31, TREATMENT: 10}
        assert [b["batch_no"] for b in chain["allocation"]["batches"]] == [1, 2]

        # 回查前 40 次分组计数仍是 30/10 的旧口径
        snapshot = client.get(
            f"/api/domain/experiment/{experiment_id}/allocation"
        ).json()
        old_batch = snapshot["batches"][0]
        assert old_batch["group_counts"] == {CONTROL: 30, TREATMENT: 10}
        assert old_batch["control_ratio"] == pytest.approx(0.30)

    def test_endpoint_404_for_unknown_experiment(self, clean_registry):
        resp = client.get("/api/domain/experiment/999999/allocation")
        assert resp.status_code == 404

    def test_quota_exhaustion_returns_409(self, experiment_id):
        for i in range(100):
            client.post(
                f"/api/domain/experiment/{experiment_id}/exposures",
                json={"exposure_no": f"FULL-{i}", "planned_exposures": 100},
            )
        resp = client.post(
            f"/api/domain/experiment/{experiment_id}/exposures",
            json={"exposure_no": "FULL-X", "planned_exposures": 100},
        )
        assert resp.status_code == 409
