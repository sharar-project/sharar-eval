from evals import hard_eval
from evals.ifbench import instructions_registry


def test_registry_holds_the_58_ood_verifiers():
    assert len(instructions_registry.INSTRUCTION_DICT) == 58
    assert not any(k.startswith(("keywords:", "detectable_format:")) for k in instructions_registry.INSTRUCTION_DICT)


def test_ifbench_scores_loose_and_records_strict():
    it = {"prompt": "Describe a fox. Write the entire response in title case.", "instruction_id_list": ["format:title_case"],
          "kwargs": [{"N": None}]}
    lead_in = "here is my answer:\nThe Quick Brown Fox Jumps Over The Lazy Dog"
    assert hard_eval.ifeval_grade(it, lead_in, "ifbench") == {"correct": True, "strict": False, "instructions": 1,
                                                              "instructions_followed": 1}
    assert hard_eval.ifeval_grade(it, "the quick brown fox", "ifbench")["correct"] is False
    assert hard_eval.ifeval_grade(it, "", "ifbench")["correct"] is False
