"""Run full train-fit mining after the old/new natural universes are complete."""
from prepare_stage1_r26 import OUT
from run_r26_followups import await_artifact,execute


if __name__ == "__main__":
    generators = ["B13","R26-O-SUP/seed13/step178","R26-O-SUP/seed29/step178"]
    for generator in generators:
        await_artifact(OUT / "train_retrieval/feedback" / generator / "RETRIEVAL_RECEIPT.json")
    args = ["--device","cpu"]
    for generator in generators:
        args.extend(["--generator",generator])
    execute("mine_stage1_r26_feedback.py",args,"feedback_mining_OSUP.log")
