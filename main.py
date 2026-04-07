import hydra
import logging
import os
import sys
from pathlib import Path

ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    # Hydra chdir's to output dir; add project root to sys.path
    sys.path.insert(0, ROOT_DIR)

    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")

    heu_llm = hydra.utils.instantiate(cfg.heu)     # task-level heuristic LLM
    meta_llm = hydra.utils.instantiate(cfg.meta)    # meta-optimizer LLM
    logging.info(f"Using heuristic LLM: {cfg.heu.model}")
    logging.info(f"Using meta LLM: {cfg.meta.model}")

    from moh import MoH
    optimizer = MoH(cfg, ROOT_DIR, heu_llm=heu_llm, meta_llm=meta_llm)
    optimizer.run_meta_optimizer()


if __name__ == "__main__":
    main()
