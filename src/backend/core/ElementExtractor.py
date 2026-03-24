import subprocess
import os
import shutil
import glob
import re

from pathlib import Path
from typing import List, Dict, Any, Tuple

from backend import config
from backend.logger import get_file_logger

class ElementExtractor:

    def __init__(self):
        self.parse_bin = os.path.join(config.JOERN_PATH, "joern-parse")
        self.export_bin = os.path.join(config.JOERN_PATH, "joern-export")
        self.logger = get_file_logger("ElementExtractor")
        self.nodes_path = config.NODES_PATH
        self.edges_path = config.EDGES_PATH
        self.next_id = 1

    def extract_elements(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

        if not self.nodes_path.exists() or not self.edges_path.exists():
            self.logger.info("Joern export not found")
            self._execute_joern_pipeline()

        all_nodes = []
        all_edges = []

        # Add repo node
        repo_node = {
            "id": self.next_id,
            "type": "REPO",
            "name": "xv6-riscv",
            "summary": "virtual node for repo",
            "code": "",
            "parent_id": "None"
        }
        all_nodes.append(repo_node)
        self.next_id += 1

        dot_files = glob.glob(os.path.join(config.JOERN_OUT_DIR, "*.dot"))
        self.logger.info(f"Processing {len(dot_files)} dot files...")

        for dotfile in dot_files:
            nodes, edges = self._parse_dot_file(dotfile)
            all_nodes.extend(nodes)
            all_edges.extend(edges)

        return all_nodes, all_edges

    def _execute_joern_pipeline(self):

        src_dir = str(config.REPO_DIR)
        cpg_bin = str(config.CPG_BIN_PATH)
        export_dir = str(config.DATA_DIR)
        
        joern_out_dir = os.path.join(export_dir, "joern_out")
        if os.path.exists(joern_out_dir):
            self.logger.info(f"Removing existing export directory: {joern_out_dir}")
            shutil.rmtree(joern_out_dir)

        try:
            self.logger.info(f"Running joern-parse on {src_dir}...")
            subprocess.run([
                self.parse_bin, src_dir, "--output", cpg_bin
            ], check=True)
            self.logger.info(f"Running joern-export to {joern_out_dir}...")
            subprocess.run([
                self.export_bin,
                cpg_bin,
                "--repr", "cpg14",
                "--format", "dot",
                "--out", joern_out_dir
            ], check=True)
            self.logger.info("Joern export successful.")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Joern command failed: {e}")
            raise

    def _parse_dot_file(self, file_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        nodes = []
        edges = []
        
        node_re = re.compile(r'"(\d+)"\s+\[label\s+=\s+<\(([A-Z_]+),([^>]+)\)(?:<SUB>(\d+)</SUB>)?>>\s+\]')
        edge_re = re.compile(r'"(\d+)"\s+->\s+"(\d+)"\s+\[\s+label\s+=\s+"([^"]+):\s+"\]')

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            for match in node_re.finditer(content):
                orig_id, n_type, n_name, line = match.groups()
                
                nodes.append({
                    "id": self.next_id,
                    "type": n_type,
                    "name": n_name.strip(),
                    "summary": "",
                    "code": "",
                    "parent_id": ""
                })

            for match in edge_re.finditer(content):
                u, v, e_type = match.groups()
                edges.append({
                    "source": "",
                    "target": "",
                    "type": e_type
                })

        return nodes, edges