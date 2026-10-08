"""Paquete principal de Gridbreak Chile."""

from __future__ import annotations

import argparse
import logging


def main() -> None:
    """CLI entrypoint para Gridbreak Chile."""
    parser = argparse.ArgumentParser(
        description="Gridbreak Chile - Análisis de Fragilidad Eléctrica RM"
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=["status", "ingest-live", "ingest-benchmark", "app"],
        help="Acción a ejecutar: status, ingest-live, ingest-benchmark, app",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.action == "ingest-live":
        from gridbreak_cl.etl.live_pipeline import run_live_pipeline

        print("⚡ Iniciando ingesta en vivo (SEC + Meteorología + IDW)...")
        df = run_live_pipeline()
        print(f"✅ Ingesta completada con éxito: {len(df)} comunas consolidadas.")
        print(
            f"   Total clientes sin suministro en la RM: {df['clientes_sin_suministro'].sum():,}"
        )
    elif args.action == "ingest-benchmark":
        from gridbreak_cl.config import get_project_root
        from gridbreak_cl.etl.historical_seed import save_benchmark_dataset

        root = get_project_root()
        target = root / "data" / "processed" / "benchmark_temporales_2024.parquet"
        save_benchmark_dataset(target)
        print(f"✅ Benchmark 2024 generado en: {target}")
    elif args.action == "app":
        import subprocess

        print("⚡ Levantando Streamlit app...")
        subprocess.run(["streamlit", "run", "app/streamlit_app.py"], check=True)
    else:
        print("⚡ Gridbreak Chile (CLI)")
        print("Uso:")
        print("  uv run gridbreak-cl ingest-live       # Adquiere telemetría en vivo SEC + DMC")
        print("  uv run gridbreak-cl ingest-benchmark  # Genera dataset benchmark 2024")
        print("  uv run gridbreak-cl app               # Levanta el simulador Streamlit")

