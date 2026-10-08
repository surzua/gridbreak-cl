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
        choices=[
            "status",
            "ingest-live",
            "ingest-benchmark",
            "build-features",
            "generate-figures",
            "app",
        ],
        help="Acción a ejecutar: status, ingest-live, ingest-benchmark, build-features, generate-figures, app",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Ejecuta en modo offline/mock sin realizar llamadas de red externas",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.action == "ingest-live":
        from gridbreak_cl.etl.live_pipeline import run_live_pipeline

        print(
            f"⚡ Iniciando ingesta en vivo (SEC + Meteorología + IDW) [offline={args.offline}]..."
        )
        df = run_live_pipeline(offline_mock=args.offline)
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
    elif args.action == "build-features":
        from gridbreak_cl.features.build_features import (
            process_and_save_feature_pipeline,
        )

        p1, p2 = process_and_save_feature_pipeline()
        print("✅ Pipeline de features ejecutado exitosamente:")
        print(f"   Panel enriquecido: {p1}")
        print(f"   Supervivencia:     {p2}")
    elif args.action == "generate-figures":
        from gridbreak_cl.visualization.static_charts import (
            generate_all_portfolio_figures,
        )

        figs = generate_all_portfolio_figures()
        print(f"✅ {len(figs)} figuras de alta resolución generadas:")
        for f in figs:
            print(f"   - {f.name}")
    elif args.action == "app":
        import subprocess

        print("⚡ Levantando Streamlit app...")
        subprocess.run(["streamlit", "run", "app/streamlit_app.py"], check=True)
    else:
        print("⚡ Gridbreak Chile (CLI)")
        print("Uso:")
        print(
            "  uv run gridbreak-cl ingest-live [--offline] # Adquiere telemetría en vivo SEC + DMC"
        )
        print(
            "  uv run gridbreak-cl ingest-benchmark       # Genera dataset benchmark 2024"
        )
        print(
            "  uv run gridbreak-cl build-features         # Genera variables temporales y supervivencia"
        )
        print(
            "  uv run gridbreak-cl generate-figures       # Genera figuras PNG 300 DPI de portafolio"
        )
        print(
            "  uv run gridbreak-cl app                    # Levanta el simulador Streamlit"
        )
