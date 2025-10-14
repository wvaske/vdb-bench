"""Command-line interface for vdbbench."""

import argparse
from typing import List, Optional

from . import compact_and_watch, list_collections, load_vdb, simple_bench


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vdbbench",
        description="Vector database benchmarking utilities for Milvus",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    bench_parser = simple_bench.build_parser(
        subparsers.add_parser(
            "bench",
            help="Run the Milvus query benchmark",
            description="Run the Milvus query benchmark",
        )
    )
    bench_parser.set_defaults(func=lambda args, p=bench_parser: simple_bench.run(args, p))

    load_parser = load_vdb.build_parser(
        subparsers.add_parser(
            "load",
            help="Load vectors into a Milvus collection",
            description="Load vectors into a Milvus collection",
        )
    )
    load_parser.set_defaults(func=lambda args, p=load_parser: load_vdb.run(args, p))

    compact_parser = compact_and_watch.build_parser(
        subparsers.add_parser(
            "compact",
            help="Monitor and optionally run compaction",
            description="Monitor Milvus index build progress and optionally compact a collection",
        )
    )
    compact_parser.set_defaults(func=lambda args, p=compact_parser: compact_and_watch.run(args, p))

    list_parser = list_collections.build_parser(
        subparsers.add_parser(
            "list",
            aliases=["list-collections"],
            help="List collections and display details",
            description="List Milvus collections and display details",
        )
    )
    list_parser.set_defaults(func=list_collections.run)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
