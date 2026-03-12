import argparse

from helpers.utils import VLM_CHOICES, load_vlm


def list_module_names(vlm_name: str, contains: str | None = None) -> list[str]:
    _, _, model = load_vlm(vlm_name)

    names = []
    needle = contains.lower() if contains else None
    for name, _ in model.named_modules():
        if not name:
            continue
        if needle and needle not in name.lower():
            continue
        names.append(name)

    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print module names for a selected VLM to help derive regex layer patterns."
    )
    parser.add_argument("--vlm", required=True, choices=VLM_CHOICES, help="Model choice from helpers.utils.VLM_CHOICES")
    parser.add_argument("--contains", default=None, help="Optional case-insensitive substring filter")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of module names to print (0 = all)")
    args = parser.parse_args()

    module_names = list_module_names(args.vlm, args.contains)

    if args.limit and args.limit > 0:
        module_names = module_names[: args.limit]

    for name in module_names:
        print(name)

    print(f"\n# total_modules={len(module_names)}")


if __name__ == "__main__":
    main()
