import logging
from pathlib import Path

from corerl.data_pipeline.db.data_reader import DataReader, TagStats
from lib_config.loader import load_config

from coreoffline.utils.config import LoadDataConfig

log = logging.getLogger(__name__)


def build_tag_config(
    tag: TagStats,
    is_action: bool,
    operating_percentiles: tuple[float, float],
    expected_percentiles: tuple[float, float],
):
    """Builds tag configuration from tag statistics and percentile ranges."""
    low_op, high_op = operating_percentiles
    low_exp, high_exp = expected_percentiles

    # Get percentile values from the percentiles dictionary
    op_low_val = tag.percentiles.get(low_op) if tag.percentiles else None
    op_high_val = tag.percentiles.get(high_op) if tag.percentiles else None
    exp_low_val = tag.percentiles.get(low_exp) if tag.percentiles else None
    exp_high_val = tag.percentiles.get(high_exp) if tag.percentiles else None

    # Use percentiles if available, otherwise fallback to min/max
    if op_low_val is not None and op_high_val is not None:
        operating_range = f"[{op_low_val:.2f}, {op_high_val:.2f}]"
    else:
        operating_range = f"[{tag.min}, {tag.max}]"

    if exp_low_val is not None and exp_high_val is not None:
        expected_range = f"[{exp_low_val:.2f}, {exp_high_val:.2f}]"
    else:
        expected_range = f"[{tag.min}, {tag.max}]"

    # Include count as a comment
    count_comment = f"  # non-null values: {tag.count:,}" if tag.count is not None else ""

    # Build config lines with proper indentation
    lines = [
        f"  - name: {tag.tag}{count_comment}",
        f"    operating_range: {operating_range}",
        f"    expected_range: {expected_range}",
    ]

    if is_action:
        lines.append("    type: ai_setpoint")

    return lines


def generate(cfg: LoadDataConfig):
    """Generate tag configurations from database statistics."""
    # Get all tags from config (same pattern as ingest_csv.py)
    to_generate = cfg.reward_tags + cfg.action_tags + cfg.input_tags

    # Create a DataReader using the database config from LoadDataConfig
    data_reader = DataReader(cfg.data_writer)

    # Get percentiles configuration from config
    operating_percentiles = cfg.operating_range_percentiles
    expected_percentiles = cfg.expected_range_percentiles

    # Collect all unique percentiles needed
    all_percentiles = set()
    all_percentiles.update(operating_percentiles)
    all_percentiles.update(expected_percentiles)
    percentiles_list = sorted(all_percentiles)

    # Get tag statistics with percentiles
    stats = [
        data_reader.get_tag_stats(tag_name, percentiles_list)
        for tag_name in to_generate
    ]

    with Path('tags.yaml').open('w', encoding='utf-8') as f:
        f.write('tags:\n')
        for tag in stats:
            # in exceptional cases, there is only np.nan data recorded for a tag
            # so both tag.min and tag.max are None. We should just skip these tags
            if tag.min is None:
                continue

            is_action = tag.tag in cfg.action_tags

            tag_cfg = build_tag_config(
                tag,
                is_action,
                operating_percentiles,
                expected_percentiles,
            )
            f.writelines(f"{line}\n" for line in tag_cfg)
            f.write('\n')


@load_config(LoadDataConfig)
def main(cfg: LoadDataConfig):
    """Main function that generates tag configs using the configuration."""
    log.info("=" * 80)
    log.info("Starting tag configuration generation")
    log.info("=" * 80)

    log.info("Generating tag configurations from database statistics...")
    generate(cfg)

    log.info("=" * 80)
    log.info("Tag configuration generation complete!")
    log.info(f"📁 Artifact saved to: {Path('tags.yaml').resolve()}")
    log.info("=" * 80)


if __name__ == '__main__':
    main()
