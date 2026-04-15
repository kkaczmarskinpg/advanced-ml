from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def normalize_text(series: pd.Series) -> pd.Series:
    """Normalize text categories to lowercase without extra spaces."""
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def assert_only_expected_values(
    values: pd.Series, expected: set[str], column_name: str, file_path: Path
) -> None:
    """Fail fast when unexpected categories appear in a source column."""
    found = set(values.dropna().unique().tolist())
    unknown = sorted(found - expected)
    if unknown:
        raise ValueError(
            f"Unexpected values in column '{column_name}' for file '{file_path.name}': {unknown}"
        )


def map_single_column(
    df: pd.DataFrame,
    column_name: str,
    mapping: dict[str, int],
    file_path: Path,
) -> None:
    normalized = normalize_text(df[column_name])
    assert_only_expected_values(normalized, set(mapping.keys()), column_name, file_path)

    mapped = normalized.map(mapping)
    if mapped.isna().any():
        raise ValueError(
            f"Mapping produced NaN in column '{column_name}' for file '{file_path.name}'."
        )

    df[column_name] = mapped.astype("int8")


def split_to_binary_columns(
    df: pd.DataFrame,
    source_column: str,
    category_to_column: dict[str, str],
    file_path: Path,
) -> None:
    normalized = normalize_text(df[source_column])
    assert_only_expected_values(
        normalized, set(category_to_column.keys()), source_column, file_path
    )

    for category, out_col in category_to_column.items():
        df[out_col] = (normalized == category).astype("int8")

    df.drop(columns=[source_column], inplace=True)


def transform_dataset(df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    required_columns = {
        "person_gender",
        "person_education",
        "person_home_ownership",
        "loan_intent",
        "previous_loan_defaults_on_file",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in '{file_path.name}': {missing}")

    result = df.copy()

    split_to_binary_columns(
        result,
        source_column="person_gender",
        category_to_column={
            "male": "male",
            "female": "female",
        },
        file_path=file_path,
    )

    map_single_column(
        result,
        column_name="person_education",
        mapping={
            "high school": 0,
            "associate": 1,
            "bachelor": 2,
            "master": 3,
            "doctorate": 4,
        },
        file_path=file_path,
    )

    split_to_binary_columns(
        result,
        source_column="person_home_ownership",
        category_to_column={
            "rent": "rent",
            "mortgage": "mortage",
            "own": "own",
            "other": "other_ownership",
        },
        file_path=file_path,
    )

    split_to_binary_columns(
        result,
        source_column="loan_intent",
        category_to_column={
            "education": "education",
            "medical": "medical",
            "venture": "venture",
            "personal": "personal",
            "debtconsolidation": "debtconsolidation",
            "homeimprovement": "homeimprovement",
        },
        file_path=file_path,
    )

    map_single_column(
        result,
        column_name="previous_loan_defaults_on_file",
        mapping={
            "yes": 1,
            "no": 0,
        },
        file_path=file_path,
    )

    return result


def process_all(input_dir: Path, output_dir: Path, pattern: str) -> None:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files found with pattern '{pattern}' in '{input_dir}'."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Pattern         : {pattern}")
    print(f"Files found     : {len(files)}")

    for file_path in files:
        df = pd.read_csv(file_path)
        transformed = transform_dataset(df, file_path)
        out_path = output_dir / file_path.name
        transformed.to_csv(out_path, index=False)

        print(
            f"- {file_path.name}: rows={len(df)}, cols_before={df.shape[1]}, "
            f"cols_after={transformed.shape[1]} -> {out_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split and encode qualitative columns for all loan_data*.csv datasets "
            "from data/imputed_and_scaled into data/splited_qualitive."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/imputed_and_scaled"),
        help="Directory with source CSV files (default: data/imputed_and_scaled)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splited_qualitive"),
        help="Directory for transformed CSV files (default: data/splited_qualitive)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="loan_data*.csv",
        help="Glob pattern for source files (default: loan_data*.csv)",
    )
    args = parser.parse_args()

    process_all(args.input_dir, args.output_dir, args.pattern)


if __name__ == "__main__":
    main()