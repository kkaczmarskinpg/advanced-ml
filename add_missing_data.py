import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def is_missing(series: pd.Series) -> pd.Series:
	"""Treat NaN and empty/whitespace-only strings as missing."""
	missing = series.isna()
	if pd.api.types.is_string_dtype(series) or series.dtype == object:
		missing = missing | series.astype(str).str.strip().eq("")
	return missing


def add_missing_per_column(
	df: pd.DataFrame, missing_rate: float, seed: int, exclude_columns: set[str]
) -> pd.DataFrame:
	"""Add missing values randomly per column based on dataset row count."""
	if not 0 <= missing_rate <= 1:
		raise ValueError("missing_rate must be between 0 and 1")

	rng = np.random.default_rng(seed)
	n_rows = len(df)
	n_to_add_per_col = int(round(n_rows * missing_rate))

	result = df.copy()

	for col in result.columns:
		if col in exclude_columns:
			continue

		col_missing = is_missing(result[col])
		available_idx = result.index[~col_missing]

		n_to_add = min(n_to_add_per_col, len(available_idx))
		if n_to_add == 0:
			continue

		chosen_idx = rng.choice(available_idx, size=n_to_add, replace=False)
		result.loc[chosen_idx, col] = np.nan

	return result


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Add 10% missing data randomly per column in a CSV file."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("data/loan_data.csv"),
		help="Path to input CSV (default: data/loan_data.csv)",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("data/loan_data_with_missing.csv"),
		help="Path to output CSV (default: data/loan_data_with_missing.csv)",
	)
	parser.add_argument(
		"--rate",
		type=float,
		default=0.10,
		help="Missing rate per column, from 0 to 1 (default: 0.10)",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for reproducible results (default: 42)",
	)
	parser.add_argument(
		"--exclude-cols",
		type=str,
		default="loan_status",
		help="Comma-separated columns to exclude from missing injection (default: loan_status)",
	)

	args = parser.parse_args()

	df = pd.read_csv(args.input)
	exclude_columns = {
		col.strip() for col in args.exclude_cols.split(",") if col.strip()
	}
	result = add_missing_per_column(
		df,
		missing_rate=args.rate,
		seed=args.seed,
		exclude_columns=exclude_columns,
	)
	result.to_csv(args.output, index=False)

	print(f"Input file: {args.input}")
	print(f"Output file: {args.output}")
	print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
	print(f"Requested missing rate per column: {args.rate:.2%}")
	print(f"Excluded columns: {', '.join(sorted(exclude_columns)) if exclude_columns else 'None'}")

	missing_summary = result.isna().mean().sort_values(ascending=False)
	print("\nMissing ratio by column:")
	for col, ratio in missing_summary.items():
		print(f"- {col}: {ratio:.2%}")


if __name__ == "__main__":
	main()
