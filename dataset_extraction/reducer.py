"""Reduce the full light novel dataset to a balanced sample of about 2,000 rows.

The sample keeps an 80:20 target ratio of novels with and without synopses
and draws a fair share from each genre. Run this file to read
``light_novels.csv`` and write ``reduced_light_novels.csv``.
"""
import pandas as pd

TARGET_SIZE = 2000
SYNOPSIS_RATIO = 0.8  # 80% with synopsis, 20% without
RANDOM_STATE = 42


def has_synopsis(df):
    """Return a boolean mask: True where 'Synopsis' holds non-blank text."""
    return df["Synopsis"].notna() & df["Synopsis"].astype(str).str.strip().ne("")


def unique_genres(genres):
    """Return the sorted set of genre names in a Series of ", "-joined strings."""
    return sorted({g for sublist in genres.fillna("").str.split(", ") for g in sublist if g})


def reduce_dataset(df, target_size=TARGET_SIZE, synopsis_ratio=SYNOPSIS_RATIO, random_state=RANDOM_STATE):
    """Return a genre-balanced sample of ``df`` with at most ``target_size`` unique rows."""
    # Step 1: Remove duplicate entries based on 'ID'
    df = df.drop_duplicates(subset=["ID"], keep="first").copy()

    # Step 2: Ensure fair representation across genres
    df["Genres"] = df["Genres"].fillna("")  # Handle missing genres

    # Sorted so the result does not depend on string hash order between runs
    genres = unique_genres(df["Genres"])

    # Split dataset into those with and without synopses
    df_with_synopsis = df[has_synopsis(df)]

    # Initialize final dataset
    parts = []

    # Step 3: Distribute samples fairly across genres
    per_genre_limit = target_size // len(genres) if genres else 0
    with_synopsis_limit = int(per_genre_limit * synopsis_ratio)
    without_synopsis_limit = per_genre_limit - with_synopsis_limit
    for genre in genres:
        genre_df = df[df["Genres"].str.contains(genre, na=False, regex=False)]

        # Separate those with and without synopsis
        mask = has_synopsis(genre_df)
        genre_with_synopsis = genre_df[mask]
        genre_without_synopsis = genre_df[~mask]

        # Sample based on limits (avoid exceeding available count)
        parts.append(genre_with_synopsis.sample(n=min(len(genre_with_synopsis), with_synopsis_limit), random_state=random_state))
        parts.append(genre_without_synopsis.sample(n=min(len(genre_without_synopsis), without_synopsis_limit), random_state=random_state))

    final_df = pd.concat(parts) if parts else df.iloc[0:0]
    # A novel with several genres can be drawn more than once; keep one copy
    final_df = final_df.drop_duplicates(subset=["ID"], keep="first")

    # Step 4: Fill remaining slots to get as close to the target as possible
    remaining_slots = target_size - len(final_df)
    if remaining_slots > 0:
        pool = df_with_synopsis[~df_with_synopsis["ID"].isin(final_df["ID"])]
        additional_samples = pool.sample(n=min(len(pool), remaining_slots), random_state=random_state)
        final_df = pd.concat([final_df, additional_samples])

    # Ensure final dataset is within target size
    return final_df.sample(n=min(len(final_df), target_size), random_state=random_state).reset_index(drop=True)


def main(input_path="light_novels.csv", output_path="reduced_light_novels.csv", target_size=TARGET_SIZE):
    """Read ``input_path``, reduce it and write ``output_path``."""
    df = pd.read_csv(input_path)
    final_df = reduce_dataset(df, target_size=target_size)

    # Save to CSV
    final_df.to_csv(output_path, index=False)

    # Output final dataset size
    print(f"Final dataset size: {len(final_df)} (Target: {target_size})")
    return final_df


if __name__ == "__main__":
    main()
