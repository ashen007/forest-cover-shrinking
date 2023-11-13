import os
import pandas as pd


def create_mask(root: str) -> pd.DataFrame:
    org_dir_list = os.listdir(root)
    year_list = list(range(1983, 2021, 1))

    availability_mask = {}

    for dir in org_dir_list:
        dir_path = os.path.join(root, dir)
        exist_years = sorted(set(int(file[:4]) for file in os.listdir(dir_path)))
        mask = [1 if year in exist_years else 0 for year in year_list]

        availability_mask[dir] = mask

    return pd.DataFrame(availability_mask, index=year_list).T


def select_years(threshold: int, root: str) -> list:
    year_availability = create_mask(root)
    year_counts = (pd.DataFrame(year_availability.sum(), columns=['count'])
                   .sort_values(by='count', ascending=False))

    return list(year_counts[year_counts > 35].dropna().index)


if __name__ == "__main__":
    print(select_years(40, "../data/raw_data"))
