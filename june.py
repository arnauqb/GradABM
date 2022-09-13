from dis import dis
import pandas as pd

from paths import default_data_path
from torch_june import Runner

default_june_data_path = default_data_path / "June"
default_june_config_path = default_june_data_path / "june_default.yaml"
default_daily_deaths_filename = default_june_data_path / "deaths_by_lad.csv"
default_mobility_data_filename = default_june_data_path / "london_mobility_data.csv"
default_area_to_district_filename = default_june_data_path / "area_district.csv"


class June:
    r"""
    Wrapper around torch_june
    """

    def __init__(self, params, device):
        self.runner = Runner.from_file(default_june_config_path)


class DistrictData:
    def __init__(
        self,
        initial_day: str,
        daily_deaths: pd.DataFrame,
        mobility_data: pd.DataFrame,
        area_to_district: pd.DataFrame,
    ):
        """
        Handles data at the district level, like deaths per day or mobility data.
        """
        self.initial_day = initial_day
        self.daily_deaths = (
            daily_deaths.set_index("date")
            .fillna(method="ffill")
            .fillna(method="backfill")
            .fillna(0)
        )
        self.weekly_deaths = self._get_weekly_deaths(self.daily_deaths, initial_day)
        self.daily_mobility_data = (
            mobility_data.set_index("date")
            .fillna(method="ffill")
            .fillna(method="backfill")
            .fillna(0)
        )

        self.weekly_mobility_data = self._get_weekly_mobility(
            self.daily_mobility_data, initial_day
        )
        self.area_to_district = area_to_district

    @classmethod
    def from_file(
        cls,
        initial_day: str,
        daily_deaths_filename: str = default_daily_deaths_filename,
        mobility_data_filename: str = default_mobility_data_filename,
        area_to_district_filename: str = default_area_to_district_filename,
    ):
        daily_deaths = pd.read_csv(daily_deaths_filename)
        mobility_data = pd.read_csv(mobility_data_filename)
        area_to_district = pd.read_csv(area_to_district_filename, index_col=0)
        return cls(
            initial_day=initial_day,
            daily_deaths=daily_deaths,
            mobility_data=mobility_data,
            area_to_district=area_to_district,
        )

    def _get_weekly_deaths(self, daily_deaths: pd.DataFrame, initial_day: pd.DataFrame):
        """
        Groups daily deaths by week, summing them.

        Args:
            daily_deaths: dataframe with number of deaths per day and district.
            initial_day: When to start counting weeks.
        """
        deaths_weekly = daily_deaths.copy()
        deaths_weekly = deaths_weekly.reset_index()
        deaths_weekly = deaths_weekly.loc[deaths_weekly.date >= initial_day]
        deaths_weekly["date"] = pd.to_datetime(deaths_weekly["date"]) - pd.to_timedelta(
            7, unit="d"
        )
        deaths_weekly = (
            deaths_weekly.groupby(["district_id", pd.Grouper(key="date", freq="W-MON")])
            .sum()
            .reset_index()
            .sort_values(["date", "district_id"])
            .set_index("date")
        )
        deaths_weekly.rename(columns={"daily_deaths": "weekly_deaths"}, inplace=True)
        return deaths_weekly

    def _get_weekly_mobility(
        self, daily_mobility: pd.DataFrame, initial_day: pd.DataFrame
    ):
        """
        Groups daily mobility by week, taking the average.
        Args:
            daily_mobility: dataframe with mobility reductions per day and district.
            initial_day: When to start counting weeks.
        """
        data_weekly = daily_mobility.copy()
        data_weekly = data_weekly.reset_index()
        data_weekly = data_weekly.loc[data_weekly.date >= initial_day]
        data_weekly["date"] = pd.to_datetime(data_weekly["date"]) - pd.to_timedelta(
            7, unit="d"
        )
        data_weekly = (
            data_weekly.groupby(["district_id", pd.Grouper(key="date", freq="W-MON")])
            .mean()
            .reset_index()
            .sort_values(["date", "district_id"])
            .set_index("date")
        )
        return data_weekly

    def get_data(self, district: int, week_1: int, week_2: int):
        """
        Gets data between week_1 and week_2

        Args:
            district: district id
            week_1: first week
            week_2: last week (included)
        """
        features = (
            self.weekly_mobility_data.loc[
                self.weekly_mobility_data.district_id == district
            ]
            .drop(columns="district_id")
            .iloc[week_1:week_2]
            .values
        )
        targets = (
            self.weekly_deaths.loc[self.weekly_deaths.district_id == district]
            .drop(columns="district_id")
            .iloc[week_1:week_2]
            .values.flatten()
        )
        if len(targets) == 0:
            targets = np.zeros(week_2 - week_1)
        return features, targets

    def get_train_data_district(self, district: int, number_of_weeks: int):
        """
        Gets training data for the specified district from the `initial_day`
        to `initial_day + number_of_weeks`.

        Args:
            district: district id
            number_of_weeks: Number of weeks from initial_day.
        """
        return self.get_data(district, 0, number_of_weeks)

    def get_test_data_district(
        self, district: int, number_of_training_weeks: int, number_of_testing_weeks: int
    ):
        """
        Gets testing data

        Args:
            district: district id
            number_of_weeks: Number of weeks from initial_day.
        """
        return self.get_data(
            district,
            number_of_training_weeks + 1,
            number_of_training_weeks + number_of_testing_weeks + 1,
        )

    def get_train_data(self, number_of_weeks: int):
        districts = np.sort(self.weekly_mobility_data.district_id.unique())
        features = []
        targets = []
        for district in districts:
            district_features, district_targets = self.get_train_data_district(
                district, number_of_weeks
            )
            district_features = StandardScaler().fit_transform(district_features)
            features.append(district_features)
            targets.append(district_targets)

        return np.array(features), np.array(targets)

    def get_static_metadata(self):
        """
        Creates static metadata for each county.
        """
        all_counties = np.sort(self.daily_deaths.reset_index().district_id.unique())
        county_idx = {r: i for i, r in enumerate(all_counties)}
        metadata = np.diag(np.ones(len(all_counties)))
        return metadata

    def create_window_seqs(self, X: np.array, y: np.array, min_sequence_length: int):
        """
        Creates windows of fixed size with appended zeros

        Args:
            X: features
            y: targets, in synchrony with features
                (i.e. x[t] and y[t] correspond to the same time)
        """
        # convert to small sequences for training, starting with length 10
        seqs = []
        targets = []
        mask_ys = []

        # starts at sequence_length and goes until the end
        # for idx in range(min_sequence_length, X.shape[0]+1, 7): # last in range is step
        for idx in range(min_sequence_length, X.shape[0] + 1, 1):
            # Sequences
            seqs.append(torch.from_numpy(X[:idx, :]))
            # Targets
            y_ = y[:idx]
            mask_y = torch.ones(len(y_))
            targets.append(torch.from_numpy(y_))
            mask_ys.append(mask_y)
        seqs = pad_sequence(seqs, batch_first=True, padding_value=0).type(torch.float)
        ys = pad_sequence(targets, batch_first=True, padding_value=-999).type(
            torch.float
        )
        mask_ys = pad_sequence(mask_ys, batch_first=True, padding_value=0).type(
            torch.float
        )

        return seqs, ys, mask_ys

    def prepare_data_for_training(self, number_of_weeks: int, batch_size=1):
        """
        Prepare train and validation dataset
        """
        metadata = self.get_static_metadata()
        c_seqs_norm, c_ys = self.get_train_data(number_of_weeks)
        print("c seqs")
        print(c_seqs_norm.shape)
        raise
        all_counties = np.sort(self.daily_deaths.district_id.unique())
        min_sequence_length = 5
        metas, seqs, y, y_mask = [], [], [], []
        for meta, seq, ys in zip(metadata, c_seqs_norm, c_ys):
            print("-----------")
            print(seq)
            print("//")
            seq, ys, ys_mask = self.create_window_seqs(seq, ys, min_sequence_length)
            print(seq.shape)
            metas.append(meta)
            seqs.append(seq[[-1]])
            y.append(ys[[-1]])
            y_mask.append(ys_mask[[-1]])

        all_metas = np.array(metas, dtype="float32")
        all_county_seqs = torch.cat(seqs, axis=0)
        all_county_ys = torch.cat(y, axis=0)
        all_county_y_mask = torch.cat(y_mask, axis=0)

        counties_train, metas_train, X_train, y_train, y_mask_train = (
            all_counties,
            all_metas,
            all_county_seqs,
            all_county_ys,
            all_county_y_mask,
        )
        print("X train")
        print(X_train.shape)

        train_dataset = SeqData(
            counties_train, metas_train, X_train, y_train, y_mask_train
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        assert all_county_seqs.shape[1] == all_county_ys.shape[1]
        seqlen = all_county_seqs.shape[1]
        return train_loader, metas_train.shape[1], X_train.shape[2], seqlen


def fetch_june_data(initial_day: str, number_of_weeks: int):
    district_data = DistrictData(initial_day=initial_day)
    return district_data.prepare_data_for_training(number_of_weeks=number_of_weeks)
