from dis import dis
from tkinter import W
import torch
import numpy as np
import yaml
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler

from paths import default_data_path
from torch_june import Runner
from model_utils import SeqData

default_june_data_path = default_data_path / "June"
default_june_config_path = default_june_data_path / "june_default.yaml"
default_daily_deaths_filename = default_june_data_path / "deaths_by_lad.csv"
default_mobility_data_filename = default_june_data_path / "london_mobility_data.csv"
default_area_to_district_filename = default_june_data_path / "area_district.csv"


def get_attribute(base, path):
    paths = path.split(".")
    for p in paths:
        base = getattr(base, p)
    return base


def set_attribute(base, path, target):
    paths = path.split(".")
    _base = base
    for p in paths[:-1]:
        _base = getattr(_base, p)
    if type(_base) == dict:
        _base[paths[-1]] = target
    else:
        setattr(_base, paths[-1], target)


class June:
    r"""
    Wrapper around torch_june
    """

    def __init__(self, params, device: str):
        self.runner = Runner.from_file(default_june_config_path)
        self.number_of_districts = None
        self.districts_map = None
        with open(default_june_config_path, "r") as f:
            june_params = yaml.safe_load(f)
            self.parameters_to_calibrate = june_params["parameters_to_calibrate"]

    @property
    def device(self):
        return self.runner.device

    def _assign_district_to_agents(self, district_data):
        district_ids = district_data.area_to_district.loc[
            self.runner.data["agent"].area, "id"
        ].values
        district_nums = torch.arange(0, np.unique(district_ids).shape[0])
        ret_idcs = np.searchsorted(np.sort(np.unique(district_ids)), district_ids)
        ret = district_nums[ret_idcs]
        self.runner.data["agent"].district = ret
        self.number_of_districts = self.runner.data["agent"].district.unique().shape[0]
        self.districts_map = np.sort(np.unique(district_ids))
        print("NUMBER OF DISTRICTS")
        print(self.runner.data["agent"].district.unique().shape[0])

    def _set_param_values(self, param_values):
        param_values = param_values.flatten()
        for param_name, param_value in zip(self.parameters_to_calibrate, param_values):
            set_attribute(self.runner.model, param_name, param_value)

    def _get_deaths_per_week(self):
        deaths_by_district_timestep = self.runner.data["results"][
            "daily_deaths_by_district"
        ].transpose(0, 1)
        deaths_cumsum = deaths_by_district_timestep.cumsum(1)
        mask = torch.zeros(deaths_cumsum.shape[1], dtype=torch.long)
        mask[::7] = 1
        mask = mask.to(torch.bool)
        ret = deaths_cumsum[:, mask]
        ret = torch.diff(ret, prepend=torch.zeros(ret.shape[0], 1), dim=1)
        return ret

    def step(self, param_values):
        self._set_param_values(param_values)
        results, _ = self.runner()
        predictions = self._get_deaths_per_week()
        return predictions


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
        features_mobility = (
            self.weekly_mobility_data.loc[
                self.weekly_mobility_data.district_id == district
            ]
            .drop(columns="district_id")
            .iloc[week_1:week_2]
            .values
        )
        features_deaths = (
            self.weekly_deaths.loc[self.weekly_deaths.district_id == district]
            .drop(columns="district_id")
            .iloc[week_1:week_2]
            .values.flatten()
        )
        if len(features_deaths) == 0:
            features_deaths = np.zeros(week_2 - week_1)
        features = np.concatenate(
            (features_mobility, features_deaths.reshape(-1, 1)), axis=-1
        )
        targets = features_deaths
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

    def get_train_data(self, number_of_weeks: int, districts):
        # districts = np.sort(self.weekly_mobility_data.district_id.unique())
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

    def prepare_data_for_training(self, number_of_weeks: int, districts_map):
        """
        Prepare train and validation dataset
        """
        metadata = self.get_static_metadata()
        c_seqs_norm, c_ys = self.get_train_data(number_of_weeks, districts_map)
        all_counties = np.sort(self.daily_deaths.district_id.unique())
        min_sequence_length = 5
        metas, seqs, y, y_mask = [], [], [], []
        for meta, seq, ys in zip(metadata, c_seqs_norm, c_ys):
            seq, ys, ys_mask = self.create_window_seqs(seq, ys, min_sequence_length)
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
        y_train = y_train.unsqueeze(2)

        train_dataset = SeqData(
            counties_train, metas_train, X_train, y_train, y_mask_train
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=X_train.shape[0], shuffle=True
        )

        assert all_county_seqs.shape[1] == all_county_ys.shape[1]
        seqlen = all_county_seqs.shape[1]
        return train_loader, metas_train.shape[1], X_train.shape[2], seqlen
