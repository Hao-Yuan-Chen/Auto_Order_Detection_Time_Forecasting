"""Test cutoff behavior without installing or fitting Prophet."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from functions import forecast_1_hour


class ForecastTests(unittest.TestCase):
    def test_only_prior_observations_and_actual_target_time(self):
        train = pd.DataFrame({"ds": pd.to_datetime(["2024-01-01 00:00"]), "y": [10]})
        test = pd.DataFrame({"ds": pd.to_datetime(["2024-01-01 01:00", "2024-01-01 03:00"]), "y": [20, 999]})
        original_train, original_test = train.copy(deep=True), test.copy(deep=True)
        captured = []

        class FakeProphet:
            def fit(self, frame):
                self.history = frame.copy()

            def predict(self, frame):
                captured.append((self.history, frame.copy()))
                return pd.DataFrame({"yhat": [12.4]})

        with patch.dict(sys.modules, {"prophet": SimpleNamespace(Prophet=FakeProphet)}):
            self.assertEqual(forecast_1_hour(0, train, test), 12)
            self.assertEqual(forecast_1_hour(1, train, test), 12)
            overlapping = train.copy()
            overlapping["ds"] = test["ds"].iloc[0]
            with self.assertRaises(ValueError):
                forecast_1_hour(0, overlapping, test)

        self.assertEqual(captured[0][0]["y"].tolist(), [10])
        self.assertEqual(captured[1][0]["y"].tolist(), [10, 20])
        self.assertEqual(captured[1][1]["ds"].iloc[0], test["ds"].iloc[1])
        self.assertEqual(captured[1][1].columns.tolist(), ["ds"])
        pd.testing.assert_frame_equal(train, original_train)
        pd.testing.assert_frame_equal(test, original_test)

    def test_invalid_index(self):
        for index in (-1, 0):
            with self.assertRaises(IndexError):
                forecast_1_hour(index, pd.DataFrame(), pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
