import unittest

from tram_mode_share import CorridorInputs, solve_scenario


class ModeShareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = CorridorInputs(
            origin_features={
                "car_density": 0.0,
                "cbd": 0.0,
                "helsinki_other": 1.0,
                "espoo_vant_kau": 0.0,
            },
            workplace_index=100.0,
        )
        self.before = solve_scenario("Before", "bus", 3.0, self.inputs)

    def test_mode_shares_sum_to_one(self) -> None:
        shares = self.before.shares

        self.assertAlmostEqual(sum(shares.values()), 1.0)
        self.assertTrue(all(0 <= share <= 1 for share in shares.values()))

    def test_distance_speeds_and_headway_determine_pt_time(self) -> None:
        expected_in_vehicle = 10 / 35 * 60
        expected_wait = 3 / 2

        self.assertAlmostEqual(
            self.before.scenario.transit_time,
            expected_in_vehicle + expected_wait,
        )

    def test_headway_change_alters_waiting_time_and_departures(self) -> None:
        after = solve_scenario("After", "tram", 10.0, self.inputs)

        self.assertAlmostEqual(self.before.scenario.departures_per_hour, 20.0)
        self.assertAlmostEqual(after.scenario.departures_per_hour, 6.0)
        self.assertAlmostEqual(after.scenario.transit_time, 10 / 35 * 60 + 5)

    def test_bus_road_flow_slows_cars_and_reduces_share_after_removal(self) -> None:
        after = solve_scenario("After", "tram", 10.0, self.inputs)

        self.assertLess(self.before.car_speed_kmh, 60.0)
        self.assertGreater(after.car_speed_kmh, self.before.car_speed_kmh)
        self.assertLess(after.shares["transit"], self.before.shares["transit"])

    def test_road_car_demand_and_od_car_flow_are_separate(self) -> None:
        car_volume = (
            self.inputs.trips_per_hour
            * self.before.shares["car"]
            / self.inputs.car_occupancy
        )

        self.assertAlmostEqual(self.before.od_car_flow_veh_per_hour, car_volume)
        self.assertEqual(
            self.inputs.car_demand_veh_per_hour_per_direction,
            3000.0,
        )
        self.assertAlmostEqual(
            self.before.bus_flow_veh_per_hour,
            60 / 3,
        )
        self.assertAlmostEqual(
            self.before.road_flow_veh_per_hour,
            3000 + 60 / 3,
        )


if __name__ == "__main__":
    unittest.main()
