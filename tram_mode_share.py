"""Illustrative HELMET 5.1 home-work mode-choice scenario.

Run with ``python tram_mode_share.py`` to write a chart and CSV under ``output``.
The model coefficients are from HSL's public HELMET 5.1 source; corridor inputs
are synthetic and are not an official HELMET forecast.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@dataclass(frozen=True)
class DestinationCoefficients:
    time: float = 0.0
    cost: float = 0.0
    log_time: float = 0.0
    parking_cost_work: float = 0.0
    parking_time: float = 0.0


@dataclass(frozen=True)
class ModeCoefficients:
    constant: tuple[float, float]
    logsum: float
    generation: Mapping[str, float | tuple[float, float]]
    destination: DestinationCoefficients
    car_user_dummy: tuple[float, float] = (0.0, 0.0)


MODE_COEFFICIENTS = {
    "car": ModeCoefficients(
        constant=(0.0, 0.16722762461943327),
        logsum=0.884331048466,
        generation={
            "car_density": (1.7013066105007886, 1.7013066105007886),
            "cbd": -0.5686410095478218,
            "helsinki_other": -0.13562834728133313,
            "espoo_vant_kau": -0.027600855630398227,
        },
        destination=DestinationCoefficients(
            time=-0.02115590561523804,
            cost=-0.13023506941,
            parking_cost_work=-0.285023839582,
            parking_time=-0.02115590561523804,
        ),
        car_user_dummy=(3.614837678535191, 3.614837678535191),
    ),
    "transit": ModeCoefficients(
        constant=(3.8193799368103605, 3.585180484302631),
        logsum=0.884331048466,
        generation={
            "cbd": 0.10117819121345614,
            "helsinki_other": 0.48575529407353774,
            "espoo_vant_kau": 0.05714665820622189,
        },
        destination=DestinationCoefficients(
            time=-0.0175895053671,
            cost=-0.13023506941,
            parking_cost_work=0.102034422069,
        ),
    ),
    "bike": ModeCoefficients(
        constant=(3.163321706084375, 4.035304380321256),
        logsum=0.884331048466,
        generation={
            "cbd": -0.20956496106611253,
            "helsinki_other": 0.3621616833962815,
            "espoo_vant_kau": 0.03617047904915257,
        },
        destination=DestinationCoefficients(
            time=-0.0343718987488,
            log_time=-0.091425785374,
        ),
    ),
    "walk": ModeCoefficients(
        constant=(4.9065912897381, 5.229553351275655),
        logsum=0.884331048466,
        generation={
            "cbd": 0.009559036798823438,
            "helsinki_other": -0.004462767064854267,
            "espoo_vant_kau": 0.33614807730599455,
        },
        destination=DestinationCoefficients(
            time=-0.0449391361826,
            log_time=-0.143557319072,
        ),
    ),
}

WORKPLACE_SIZE_COEFFICIENT = 1.0
SHOP_SIZE_COEFFICIENT = 0.5534880822318661


@dataclass(frozen=True)
class CorridorInputs:
    trips_per_hour: float = 2400
    car_demand_veh_per_hour_per_direction: float = 3000.0
    distance_km: float = 10.0
    car_free_flow_speed_kmh: float = 60.0
    transit_speed_kmh: float = 35.0
    road_lanes: int = 4
    lane_capacity_veh_per_hour: float = 1800.0
    bpr_alpha: float = 0.15
    bpr_beta: float = 4.0
    car_occupancy: float = 1.2
    home_subregion: int = 0
    car_users_share: float = 0.5
    origin_features: Mapping[str, float] | None = None
    workplace_index: float = 1.0
    shop_index: float = 1.0
    parking_cost_work: float = 0.0
    parking_time: float = 0.0
    car_cost: float = 4.0
    transit_cost: float = 3.0
    bike_time: float = 180.0
    walk_time: float = 240.0

    def __post_init__(self) -> None:
        if self.trips_per_hour <= 0:
            raise ValueError("trips_per_hour must be positive")
        if self.car_demand_veh_per_hour_per_direction <= 0:
            raise ValueError("directional car demand must be positive")
        if self.distance_km <= 0:
            raise ValueError("distance_km must be positive")
        if self.car_free_flow_speed_kmh <= 0 or self.transit_speed_kmh <= 0:
            raise ValueError("speeds must be positive")
        if self.road_lanes <= 0 or self.road_lanes % 2:
            raise ValueError("road_lanes must be a positive even number")
        if self.lane_capacity_veh_per_hour <= 0 or self.car_occupancy <= 0:
            raise ValueError("lane capacity and car occupancy must be positive")
        if self.bpr_alpha < 0 or self.bpr_beta <= 0:
            raise ValueError("BPR parameters require alpha >= 0 and beta > 0")
        if self.home_subregion not in (0, 1):
            raise ValueError("home_subregion must be 0 (capital region) or 1")
        if not 0 <= self.car_users_share <= 1:
            raise ValueError("car_users_share must be between 0 and 1")
        if self.workplace_index < 0 or self.shop_index < 0:
            raise ValueError("destination size indices cannot be negative")
        if self.workplace_index + SHOP_SIZE_COEFFICIENT * self.shop_index <= 0:
            raise ValueError("combined destination size must be positive")


@dataclass(frozen=True)
class Scenario:
    name: str
    transit_vehicle: str
    car_time: float
    transit_time: float
    headway: float

    def __post_init__(self) -> None:
        if self.car_time <= 0 or self.transit_time <= 0 or self.headway <= 0:
            raise ValueError("travel times and headway must be positive")
        if self.transit_vehicle not in ("bus", "tram"):
            raise ValueError("transit_vehicle must be 'bus' or 'tram'")

    @property
    def departures_per_hour(self) -> float:
        return 60.0 / self.headway


@dataclass(frozen=True)
class ScenarioResult:
    scenario: Scenario
    shares: Mapping[str, float]
    car_speed_kmh: float
    od_car_flow_veh_per_hour: float
    bus_flow_veh_per_hour: float
    road_flow_veh_per_hour: float
    road_capacity_veh_per_hour: float


def _log1pexp(value: float) -> float:
    if value > 0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def _softmax(utilities: Mapping[str, float]) -> dict[str, float]:
    maximum = max(utilities.values())
    exponentials = {
        mode: math.exp(utility - maximum)
        for mode, utility in utilities.items()
    }
    denominator = sum(exponentials.values())
    return {mode: value / denominator for mode, value in exponentials.items()}


def mode_shares(
    scenario: Scenario, inputs: CorridorInputs
) -> dict[str, float]:
    """Return mode shares using HELMET's HW mode/destination nested-logit form.

    The synthetic corridor has one destination. Park-and-ride is unavailable;
    its utility needs facility and access/egress inputs not present here.
    """
    features = inputs.origin_features or {}
    destination_size = (
        WORKPLACE_SIZE_COEFFICIENT * inputs.workplace_index
        + SHOP_SIZE_COEFFICIENT * inputs.shop_index
    )
    travel_times = {
        "car": scenario.car_time,
        "transit": scenario.transit_time,
        "bike": inputs.bike_time,
        "walk": inputs.walk_time,
    }
    travel_costs = {
        "car": inputs.car_cost,
        "transit": inputs.transit_cost,
        "bike": 0.0,
        "walk": 0.0,
    }

    utilities: dict[str, float] = {}
    for mode, coefficients in MODE_COEFFICIENTS.items():
        destination = coefficients.destination
        time = travel_times[mode]
        destination_utility = (
            destination.time * time
            + destination.cost * travel_costs[mode]
            + destination.log_time * math.log1p(time)
            + destination.parking_cost_work * inputs.parking_cost_work
            + destination.parking_time * inputs.parking_time
            + math.log(destination_size)
        )

        zone_utility = 0.0
        for feature, coefficient in coefficients.generation.items():
            if isinstance(coefficient, tuple):
                coefficient = coefficient[inputs.home_subregion]
            zone_utility += coefficient * features.get(feature, 0.0)

        utilities[mode] = (
            coefficients.constant[inputs.home_subregion]
            + zone_utility
            + coefficients.logsum * _log1pexp(destination_utility)
        )

    regular_choices = _softmax(utilities)
    car_user_utilities = utilities.copy()
    car_user_utilities["car"] += MODE_COEFFICIENTS["car"].car_user_dummy[
        inputs.home_subregion
    ]
    car_user_choices = _softmax(car_user_utilities)
    return {
        mode: (
            (1 - inputs.car_users_share) * regular_choices[mode]
            + inputs.car_users_share * car_user_choices[mode]
        )
        for mode in utilities
    }


def solve_scenario(
    name: str,
    transit_vehicle: str,
    headway: float,
    inputs: CorridorInputs,
) -> ScenarioResult:
    """Calculate mode shares and road speed for an illustrative corridor.

    The car travel-time relationship is a BPR-style illustrative function,
    not HELMET's assignment. Road car demand is an exogenous directional flow;
    HELMET mode-choice person demand is a separate, illustrative OD market.
    Buses follow the scenario headway and use the road only pre-tram.
    """
    car_free_flow_time = inputs.distance_km / inputs.car_free_flow_speed_kmh * 60
    transit_in_vehicle_time = inputs.distance_km / inputs.transit_speed_kmh * 60
    transit_time = transit_in_vehicle_time + headway / 2
    road_capacity = (
        inputs.road_lanes / 2 * inputs.lane_capacity_veh_per_hour
    )
    bus_flow = 60.0 / headway if transit_vehicle == "bus" else 0.0
    road_flow = inputs.car_demand_veh_per_hour_per_direction + bus_flow
    volume_capacity_ratio = road_flow / road_capacity
    car_time = car_free_flow_time * (
        1 + inputs.bpr_alpha * volume_capacity_ratio**inputs.bpr_beta
    )
    scenario = Scenario(
        name=name,
        transit_vehicle=transit_vehicle,
        car_time=car_time,
        transit_time=transit_time,
        headway=headway,
    )
    shares = mode_shares(scenario, inputs)
    od_car_flow = inputs.trips_per_hour * shares["car"] / inputs.car_occupancy
    car_speed = inputs.distance_km / car_time * 60
    return ScenarioResult(
        scenario=scenario,
        shares=shares,
        car_speed_kmh=car_speed,
        od_car_flow_veh_per_hour=od_car_flow,
        bus_flow_veh_per_hour=bus_flow,
        road_flow_veh_per_hour=road_flow,
        road_capacity_veh_per_hour=road_capacity,
    )


def _draw_road(ax: plt.Axes, result: ScenarioResult, before: bool) -> None:
    scenario = result.scenario
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    road_left, road_right = 1.2, 8.8
    road_bottom, lane_height = 1.75, 0.3
    ax.add_patch(Rectangle(
        (road_left, road_bottom), road_right - road_left, 4 * lane_height,
        facecolor="#667085", edgecolor="none", zorder=1,
    ))
    for lane in range(1, 4):
        y = road_bottom + lane * lane_height
        ax.plot((road_left, road_right), (y, y), color="white",
                linewidth=1.2, linestyle="--", zorder=2)
    lane_centers = [road_bottom + (lane + 0.5) * lane_height for lane in range(4)]
    ax.text(5, 3.25, "FOUR-LANE ROAD  |  2 lanes each way",
            ha="center", color="#344054", weight="bold")

    ax.scatter((0.55, 9.45), (2.35, 2.35), s=420,
               c=["#e6a700", "#344054"], zorder=4)
    ax.text(0.55, 2.35, "S", ha="center", va="center", color="white", weight="bold", zorder=5)
    ax.text(9.45, 2.35, "CBD", ha="center", va="center", color="white", weight="bold", fontsize=8, zorder=5)
    ax.text(0.55, 1.88, "SUBURB", ha="center", color="#344054", weight="bold", fontsize=8)

    def draw_car(center_x: float, center_y: float) -> None:
        ax.add_patch(Rectangle(
            (center_x - 0.27, center_y - 0.075), 0.54, 0.15,
            facecolor="#d0d5dd", edgecolor="white", linewidth=0.7, zorder=3,
        ))
        ax.add_patch(Rectangle(
            (center_x - 0.12, center_y - 0.045), 0.24, 0.09,
            facecolor="#98a2b3", edgecolor="none", zorder=4,
        ))

    edge_lane_index = 0
    occupied_lane = edge_lane_index if before else -1
    for lane, center_y in enumerate(lane_centers):
        if lane == occupied_lane:
            continue
        draw_car((2.3, 3.8, 6.2, 7.7)[lane], center_y)

    if before:
        bus_y = lane_centers[occupied_lane]
        ax.add_patch(Rectangle(
            (4.55, bus_y - 0.09), 0.9, 0.18,
            facecolor="#3976af", edgecolor="white", linewidth=0.8, zorder=3,
        ))
        ax.text(5, 0.94, f"Bus in edge lane  |  {scenario.departures_per_hour:g} buses/hour/direction",
                ha="center", color="#245b8f")
    else:
        ax.text(5, 1.35, "TWO-LANE TRAMWAY  |  separate from road",
                ha="center", color="#344054", weight="bold")
        for track_center in (0.65, 1.0):
            ax.plot((road_left, road_right), (track_center - 0.055, track_center - 0.055),
                    color="#3976af", linewidth=1.5)
            ax.plot((road_left, road_right), (track_center + 0.055, track_center + 0.055),
                    color="#3976af", linewidth=1.5)
        tram_center = 0.65
        ax.add_patch(Rectangle(
            (4.45, tram_center - 0.105), 1.1, 0.21,
            facecolor="#3976af", edgecolor="white", linewidth=0.8, zorder=3,
        ))
        ax.text(5, 0.25, f"Tram replaces buses  |  {scenario.departures_per_hour:g} trams/hour/direction",
                ha="center", color="#245b8f")

    ax.set_title(f"{scenario.name}  |  car {result.car_speed_kmh:.2f} km/h",
                 loc="left", fontsize=11, weight="bold", pad=5)


def create_figure(
    results: tuple[ScenarioResult, ScenarioResult],
    inputs: CorridorInputs,
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(12, 8), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(1.1, 1.0))
    before_ax = figure.add_subplot(grid[0, 0])
    after_ax = figure.add_subplot(grid[0, 1])
    shares_ax = figure.add_subplot(grid[1, 0])
    sensitivity_ax = figure.add_subplot(grid[1, 1])

    _draw_road(before_ax, results[0], before=True)
    _draw_road(after_ax, results[1], before=False)
    figure.suptitle("A tram frees road space—but does public-transport mode share change?",
                    fontsize=16, weight="bold")

    colors = {
        "transit": "#3976af",
        "car": "#e6a700",
        "bike": "#56a889",
        "walk": "#98a2b3",
    }
    mode_labels = {"transit": "Public transport", "car": "Car", "bike": "Bike", "walk": "Walk"}
    for row, result in enumerate(results):
        shares = result.shares
        left = 0.0
        for mode in ("transit", "car", "bike", "walk"):
            share = shares[mode]
            shares_ax.barh(
                row, share * 100, left=left * 100, height=0.55,
                color=colors[mode], label=mode_labels[mode] if row == 0 else None,
            )
            if share >= 0.055:
                share_label = (
                    f"{share:.2%}"
                    if mode in ("transit", "car")
                    else f"{share:.0%}"
                )
                shares_ax.text(
                    (left + share / 2) * 100, row, share_label,
                    ha="center", va="center", color="white", fontsize=9, weight="bold",
                )
            left += share
        shares_ax.text(-1.5, row, result.scenario.name, ha="right", va="center")

    shares_ax.set_xlim(0, 100)
    shares_ax.set_yticks([])
    shares_ax.set_xlabel("Share of home–work person trips")
    shares_ax.set_title("Mode choice (illustrative peak-hour OD)", loc="left", weight="bold")
    shares_ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=4, frameon=False)
    shares_ax.grid(axis="x", alpha=0.2)
    shares_ax.set_axisbelow(True)

    demand_max = max(inputs.car_demand_veh_per_hour_per_direction * 2, 6000)
    demands = [demand_max * step / 30 for step in range(1, 31)]
    for result, color in zip(results, ("#667085", colors["transit"])):
        scenario = result.scenario
        shares_by_demand = []
        for demand in demands:
            sensitivity_inputs = replace(
                inputs,
                car_demand_veh_per_hour_per_direction=demand,
            )
            sensitivity_result = solve_scenario(
                name=scenario.name,
                transit_vehicle=scenario.transit_vehicle,
                headway=scenario.headway,
                inputs=sensitivity_inputs,
            )
            shares_by_demand.append(sensitivity_result.shares["transit"] * 100)
        sensitivity_ax.plot(
            demands,
            shares_by_demand,
            color=color,
            linewidth=2.2,
            label=f"{scenario.name} ({scenario.headway:g}-min)",
        )
    sensitivity_ax.axvline(
        inputs.car_demand_veh_per_hour_per_direction,
        color="#d0442c",
        linestyle="--",
                           linewidth=1.5, label="Example demand")
    for result, marker, color in zip(
        results, ("s", "o"), ("#667085", colors["transit"])
    ):
        sensitivity_ax.scatter(
            inputs.car_demand_veh_per_hour_per_direction,
            result.shares["transit"] * 100,
            marker=marker,
            color=color,
            s=45,
            zorder=5,
        )
    pt_change_pp = (
        results[1].shares["transit"] - results[0].shares["transit"]
    ) * 100
    sensitivity_ax.set_xlabel("Road car demand (vehicles/hour/direction)")
    sensitivity_ax.set_ylabel("Public-transport share (%)")
    sensitivity_ax.set_title(
        f"Road-demand sensitivity | PT change at "
        f"{inputs.car_demand_veh_per_hour_per_direction:,.0f}: "
        f"{pt_change_pp:+.2f} pp",
        loc="left",
        weight="bold",
        fontsize=10,
    )
    sensitivity_ax.legend(frameon=False, loc="best")
    sensitivity_ax.grid(alpha=0.2)
    sensitivity_ax.set_axisbelow(True)

    transit_time = (
        inputs.distance_km / inputs.transit_speed_kmh * 60
    )
    car_free_flow_time = (
        inputs.distance_km / inputs.car_free_flow_speed_kmh * 60
    )
    figure.text(
        0.5, -0.015,
        f"OD: {inputs.distance_km:g} km | bus/tram: {inputs.transit_speed_kmh:g} km/h "
        f"({transit_time:.1f} min in-vehicle) | "
        f"headway/wait: bus {results[0].scenario.headway:g}/{results[0].scenario.headway / 2:.1f} min, "
        f"tram {results[1].scenario.headway:g}/{results[1].scenario.headway / 2:.1f} min | "
        f"car: {inputs.car_free_flow_speed_kmh:g} km/h max "
        f"({car_free_flow_time:.1f} min free-flow) | "
        f"{inputs.road_lanes} road lanes; 2 tram tracks | "
        f"{inputs.car_demand_veh_per_hour_per_direction:g} cars/h/direction. "
        "BPR/capacity illustrative; P+R unavailable.",
        ha="center", fontsize=9, color="#475467",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_results(
    results: tuple[ScenarioResult, ScenarioResult],
    inputs: CorridorInputs,
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow((
            "scenario", "mode", "share", "person_trips_per_hour",
            "distance_km", "car_speed_kmh", "car_time_min",
            "transit_speed_kmh", "transit_in_vehicle_time_min",
            "transit_wait_time_min", "transit_total_time_min",
            "road_car_demand_vehicles_per_hour",
            "od_car_vehicles_per_hour", "bus_vehicles_per_hour",
            "total_road_flow_vehicles_per_hour",
            "road_capacity_vehicles_per_hour",
        ))
        for result in results:
            scenario = result.scenario
            for mode in ("transit", "car", "bike", "walk"):
                writer.writerow((
                    scenario.name,
                    mode,
                    f"{result.shares[mode]:.8f}",
                    f"{result.shares[mode] * inputs.trips_per_hour:.2f}",
                    f"{inputs.distance_km:.2f}",
                    f"{result.car_speed_kmh:.3f}",
                    f"{scenario.car_time:.3f}",
                    f"{inputs.transit_speed_kmh:.2f}",
                    f"{inputs.distance_km / inputs.transit_speed_kmh * 60:.3f}",
                    f"{scenario.headway / 2:.3f}",
                    f"{scenario.transit_time:.3f}",
                    f"{inputs.car_demand_veh_per_hour_per_direction:.2f}",
                    f"{result.od_car_flow_veh_per_hour:.2f}",
                    f"{result.bus_flow_veh_per_hour:.2f}",
                    f"{result.road_flow_veh_per_hour:.2f}",
                    f"{result.road_capacity_veh_per_hour:.2f}",
                ))


def main() -> None:
    inputs = CorridorInputs(
        origin_features={
            "car_density": 0.0,
            "cbd": 0.0,
            "helsinki_other": 1.0,
            "espoo_vant_kau": 0.0,
        },
        workplace_index=100.0,
    )
    results = (
        solve_scenario("Before: bus", "bus", 3.0, inputs),
        solve_scenario("After: tram", "tram", 10.0, inputs),
    )

    output_dir = Path(__file__).resolve().parent / "output"
    figure_path = output_dir / "tram_mode_share.png"
    csv_path = output_dir / "mode_share_results.csv"
    create_figure(results, inputs, figure_path)
    write_results(results, inputs, csv_path)

    before_pt = results[0].shares["transit"]
    after_pt = results[1].shares["transit"]
    print(f"Before PT share: {before_pt:.4%} ({before_pt * inputs.trips_per_hour:.2f} trips/hour)")
    print(f"After PT share:  {after_pt:.4%} ({after_pt * inputs.trips_per_hour:.2f} trips/hour)")
    print(
        f"Change:          {(after_pt - before_pt) * 100:+.4f} percentage points "
        f"({(after_pt - before_pt) * inputs.trips_per_hour:+.3f} trips/hour)"
    )
    for result in results:
        print(
            f"{result.scenario.name}: car {result.car_speed_kmh:.3f} km/h, "
            f"{result.scenario.car_time:.4f} min; "
            f"PT generalized time {result.scenario.transit_time:.2f} min"
        )
    print(f"Wrote {figure_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
