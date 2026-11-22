# Tram versus bus: HELMET 5.1 mode-choice illustration

Run:

```powershell
python tram_mode_share.py
python -m unittest
```

The script writes `output/tram_mode_share.png` and
`output/mode_share_results.csv`. It uses a 10 km CBD–suburb trip, a 35 km/h
bus/tram speed, and a 60 km/h maximum (free-flow) car speed. Bus headway is
3 minutes before conversion; tram headway is 10 minutes after conversion.
With a simple half-headway wait assumption, transit generalized time is
18.6 minutes before (17.1 in-vehicle + 1.5 wait) and 22.1 minutes after
(17.1 + 5.0 wait). The drawing shows four road lanes (two per direction) and,
after conversion, a separate two-track tramway. Vehicles are centered within
the lanes/tracks.

## What is model-based and what is assumed

The mode and destination utility coefficients, including the home–work
individual car-user adjustment, are taken from HSL's public
[HELMET 5.1.0 source](https://github.com/HSLdevcom/helmet-model-system/tree/v5.1.0):

- [`mode_choice.py`](https://github.com/HSLdevcom/helmet-model-system/blob/v5.1.0/Scripts/parameters/mode_choice.py)
- [`destination_choice.py`](https://github.com/HSLdevcom/helmet-model-system/blob/v5.1.0/Scripts/parameters/destination_choice.py)
- [`logit.py`](https://github.com/HSLdevcom/helmet-model-system/blob/v5.1.0/Scripts/models/logit.py)

For each available mode, the script calculates a single-destination utility
using the HELMET home–work coefficients. It then applies the upper-level mode
utility, including the logsum term, and mixes regular and car-user probabilities
using the supplied car-user share. Park-and-ride is unavailable because this
synthetic corridor has no park-and-ride facility inputs.

The zone features, opportunity indices, 2,400 peak-direction person trips/hour
in the mode-choice OD, 3,000 cars/hour/direction on the road, car occupancy,
and road capacity are illustrative, not local measurements. The 3,000 road
cars/hour is an exogenous total corridor car flow, separate from the smaller
illustrative OD used to calculate mode shares; it is not added to the OD car
trips a second time.
Car time is calculated from the 60 km/h free-flow cap using a BPR-style
volume-delay curve with alpha 0.15, beta 4, two lanes per direction, and a
capacity assumption of 1,800 vehicles/hour/lane. The modelled OD car flow is
reported separately. Buses add one vehicle per scheduled departure to the
3,000-car road flow, while the tram runs on its separate tracks. That gives
3,020 road vehicles/hour before conversion and 3,000 after. Headways are
scenario inputs; service is counted per direction (20 buses/hour before,
6 trams/hour after).

This simplified congestion curve is **not** HELMET's road assignment, and the
script does not model junctions, queue spillback, stops, or bus dwell time.
Consequently, the calculated change in car speed and transit share is a
thought-experiment result, not a local forecast. The destination workplace
index is a synthetic value of 100, not an actual CBD workplace count. Replace
the example inputs with corridor demand, capacity and zone/skimming data before
interpreting absolute mode shares. This is a small, transparent
thought experiment, not a substitute for a full HELMET model run.
