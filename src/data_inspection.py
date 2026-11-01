"""
data_inspection.py
==================

Synthetic baseline generator and lightweight inspector for half-hourly
micrometeorological time series, designed to mimic the tower record of the
Reserva Biologica do Jaru (Rondonia, Brazil) before real-data ingestion.

The synthetic record is *physically coupled* rather than a sum of independent
sinusoids:

* Net radiation (R_liq) follows solar geometry (declination, hour angle,
  air-mass attenuation) modulated by an autocorrelated cloud field whose
  climatology is wetter from November to April.
* Air temperature (T_ar_62m) is obtained by integrating a known linear
  relaxation ODE forced by net radiation,

      dT/dt = a * Rn - lam * (T - T_eq(t)),

  integrated with an adaptive solver, so that equation-discovery tools (PySR, AI Feynman, AI-Lorenz-type
  pipelines) can be validated against an exact ground truth.
* Relative humidity (UR_62m) is diagnosed from temperature and a slowly
  varying dew point via the Tetens saturation vapour-pressure relation.
* Wind speed, surface pressure (with the semidiurnal atmospheric tide) and
  intermittent convective precipitation complete the set.

Column names follow the "DADOS OK" sheet of the project spreadsheet
(Ano, Mes, Dia, Hora, R_liq, T_ar_62m, UR_62m, VV_ane_62m, P_atm, PRP).
Noise-free states, the analytic derivative dT/dt and the generating
parameters are stored alongside the noisy observations.

Usage
-----
    python src/data_inspection.py                     # 365 days -> data/synthetic/
    python src/data_inspection.py --days 60 --missing 0.1 --seed 7
    python src/data_inspection.py --inspect "Dados Rebio Jaru 2006 a 2008.xlsx"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann constant (W m-2 K-4)
SOLAR_CONSTANT = 1361.0    # W m-2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SiteConfig:
    """Site and record descriptors (coordinates are approximate)."""

    latitude_deg: float = -10.08
    longitude_deg: float = -61.93
    start: str = "2006-01-01 00:30"
    n_days: int = 365
    dt_minutes: int = 30


@dataclass
class PhysicsConfig:
    """Parameters of the generating model (the 'ground truth')."""

    albedo: float = 0.13                 # tropical forest canopy
    clear_sky_transmissivity: float = 0.75
    cloud_attenuation: float = 0.65      # fractional loss of K_down at full cover
    lw_net_clear: float = -75.0          # net long-wave, clear sky (W m-2)
    lw_cloud_recovery: float = 0.7       # fraction of L* removed at full cover
    # Temperature relaxation ODE: dT/dt = a*Rn - lam*(T - T_eq)
    a_heating: float = 5.5e-3            # K h-1 per W m-2
    lam_relax: float = 0.30              # h-1 (relaxation time ~3.3 h)
    t_eq_mean: float = 23.0              # deg C
    t_eq_seasonal_amp: float = 1.0       # deg C
    # Moisture (dew point) climatology
    td_wet: float = 22.0                 # deg C, wet season
    td_dry: float = 17.5                 # deg C, dry season
    # Wind
    wind_base: float = 1.25              # m s-1
    wind_convective: float = 1.6         # m s-1 added at Rn_max
    # Pressure
    p_mean: float = 987.0                # hPa
    p_seasonal_amp: float = 1.5          # hPa
    p_semidiurnal_amp: float = 1.2       # hPa (S2 tide)
    p_diurnal_amp: float = 0.5           # hPa (S1 tide)
    # Precipitation
    rain_prob_wet: float = 0.09          # per half-hour, afternoon-weighted
    rain_prob_dry: float = 0.008
    rain_mean_mm: float = 4.0


@dataclass
class NoiseConfig:
    """Measurement noise (1 sigma) and data-gap settings."""

    r_liq: float = 15.0
    t_air: float = 0.15
    rh: float = 1.5
    wind: float = 0.25
    p_atm: float = 0.2
    missing_fraction: float = 0.0        # fraction of samples removed in blocks
    mean_gap_steps: int = 12             # mean gap length (6 h)


@dataclass
class GeneratorConfig:
    site: SiteConfig = field(default_factory=SiteConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    seed: int = 42


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def make_time_index(site: SiteConfig) -> pd.DatetimeIndex:
    n = site.n_days * (24 * 60 // site.dt_minutes)
    return pd.date_range(site.start, periods=n, freq=f"{site.dt_minutes}min")


def ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary AR(1) process with marginal standard deviation `sigma`."""
    eps = rng.normal(0.0, sigma * np.sqrt(1.0 - phi**2), n)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def wet_season_weight(doy: np.ndarray) -> np.ndarray:
    """Smooth 0-1 weight: ~1 in Jan-Feb (wet), ~0 in Jul-Aug (dry)."""
    return 0.5 * (1.0 + np.cos(2.0 * np.pi * (doy - 30.0) / 365.25))


def cos_solar_zenith(t: pd.DatetimeIndex, site: SiteConfig) -> np.ndarray:
    """Cosine of the solar zenith angle (local mean solar time, mid-interval)."""
    doy = t.dayofyear.to_numpy()
    utc_offset_h = -4.0  # Amazon time (UTC-4); timestamps are local standard time
    hours = t.hour.to_numpy() + t.minute.to_numpy() / 60.0 - site.dt_minutes / 120.0
    solar_time = hours + (site.longitude_deg / 15.0 - utc_offset_h)
    decl = np.deg2rad(23.44) * np.sin(2.0 * np.pi * (284 + doy) / 365.0)
    lat = np.deg2rad(site.latitude_deg)
    hour_angle = np.deg2rad(15.0 * (solar_time - 12.0))
    return np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)


def cloud_field(t: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Cloud fraction in [0, 1]: seasonal climatology + persistent anomalies."""
    w = wet_season_weight(t.dayofyear.to_numpy())
    clim = 0.15 + 0.45 * w
    anomaly = ar1(len(t), phi=0.97, sigma=0.18, rng=rng)
    return np.clip(clim + anomaly, 0.0, 1.0)


def net_radiation(cosz: np.ndarray, cloud: np.ndarray, p: PhysicsConfig) -> np.ndarray:
    """Rn = (1 - albedo) K_down + L*, with air-mass attenuation and cloud effects."""
    mu = np.clip(cosz, 0.0, None)
    air_mass = np.where(mu > 0.02, 1.0 / np.maximum(mu, 0.02), np.inf)
    k_down = SOLAR_CONSTANT * mu * p.clear_sky_transmissivity ** air_mass
    k_down *= 1.0 - p.cloud_attenuation * cloud
    lw_net = p.lw_net_clear * (1.0 - p.lw_cloud_recovery * cloud)
    return (1.0 - p.albedo) * k_down + lw_net


def integrate_temperature(
    rn: np.ndarray, t_eq: np.ndarray, p: PhysicsConfig, dt_h: float, t0: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate dT/dt = a*Rn(t) - lam*(T - T_eq(t)) with Rn and T_eq interpolated
    by cubic splines, so the sampled dT/dt is the exact derivative of the
    sampled trajectory (no zero-order-hold lag). Returns (T, dT/dt) in K h-1.
    """
    from scipy.integrate import solve_ivp
    from scipy.interpolate import CubicSpline

    t = np.arange(len(rn)) * dt_h
    rn_s, teq_s = CubicSpline(t, rn), CubicSpline(t, t_eq)
    sol = solve_ivp(
        lambda tt, y: p.a_heating * rn_s(tt) - p.lam_relax * (y - teq_s(tt)),
        (t[0], t[-1]), [t0], t_eval=t, method="LSODA", rtol=1e-9, atol=1e-9, max_step=dt_h,
    )
    temp = sol.y[0]
    dtdt = p.a_heating * rn - p.lam_relax * (temp - t_eq)
    return temp, dtdt


def saturation_vapour_pressure(t_c: np.ndarray) -> np.ndarray:
    """Tetens formula, hPa."""
    return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))


def relative_humidity(
    temp: np.ndarray, doy: np.ndarray, p: PhysicsConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    w = wet_season_weight(doy)
    td = p.td_dry + (p.td_wet - p.td_dry) * w + ar1(len(temp), 0.995, 1.0, rng)
    td = np.minimum(td, temp)  # dew point cannot exceed air temperature
    rh = 100.0 * saturation_vapour_pressure(td) / saturation_vapour_pressure(temp)
    return rh, td


def wind_speed(rn: np.ndarray, p: PhysicsConfig, rng: np.random.Generator) -> np.ndarray:
    mixing = np.clip(rn, 0.0, None) / max(rn.max(), 1.0)
    gust = np.abs(ar1(len(rn), 0.8, 0.5, rng))
    return np.clip(p.wind_base + p.wind_convective * mixing + gust - 0.4, 0.0, None)


def surface_pressure(
    t: pd.DatetimeIndex, p: PhysicsConfig, rng: np.random.Generator
) -> np.ndarray:
    doy = t.dayofyear.to_numpy()
    hour = t.hour.to_numpy() + t.minute.to_numpy() / 60.0
    seasonal = p.p_seasonal_amp * np.cos(2.0 * np.pi * (doy - 200.0) / 365.25)
    s2 = p.p_semidiurnal_amp * np.cos(2.0 * np.pi * (hour - 10.0) / 12.0)
    s1 = p.p_diurnal_amp * np.cos(2.0 * np.pi * (hour - 8.0) / 24.0)
    synoptic = ar1(len(t), 0.999, 1.0, rng)
    return p.p_mean + seasonal + s2 + s1 + synoptic


def precipitation(
    t: pd.DatetimeIndex, cloud: np.ndarray, p: PhysicsConfig, rng: np.random.Generator
) -> np.ndarray:
    """Intermittent convective rain, peaking mid-afternoon and under thick cloud."""
    w = wet_season_weight(t.dayofyear.to_numpy())
    hour = t.hour.to_numpy() + t.minute.to_numpy() / 60.0
    diurnal = 1.0 + 0.8 * np.cos(2.0 * np.pi * (hour - 15.0) / 24.0)
    prob = (p.rain_prob_dry + (p.rain_prob_wet - p.rain_prob_dry) * w) * diurnal * cloud
    occurs = rng.random(len(t)) < prob
    return np.where(occurs, rng.exponential(p.rain_mean_mm, len(t)), 0.0)


def insert_gaps(
    df: pd.DataFrame, cols: list[str], noise: NoiseConfig, rng: np.random.Generator
) -> pd.DataFrame:
    """Remove contiguous blocks (instrument outages) until the target fraction is met."""
    if noise.missing_fraction <= 0:
        return df
    n = len(df)
    target = int(noise.missing_fraction * n)
    mask = np.zeros(n, dtype=bool)
    while mask.sum() < target:
        start = rng.integers(0, n)
        length = max(1, int(rng.exponential(noise.mean_gap_steps)))
        mask[start : start + length] = True
    out = df.copy()
    out.loc[mask, cols] = np.nan
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
OBSERVED_COLS = ["R_liq", "T_ar_62m", "UR_62m", "VV_ane_62m", "P_atm", "PRP"]


def generate_synthetic(cfg: GeneratorConfig | None = None) -> pd.DataFrame:
    """Return a half-hourly synthetic record with noisy observations and truths."""
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    s, p, nz = cfg.site, cfg.physics, cfg.noise

    t = make_time_index(s)
    doy = t.dayofyear.to_numpy()
    dt_h = s.dt_minutes / 60.0

    cloud = cloud_field(t, rng)
    rn = net_radiation(cos_solar_zenith(t, s), cloud, p)
    t_eq = p.t_eq_mean + p.t_eq_seasonal_amp * np.cos(2.0 * np.pi * (doy - 250.0) / 365.25)
    temp, dtdt = integrate_temperature(rn, t_eq, p, dt_h, t0=float(t_eq[0]))
    rh, td = relative_humidity(temp, doy, p, rng)
    ws = wind_speed(rn, p, rng)
    pa = surface_pressure(t, p, rng)
    prp = precipitation(t, cloud, p, rng)

    df = pd.DataFrame(
        {
            "timestamp": t,
            "Ano": t.year,
            "Mes": t.month,
            "Dia": doy,
            "Hora": t.hour * 100 + t.minute,
            "R_liq": rn + rng.normal(0, nz.r_liq, len(t)),
            "T_ar_62m": temp + rng.normal(0, nz.t_air, len(t)),
            "UR_62m": np.clip(rh + rng.normal(0, nz.rh, len(t)), 0, 100),
            "VV_ane_62m": np.clip(ws + rng.normal(0, nz.wind, len(t)), 0, None),
            "P_atm": pa + rng.normal(0, nz.p_atm, len(t)),
            "PRP": prp,
            # Noise-free states and the analytic tendency (ground truth)
            "R_liq_true": rn,
            "T_ar_62m_true": temp,
            "T_eq_true": t_eq,
            "dTdt_true_K_per_h": dtdt,
            "T_dew_true": td,
            "cloud_fraction_true": cloud,
        }
    )
    # Midnight is stored as 2400 of the previous day in the tower files
    df.loc[df["Hora"] == 0, "Hora"] = 2400
    return insert_gaps(df, OBSERVED_COLS, nz, rng)


def inspect_dataframe(df: pd.DataFrame, max_cols: int = 40) -> pd.DataFrame:
    """
    Concise, token-cheap structural summary: dtype, missing ratio and moments.
    Intended for both the synthetic baseline and the real spreadsheet.
    """
    num = df.select_dtypes("number")
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().mean().round(3),
        }
    ).join(num.agg(["mean", "std", "min", "max"]).T.round(2), how="left")
    return summary.head(max_cols)


def diurnal_composite(df: pd.DataFrame, cols: list[str], hour_col: str = "Hora") -> pd.DataFrame:
    """Mean diurnal cycle (one row per half-hour slot)."""
    return df.groupby(hour_col)[cols].mean().round(2)


def save_dataset(df: pd.DataFrame, cfg: GeneratorConfig, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"synthetic_jaru_{cfg.site.n_days}d_seed{cfg.seed}"
    csv_path = out_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    meta = {
        "config": asdict(cfg),
        "ground_truth_equation": "dT/dt [K h-1] = a_heating*Rn - lam_relax*(T - T_eq)",
        "units": {
            "R_liq": "W m-2", "T_ar_62m": "degC", "UR_62m": "%", "VV_ane_62m": "m s-1",
            "P_atm": "hPa", "PRP": "mm per 30 min", "dTdt_true_K_per_h": "K h-1",
        },
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--missing", type=float, default=0.0, help="fraction of samples removed in gaps")
    ap.add_argument("--noise-scale", type=float, default=1.0, help="multiplies every noise sigma")
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--inspect", type=Path, help="only inspect an existing .csv/.xlsx (first sheet)")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    pd.set_option("display.width", 160)

    if args.inspect:
        reader = pd.read_excel if args.inspect.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv
        df = reader(args.inspect)
        print(f"{args.inspect.name}: {df.shape[0]} rows x {df.shape[1]} cols")
        print(inspect_dataframe(df).to_string())
        return

    cfg = GeneratorConfig(seed=args.seed)
    cfg.site.n_days = args.days
    cfg.noise.missing_fraction = args.missing
    for name in ("r_liq", "t_air", "rh", "wind", "p_atm"):
        setattr(cfg.noise, name, getattr(cfg.noise, name) * args.noise_scale)

    df = generate_synthetic(cfg)
    path = save_dataset(df, cfg, args.out)
    print(f"Saved {len(df)} rows -> {path}")
    print(inspect_dataframe(df[OBSERVED_COLS + ["dTdt_true_K_per_h"]]).to_string())
    print("\nMean diurnal cycle (every 3 h):")
    print(diurnal_composite(df, ["R_liq", "T_ar_62m", "UR_62m", "VV_ane_62m", "P_atm"]).iloc[5::6].to_string())


if __name__ == "__main__":
    main()
