#!/usr/bin/env python3
"""stress_test.py
Locust load-test script for vroong-ml-service.

Features
--------
1. Two traffic scenarios selectable via env var SCENARIO ("1" or "2").
2. Simulates daily traffic pattern (11-14h, 17-20h peaks) within a compressed
   test window (default 30 min) using a custom LoadShape.
3. Generates random request payloads that match the provided API spec.
4. Base URL and target endpoint are provided via env vars TARGET_HOST and
   ENDPOINT_PATH so the same script works across environments.
5. Success criteria (to be inspected in the Locust Web UI):
   • Success = HTTP 200 response within 1000 ms.
   • 95th & 99th percentile response time must be ≤ 1000 ms.
   • Failure ratio must remain < 1 % (⇒ ≥ 99 % success-rate).

Run
---
$  uv run locust -f deployment/scripts/stress_test.py --headless --csv=deployment/scripts/logs/stress_test_resuts

Locust will override the dummy "-u/-r" values as soon as the LoadShape kicks
in (they are still required by the CLI parser).
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from locust import HttpUser, LoadTestShape, TaskSet, events, task, between

# ---------------------------------------------------------------------------
# Configuration via environment variables
# ---------------------------------------------------------------------------
BASE_URL: str = os.getenv(
    "TARGET_HOST",
    "http://172.20.207.198:8000/",
)
ENDPOINT_PATH: str = os.getenv("ENDPOINT_PATH", "/order_acceptance/predict_batch")
SCENARIO: str = os.getenv("SCENARIO", "max")  # either "1" or "2"
SIM_DURATION: int = int(os.getenv("SIM_DURATION", 30 * 60))  # seconds
AVG_LATENCY_S: float = float(os.getenv("AVG_LATENCY", 0.66))  # sec, used to map RPS→users
OK_LATENCY_MS: int = int(os.getenv("OK_LATENCY_MS", 1000))  # max acceptable latency per request
# New: configurable think time between tasks (helps reach high RPS when set to 0)
WAIT_TIME_MIN_S: float = float(os.getenv("WAIT_TIME_MIN_S", 0.1))
WAIT_TIME_MAX_S: float = float(os.getenv("WAIT_TIME_MAX_S", 0.3))
# New: safety factor to overprovision users to reliably hit target RPS
OVERPROVISION_FACTOR: float = float(os.getenv("OVERPROVISION_FACTOR", 1.25))


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
@events.init.add_listener
def setup_logging(environment, **kwargs):
    """Set up logging to a file with a timestamp (e.g., 'stress_test_240726.log')."""
    log_dir = "deployment/scripts/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"stress_test_{datetime.now().strftime('%y%m%d')}.log")
    handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s/%(name)s: %(message)s")
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Test run logs are being saved to {log_filename}")


def _random_coordinates() -> Dict[str, float]:
    return {
        "latitude": random.uniform(-90, 90),
        "longitude": random.uniform(-180, 180),
    }


def _iso_timestamp() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _random_driver_feature() -> Dict[str, Any]:
    return {
        "agent_external_id": str(uuid.uuid4()),
        "base_surge_amount": random.choice([0, 500, 1000, 1500, 2000, 2500, 3000]),
        "sla_achieve_surge_amount": 0,
        "agent_fee_amount": 0,
        "extra_charge_amount": 0,
        "driver_current_coordinates": _random_coordinates(),
        "task_routes": [
            {
                "task_type": "pickup",
                "order_id": str(uuid.uuid4()),
                "coordinates": _random_coordinates(),
                "waiting_time": 0,
            }
        ],
    }


def _random_driver_features(min_drivers: int = 1, max_drivers: int = 20) -> List[Dict[str, Any]]:
    """Generate a list of random driver features."""
    num_drivers = random.randint(min_drivers, max_drivers)
    return [_random_driver_feature() for _ in range(num_drivers)]


def _random_order_info() -> Dict[str, Any]:
    timestamp = _iso_timestamp()
    return {
        "order_id": str(uuid.uuid4()),
        "created_at": timestamp,
        "requested_pickup_at": timestamp,
        "sla_deadline_at": timestamp,
        "pickup_info": {
            "coordinates": _random_coordinates(),
            "si_do": "",
            "si_gun_gu": "",
            "eup_myeon_dong": "",
            "address_roadaddress": "",
            "company_name": "",
            "task_duration": 0,
            "waiting_time": 0,
        },
        "destination_info": {
            "coordinates": _random_coordinates(),
            "si_do": "",
            "si_gun_gu": "",
            "eup_myeon_dong": "",
            "address_roadaddress": "",
            "company_name": "",
            "task_duration": 0,
            "waiting_time": 0,
        },
    }


def generate_payload() -> Dict[str, Any]:
    """Return a fresh request payload matching the API spec."""
    return {
        "order_id": str(uuid.uuid4()),
        "driver_features": _random_driver_features(),
        "order_features": _random_order_info(),
        "variants": [
            {"base_surge_amount": 0},
            {"base_surge_amount": 100},
            {"base_surge_amount": 200},
            {"base_surge_amount": 300},
            {"base_surge_amount": 400},
            {"base_surge_amount": 500},
            {"base_surge_amount": 600},
            {"base_surge_amount": 700},
            {"base_surge_amount": 800},
            {"base_surge_amount": 900},
            {"base_surge_amount": 1000},
        ],
    }


# ---------------------------------------------------------------------------
# Locust user and tasks
# ---------------------------------------------------------------------------
class OrderAcceptanceUser(HttpUser):
    host = BASE_URL
    wait_time = between(WAIT_TIME_MIN_S, WAIT_TIME_MAX_S)  # configurable think time (default 0)

    @task
    def rank_variants(self) -> None:
        payload = generate_payload()
        with self.client.post(
            ENDPOINT_PATH, json=payload, catch_response=True, timeout=10
        ) as response:
            if response.status_code != 200:
                response.failure(f"Unexpected status {response.status_code}")
                logging.warning(
                    f"Request to {ENDPOINT_PATH} failed with status {response.status_code}. Response: {response.text}"
                )
            else:
                # Enforce success threshold: 1s (configurable via OK_LATENCY_MS)
                try:
                    latency_ms = int(response.elapsed.total_seconds() * 1000)
                except Exception:
                    # Fallback: mark as failure if latency can't be determined
                    response.failure("Latency measurement unavailable")
                else:
                    if latency_ms > OK_LATENCY_MS:
                        response.failure(f"Latency {latency_ms}ms > {OK_LATENCY_MS}ms threshold")
                    else:
                        response.success()


# ---------------------------------------------------------------------------
# Custom Load Shape – compress daily traffic into SIM_DURATION seconds
# ---------------------------------------------------------------------------
class DailyTrafficShape(LoadTestShape):
    """Simulate the specified daily traffic pattern within a shorter test window."""

    SECONDS_PER_DAY = 24 * 3600

    # Desired requests-per-second for each scenario
    _SCENARIO_RPS: Dict[str, Dict[str, float]] = {
        # peak1 11-14h, peak2 17-20h, off-peak elsewhere
        "1": {
            "peak1": 6.0,  # ≈ 60 000 requests / 3 h
            "peak2": 8.0,  # ≈ 80 000 requests / 3 h
            "off": 1.0,  # ≈ 60 000 requests / 18 h
        },
        # 200k requests distributed as peak1=30%, peak2=50%, off=20%
        "2": {
            "peak1": 5.56,  # 60 000 requests / 3 h
            "peak2": 9.26,  # 100 000 requests / 3 h
            "off": 0.62,  # 40 000 requests / 18 h
        },
        # Max capacity scenario: target 40 rps (peak1) and 60 rps (peak2)
        "3": {
            "peak1": 20.0,
            "peak2": 30.0,
            "off": 1.0,
        },
        "max": {  # alias for convenience
            "peak1": 20.0,
            "peak2": 30.0,
            "off": 1.0,
        },
    }

    def __init__(self) -> None:
        super().__init__()
        if SCENARIO not in self._SCENARIO_RPS:
            raise ValueError(
                f"Unknown SCENARIO={SCENARIO}. Expected one of {list(self._SCENARIO_RPS)}"
            )
        self.rps_conf = self._SCENARIO_RPS[SCENARIO]
        logging.info(
            f"Using scenario '{SCENARIO}' with targets: peak1={self.rps_conf['peak1']} rps, "
            f"peak2={self.rps_conf['peak2']} rps, off={self.rps_conf['off']} rps; "
            f"AVG_LATENCY_S={AVG_LATENCY_S}s, WAIT_TIME=[{WAIT_TIME_MIN_S},{WAIT_TIME_MAX_S}]s, "
            f"OVERPROVISION_FACTOR={OVERPROVISION_FACTOR}"
        )

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _simulated_time_of_day(self, run_time: float) -> float:
        """Map test run_time (sec) → simulated seconds since midnight."""
        fraction_of_test = (run_time % SIM_DURATION) / SIM_DURATION
        return fraction_of_test * self.SECONDS_PER_DAY

    def _current_rps(self, t_day: float) -> float:
        """Return desired RPS for the simulated time of day."""
        if 11 * 3600 <= t_day < 14 * 3600:
            return self.rps_conf["peak1"]
        if 17 * 3600 <= t_day < 20 * 3600:
            return self.rps_conf["peak2"]
        return self.rps_conf["off"]

    @staticmethod
    def _rps_to_users(rps: float) -> int:
        """Convert RPS → concurrent users based on expected latency and think time.

        Needed users U ≈ rps × (latency + avg_think_time) × overprovision_factor
        """
        avg_think_time = (WAIT_TIME_MIN_S + WAIT_TIME_MAX_S) / 2.0
        users = rps * (AVG_LATENCY_S + avg_think_time) * OVERPROVISION_FACTOR
        return max(1, int(users))

    # ------------------------------------------------------------------
    # Main tick – called every second by Locust
    # ------------------------------------------------------------------
    def tick(self) -> Tuple[int, int] | None:  # noqa: D401 (simple signature)
        run_time = self.get_run_time()
        if run_time > SIM_DURATION:
            return None  # stop the test

        t_day = self._simulated_time_of_day(run_time)
        rps = self._current_rps(t_day)
        user_count = self._rps_to_users(rps)

        # Spawn rate: spawn all users within ~5 seconds for responsiveness
        spawn_rate = max(1, user_count // 5)
        return user_count, spawn_rate