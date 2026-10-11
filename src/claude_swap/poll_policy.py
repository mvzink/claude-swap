"""Cadence policy for the ``/api/oauth/usage`` endpoint — every number in one place.

The endpoint enforces a budget on non-first-party clients: a **~60-minute
window of ~28-30 requests per identity × UA-class** (measured 2026-07-11,
probe3, two runs: a rested identity admitted 30 requests before the first
429; the post-drain 429 oscillation ended exactly when the drain burst aged
60 minutes; steady 1/180 s polling then ran 96 minutes from a rested window
with zero 429s). It is NOT a bucket with a refill rate: capacity returns
only as old requests age out of the trailing hour, so a burst saturates the
identity for up to a full hour — pausing does not restore headroom early,
and earlier "refill rate" estimates were artifacts of measuring while
saturated.

What that identity is depends on which 429 regime the org is on — the two
regimes coexist across orgs (see the Retry-After discussion in
``usage_store``). Under the fixed-deadline regime it is the **account/org**
(measured 2026-07-28: a freshly minted token was blocked 135 s after issue,
which a per-token counter cannot produce). Under the saturated-edge
(``Retry-After: 0``) regime it is the **access token** (measured 2026-07-29,
probe4: at saturation, a freshly minted token of the same lineage was
admitted while the old token stayed blocked, requests interleaved). Plan for
the account-scoped case — it is the conservative one: re-authenticating
cannot be relied on to clear a block, and two machines holding different
tokens for one account may share one budget, which is what
``POST_429_BACKOFF_MULT`` below exists to converge.

Error bars: the horizon is bracketed to ~55-64 minutes from a single
transition event, the exact edge algorithm (likely a Cloudflare
sliding-window approximation) is undocumented, and Anthropic can retune it
any day — so the constants below lean only on the robust parts: a sustained
rate safely under the cap, and an ~hour recovery horizon. The budget target
is an **average of at most ~1 request / 3 minutes** (20/hour vs the ~28-30/
hour cap), leaving ~8-10 requests/hour of headroom for manual commands,
wake-from-sleep catch-up, and the bounded urgent mode below.

Health invariant to watch in the logs: steady state shows zero http-429.
A post-burst 429 does NOT reliably clear at its stated horizon — measured
over one machine's full log (re-measured 2026-08-03; re-derive rather than
trust this verbatim, it ages as the log grows — method recorded next to
``usage_store.RETRY_AFTER_MARGIN_S``), 20 of 35 lapsed blocks re-blocked
within 900s of their own deadline (+2s..+887s), each for a fresh full hour
("of 35", not 38 raw gaps: 3 are negative — NOT a uniform mechanism (one has
no within-block revision at all, one is revised BACKWARD mid-block, one is
unchanged — see usage_store.RETRY_AFTER_MARGIN_S's comment for the per-gap
detail) — excluded from both numerator and denominator; round 8 switched
from the prior "21 of 36"/"2 of 38" figures here to these, on the OTHER of
two equally-reproducing readings — see usage_store.RETRY_AFTER_MARGIN_S's
comment for which reading and why).
The prior "10 of 23" figure here did not reproduce under any of 40 method
variants swept and is superseded. That is why the wait is Retry-After plus
``usage_store.RETRY_AFTER_MARGIN_S`` and not Retry-After alone. What would
mean this model needs revisiting is a 429 episode at modest rates that
outlasts an hour *past* that margin.

Plans computed here are persisted per account in the usage store
(``nextPollAt``/``pollIntervalS``) by whichever collector fetched, so every
surface — ``cswap list``, the TUI, the menu bar, the auto engine — inherits
the same cadence no matter how often it repaints.

If a future probe revises the measured shape, adjust the constants in this
module only.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import datetime

from claude_swap import oauth

# Freshness floor shared by every collector: an entry younger than this is
# served from the store without any fetch, so the maximum sustained rate on
# one token is 1/SERVE_TTL_S regardless of how many surfaces are open.
SERVE_TTL_S = 180.0

# Normal cadence floor — movement can halve an interval down to this, never
# below.
MIN_INTERVAL_S = 180.0

# Urgent mode: the ACTIVE account, within ESCALATION_MARGIN_PCT of the
# switch threshold, with movement observed this poll (i.e. actually burning
# toward the limit). Bounded by construction: either the threshold is crossed
# (the engine switches away) or the movement stops (the next poll decays back
# to MIN_INTERVAL_S) — worst case margin/movement-delta ≈ 15 polls per
# episode, inside the measured ~28-30 request rolling-hour window; overshoot
# on top of steady traffic is absorbed by the post-429 floor below.
URGENT_INTERVAL_S = 60.0

# Decay ceilings for an account whose usage is not moving: the active account
# stays reasonably fresh, an idle alternate drifts out to ten minutes.
ACTIVE_MAX_INTERVAL_S = 300.0
CANDIDATE_DEFAULT_INTERVAL_S = 300.0
CANDIDATE_MAX_INTERVAL_S = 600.0

# Exhaustion is stable enough to poll slowly, but not to stop polling until a
# reported reset. Quota grants and provider-side corrections can make an
# account usable before that timestamp, and decision-grade status must not age
# into "unavailable" while the scheduler is deliberately waiting. Ten-minute
# polling (six requests/hour) stays below the measured budget and
# detects recovery promptly; a nearer reported reset still pulls the next poll
# forward.
EXHAUSTED_INTERVAL_S = 600.0

# A window whose binding pct moved at least this much between polls is being
# consumed somewhere (this machine, another PC, session mode) → tighten; an
# unmoved one backs off toward its ceiling.
MOVEMENT_DELTA_PCT = 1.0

# ±fraction applied to each scheduled interval so independent processes
# (watch + menu bar + auto) drift apart instead of fetching in lockstep.
JITTER_FRAC = 0.1

# Reaction to a 429 with ``Retry-After: 0`` (the saturated-window edge):
# probe at most every 5 minutes (≤12/hour) so aging-out — up to ~30/hour —
# outpaces the probing (used by the usage store's failure backoff)...
EDGE_BACKOFF_S = 300.0
# ...and while any 429 was seen on the token within this window, floor the
# planned cadence here so freed capacity accumulates instead of being
# re-spent. The window matches the saturation horizon: a full trailing hour
# takes up to 60 minutes to age out.
POST_429_MIN_INTERVAL_S = 360.0
RECENT_429_WINDOW_S = 3600.0

# AIMD on a contended budget. The budget is shared across every machine
# polling the same account (under the account-scoped regime a machine with its
# own token is no less a competitor — see the module docstring on scope),
# none of them can see the others, and the endpoint
# exposes no remaining-request count — only a Retry-After once already
# blocked. So while 429s recur, each successful poll multiplicatively grows the
# interval (×POST_429_BACKOFF_MULT) toward POST_429_MAX_INTERVAL_S — wider than
# the normal candidate ceiling so several machines can each back off far enough
# that their combined rate fits under the budget. Movement (a real success run
# with no recent 429) decays it back down. This is TCP-style congestion control:
# the budget gets fair-shared by reaction alone, with no machine count or
# shared state to configure.
POST_429_BACKOFF_MULT = 1.5
POST_429_MAX_INTERVAL_S = 1800.0

# The engine escalates to a full candidate refresh when the active account is
# within this margin of the threshold (decision policy, but the urgent-mode
# cadence keys on the same band, so it lives with the cadence numbers).
ESCALATION_MARGIN_PCT = 15.0

# --- burn-rate catch-up ----------------------------------------------------
# A rate is only measurable from two samples close enough together that the
# delta is the workload rather than bookkeeping. Below the floor the delta is
# inside the store's own noise; above the ceiling the samples straddle a window
# reset or a sleep, and the ratio is meaningless either way.
BURN_RATE_MIN_DT_S = 30.0
BURN_RATE_MAX_DT_S = 1800.0
# A rate this small is indistinguishable from idle: catch-up would ask for an
# interval above the ceiling anyway.
BURN_RATE_MIN_PCT_PER_MIN = 0.01
# Catch-up fires once the headroom left is worth no more than this many poll
# intervals of measured burn. At 1.0 the next poll lands exactly on the
# threshold; 2.0 therefore buys one whole interval of slack, which is the
# smallest unit that can close the gap the reactive test leaves open.
BURN_MARGIN_INTERVALS = 2.0
# Floor for the burn-derived trigger, as a utilization. Without it a very fast
# burn over a long interval would compute a trigger at or below the account's
# current utilization and switch on every tick regardless of the configured
# threshold; with it, a hot account still gets one interval's worth of margin.
BURN_EFFECTIVE_THRESHOLD_FLOOR_PCT = 50.0

# Never schedule a poll later than a known window reset (+ slack): stored
# usage is obsolete the moment the window rolls over.
RESET_SLACK_S = 60.0


def binding_pct(usage: dict | None, models: tuple[str, ...] = ()) -> float | None:
    """Utilization of the binding (worst) relevant window, or None."""
    headroom = oauth.account_headroom(usage, models)
    return None if headroom is None else 100.0 - headroom


def limiting_reset_ts(
    usage: dict | None, models: tuple[str, ...] = ()
) -> float | None:
    """Epoch when the last of the ≥100% relevant windows resets (account
    usable again)."""
    latest: float | None = None
    for _, pct, resets_at in oauth.relevant_windows(usage, models):
        if pct < 100.0:
            continue
        ts = parse_reset_ts(resets_at)
        if ts is not None and (latest is None or ts > latest):
            latest = ts
    return latest


def earliest_future_reset_ts(
    usage: dict | None, now: float, models: tuple[str, ...] = ()
) -> float | None:
    """Epoch of the next relevant-window reset ahead of ``now``, any
    utilization."""
    earliest: float | None = None
    for _, _, resets_at in oauth.relevant_windows(usage, models):
        ts = parse_reset_ts(resets_at)
        if ts is not None and ts > now and (earliest is None or ts < earliest):
            earliest = ts
    return earliest


def parse_reset_ts(resets_at: str | None) -> float | None:
    if not resets_at:
        return None
    try:
        return datetime.fromisoformat(
            str(resets_at).replace("Z", "+00:00")
        ).timestamp()
    except ValueError:
        return None


def burn_effective_threshold_pct(
    *,
    threshold: float,
    burn_pct_per_min: float | None,
    interval_s: float,
    models: tuple[str, ...],
) -> float:
    """The threshold a measured burn earns, in the unit the loop works in.

    A threshold is a percentage of a window, but the engine can only test it
    at a poll -- so a margin of ``100 - threshold`` points is a margin of
    ``(100 - threshold) / burn`` seconds, while the reaction time is
    ``interval_s``. Those two were never commensurate, and at the settings
    this was reported on the mismatch was stark: 0.5 points of margin is
    roughly 25% of *one* poll at a 2%/min burn, and about a second at the
    99th-percentile burn. Measured on a real log, every overshoot that reached
    100% with the engine still testing below the threshold did so with the
    account already inside the escalation band and the cadence already at its
    60s floor -- so no amount of faster polling closes it. The margin itself
    was too small.

    This returns ``threshold`` unchanged when no rate is measurable (the
    degradation path is exactly the pre-existing engine), and otherwise
    raises the effective trigger to leave ``BURN_MARGIN_INTERVALS`` intervals
    of burn in hand: a switch at ``100 - burn x interval x K``. A slow account
    is unaffected -- at 0.5%/min and 180s that is 3 points, so 97 rather than
    99.5 -- which is the point: the trigger follows consumption instead of
    assuming every account burns at the same rate.

    Never lowers the configured threshold, and never raises the effective
    trigger above ``100 - BURN_RATE_MIN_PCT_PER_MIN``: an account always gets
    *some* margin before the line, however fast it burns.
    """
    if burn_pct_per_min is None or burn_pct_per_min <= 0:
        return threshold
    per_interval = burn_pct_per_min / 60.0 * interval_s * BURN_MARGIN_INTERVALS
    if per_interval <= 0:
        return threshold
    # The burn overrides the configured value in BOTH directions -- that is the
    # whole point: a hot account needs its trigger pulled closer, and one whose
    # margin the configured threshold already covers (fast burn over a long
    # interval, i.e. a small ``100 - per_interval``) keeps whichever is stricter.
    # Bounded so an account always gets some margin before the hard limit.
    return max(
        BURN_EFFECTIVE_THRESHOLD_FLOOR_PCT,
        min(threshold, 100.0 - per_interval),
    )


def burn_pct_per_min_from_pcts(
    prev_pct: float | None,
    new_pct: float | None,
    *,
    dt_s: float | None,
) -> float | None:
    """Percent-per-minute burn from two already-resolved binding percentages.

    The single home of the measurability guards, shared by the collector (which
    has two usage dicts) and the engine (which has two percentages remembered
    from its own previous ticks).
    """
    if dt_s is None or dt_s <= 0:
        return None
    if dt_s < BURN_RATE_MIN_DT_S or dt_s > BURN_RATE_MAX_DT_S:
        return None
    if prev_pct is None or new_pct is None:
        return None
    delta = new_pct - prev_pct
    if delta <= 0:
        return None
    rate = delta / (dt_s / 60.0)
    if rate < BURN_RATE_MIN_PCT_PER_MIN:
        return None
    return rate


def burn_pct_per_min(
    prev_usage: dict | None,
    new_usage: dict | None,
    *,
    dt_s: float | None,
    models: tuple[str, ...],
) -> float | None:
    """Percent-per-minute burn of the binding window between two samples.

    ``None`` whenever the rate is not trustworthy: no previous sample, an
    unreadable binding window on either side, an interval outside
    ``[BURN_RATE_MIN_DT_S, BURN_RATE_MAX_DT_S]`` (too close to read, or
    straddling a reset/sleep), a non-positive interval, or a delta that is flat
    or negative (an idle account, or a window that just rolled over). Callers
    treat ``None`` as "unmeasured", never as zero burn.

    Measured on the BINDING window rather than the 5h window alone, so a burn
    led by the weekly or a scoped per-model window is not missed, and so the
    rate and the threshold agree about what is being consumed.
    """
    return burn_pct_per_min_from_pcts(
        binding_pct(prev_usage, models),
        binding_pct(new_usage, models),
        dt_s=dt_s,
    )


def burn_catchup_interval_s(
    *,
    prev_pct: float | None,
    new_pct: float | None,
    dt_s: float | None,
    interval_s: float,
    is_active: bool,
    threshold: float,
    recent_429: bool,
    models: tuple[str, ...],
) -> float | None:
    """The faster interval a measured burn earns, or ``None`` to keep
    ``interval_s``.

    The threshold is a percentage of a window and the engine can only test it
    at a poll, so a reactive test is late by up to one whole poll interval of
    burn *by construction*. Measured on a real log, that is precisely the
    budget the last points of headroom were being spent from: the median
    sampling interval while the active account sat at 98-99% used was 240s
    (2100s at the 90th percentile), while the median measured burn spent the
    last 2 points in 240s and the 90th percentile spent them in 45s. The
    margin and the reaction time were in different units, so the margin was
    gone before the test could run.

    This makes the margin a *time* margin, in the unit the loop actually
    works in: keep the headroom worth at least ``BURN_MARGIN_INTERVALS`` poll
    intervals of measured burn, and ask to be polled at a fraction of the time
    that headroom takes to spend, so the next reading still has margin left
    when it arrives rather than landing exactly on the line. A slow account
    that is merely *near* the threshold is untouched -- it keeps the ordinary
    cadence and the percentage test, which is not late for it by more than it
    can afford.

    The result is clamped to ``[MIN_INTERVAL_S, interval_s]``:

      * never longer than what the caller already chose (only ever speeds up),
      * never shorter than ``MIN_INTERVAL_S``, the cadence floor the
        ~28-30-req/hour usage-endpoint budget was measured against. Sub-floor
        polling stays the existing urgent tier's job (it fires at 60s when a
        single poll catches ``MOVEMENT_DELTA_PCT`` of movement), so this
        **adds no new budget pressure**: it only pulls a decayed cadence back
        toward the floor when a burn justifies it.

    ``recent_429`` suppresses it outright: a throttled token must accumulate
    budget, and billing a catch-up poll against a saturated window is how the
    *next* real reading gets refused (upstream issue #208).
    """
    if recent_429 or not is_active:
        return None
    if prev_pct is None or new_pct is None or new_pct <= prev_pct:
        return None
    if dt_s is None or dt_s < BURN_RATE_MIN_DT_S or dt_s > BURN_RATE_MAX_DT_S:
        return None
    burn = (new_pct - prev_pct) / (dt_s / 60.0)
    if burn < BURN_RATE_MIN_PCT_PER_MIN:
        return None
    headroom_pct = 100.0 - new_pct
    if headroom_pct <= 0:
        return None
    # Only the band the decision already escalates in: outside it there is no
    # crossing to catch up to, and the ordinary cadence is cheaper. This is a
    # precondition, not a floor -- a slow account sitting inside the band is
    # still left alone by the time test below.
    if headroom_pct > ESCALATION_MARGIN_PCT:
        return None
    # burn is percent per MINUTE and interval_s is in seconds: convert before
    # comparing, or the margin is 60x too generous and every slow account
    # inside the band gets tightened.
    margin_pct = burn / 60.0 * interval_s * BURN_MARGIN_INTERVALS
    if headroom_pct > margin_pct:
        return None
    # Aim to arrive with the margin half spent: polling at the full
    # time-to-exhaust would land exactly on 100%.
    time_to_exhaust_s = headroom_pct / burn * 60.0
    want = time_to_exhaust_s / BURN_MARGIN_INTERVALS
    return min(interval_s, max(MIN_INTERVAL_S, want))


def plan_after_fetch(
    *,
    prev_interval_s: float | None,
    prev_usage: dict | None,
    new_usage: dict | None,
    is_active: bool,
    threshold: float,
    models: tuple[str, ...],
    recent_429: bool,
    now: float,
    dt_s: float | None = None,
    rng: Callable[[], float] = random.random,
) -> tuple[float, float]:
    """``(next_poll_at, interval_s)`` for an account just fetched successfully.

    Movement (binding pct changed ≥ ``MOVEMENT_DELTA_PCT`` since the previous
    poll) halves the interval, floored at ``MIN_INTERVAL_S`` — or drops to
    ``URGENT_INTERVAL_S`` when the active account is moving inside the
    escalation band. No movement backs off ×1.5 toward the account's ceiling;
    unknown utilization uses the default. A recent 429 on this token floors
    the cadence at ``POST_429_MIN_INTERVAL_S`` (and suppresses urgent mode)
    until ``RECENT_429_WINDOW_S`` has passed. ``dt_s`` is the wall time since
    ``prev_usage`` was measured; with it, a measurable burn rate additionally
    tightens the active account's cadence to ``burn_catchup_interval_s`` so the
    threshold is not crossed between polls. The scheduled time gets
    ``JITTER_FRAC`` noise, is never later than the account's next window
    reset (+ ``RESET_SLACK_S``). An at-limit account keeps a bounded slow
    poll instead of sleeping until that reset, so an early provider-side
    quota grant is observed and its decision-grade status stays current.
    """
    default = MIN_INTERVAL_S if is_active else CANDIDATE_DEFAULT_INTERVAL_S
    ceiling = ACTIVE_MAX_INTERVAL_S if is_active else CANDIDATE_MAX_INTERVAL_S
    base = prev_interval_s or default
    prev_pct = binding_pct(prev_usage, models)
    new_pct = binding_pct(new_usage, models)
    if prev_pct is None or new_pct is None:
        moving = False
        interval = default
    elif abs(new_pct - prev_pct) >= MOVEMENT_DELTA_PCT:
        moving = True
        interval = max(MIN_INTERVAL_S, base / 2)
    else:
        # Floored so a sub-floor base (urgent mode's 60s) snaps straight back
        # to the normal cadence once movement stops, instead of decaying
        # through 90s/135s polls that the budget never intended.
        moving = False
        interval = min(ceiling, max(MIN_INTERVAL_S, base * 1.5))
    if (
        is_active
        and moving
        and not recent_429
        and new_pct is not None
        and new_pct >= threshold - ESCALATION_MARGIN_PCT
    ):
        interval = URGENT_INTERVAL_S
    if recent_429:
        # AIMD additive-increase: grow the interval multiplicatively from the
        # last one toward the wider 429 ceiling, so machines sharing a
        # contended token each retreat until their combined rate fits the
        # budget. Floored at POST_429_MIN_INTERVAL_S for the first 429.
        increased = max(base * POST_429_BACKOFF_MULT, POST_429_MIN_INTERVAL_S)
        interval = min(POST_429_MAX_INTERVAL_S, max(interval, increased))
    else:
        # Burn-rate catch-up, after the movement/decay decision so it can only
        # tighten it, and never inside a 429 window (see the helper).
        catch_up = burn_catchup_interval_s(
            prev_pct=prev_pct,
            new_pct=new_pct,
            dt_s=dt_s,
            interval_s=interval,
            is_active=is_active,
            threshold=threshold,
            recent_429=recent_429,
            models=models,
        )
        if catch_up is not None:
            interval = catch_up

    headroom = oauth.account_headroom(new_usage, models)
    if headroom is not None and headroom <= 0:
        # Keep probing exhausted accounts: Anthropic can grant/reset quota
        # before the previously advertised timestamp. Preserve a wider
        # post-429 interval if congestion control already selected one.
        interval = max(interval, EXHAUSTED_INTERVAL_S)

    next_poll = now + interval * (1.0 + JITTER_FRAC * (2.0 * rng() - 1.0))
    if headroom is not None and headroom <= 0:
        reset_ts = limiting_reset_ts(new_usage, models)
        if reset_ts is not None and reset_ts > now:
            next_poll = min(next_poll, reset_ts + RESET_SLACK_S)
    else:
        reset_ts = earliest_future_reset_ts(new_usage, now, models)
        if reset_ts is not None:
            next_poll = min(next_poll, reset_ts + RESET_SLACK_S)
    return next_poll, interval
