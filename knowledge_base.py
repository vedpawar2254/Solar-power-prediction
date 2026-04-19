"""Grid management knowledge base for RAG retrieval."""

GRID_KNOWLEDGE = [
    # Ramp rate management
    """Ramp Rate Management: Solar power output can change rapidly due to cloud
    transients. Grid operators should enforce ramp rate limits of 10% of rated
    capacity per minute. When forecast variability exceeds 20%, pre-position
    spinning reserves and activate automatic generation control (AGC) to
    compensate for sudden ramps.""",

    """Ramp Rate Mitigation with Storage: Battery energy storage systems (BESS)
    can absorb fast ramps by charging during sudden output spikes and discharging
    during drops. A 15-minute storage buffer at 25% of plant capacity effectively
    smooths ramp rates below grid-acceptable thresholds.""",

    # Battery storage dispatch
    """Battery Storage Dispatch Rules: Dispatch stored energy when solar output
    drops below 60% of forecasted value for more than 10 minutes. Maintain
    minimum state of charge (SOC) at 20% for emergency grid support. Priority
    dispatch order: frequency regulation first, peak shaving second, energy
    arbitrage third.""",

    """Battery Charging Strategy: Charge batteries during periods of excess
    solar generation (output exceeding demand by >15%). Optimal charging windows
    are typically 10:00-14:00 when irradiation is highest. Avoid deep discharge
    below 10% SOC to preserve battery cycle life.""",

    # Grid frequency regulation
    """Grid Frequency Regulation: Maintain grid frequency within 49.95-50.05 Hz
    under normal conditions. Solar variability can cause frequency deviations
    when penetration exceeds 30% of total generation. Deploy fast-response
    inverters with synthetic inertia capability to provide sub-second frequency
    support.""",

    """Frequency Response Requirements: Primary frequency response must activate
    within 2 seconds. Solar-plus-storage plants should reserve 5% of capacity
    for frequency regulation. During high variability periods, increase frequency
    regulation reserve to 10% of capacity.""",

    # Curtailment strategies
    """Curtailment Strategy: When solar generation exceeds grid absorption
    capacity, implement graduated curtailment starting at 5% reduction
    increments. Prioritize curtailment of plants with lowest marginal cost
    impact. Coordinate curtailment with battery charging to minimize energy
    waste.""",

    """Smart Curtailment During Oversupply: During midday oversupply periods,
    redirect excess generation to battery storage before curtailing. If storage
    is full, implement demand response programs to shift load to high-generation
    periods. Curtailment should be last resort after storage and demand
    shifting.""",

    # Demand response
    """Demand Response During Low Irradiation: When irradiation drops below
    200 W/m², activate demand response programs to reduce non-critical loads.
    Shift flexible loads (HVAC pre-cooling, water heating, EV charging) to
    high-irradiation periods. Target 10-15% demand reduction during low
    generation events.""",

    """Load Shifting Strategies: Identify shiftable loads that can move to
    peak solar hours (10:00-15:00). Industrial processes, cold storage
    charging, and water treatment are prime candidates. Automated demand
    response systems should activate when forecast confidence drops below
    70%.""",

    # Reserve margins
    """Reserve Margin Requirements: Maintain operating reserves at minimum
    15% of forecasted solar capacity during stable conditions. Increase to
    25% during high-variability periods (cloud cover >60%, wind speed >8 m/s).
    Reserves should include both spinning (immediate) and non-spinning
    (10-minute start) components.""",

    """Dynamic Reserve Allocation: Adjust reserve requirements based on
    forecast uncertainty bands. When prediction interval width exceeds 30%
    of mean forecast, activate additional fast-start generation. Coordinate
    reserves across interconnected grid regions to optimize cost.""",

    # Forecasting uncertainty
    """Forecasting Uncertainty Buffers: Apply uncertainty margins based on
    forecast horizon: ±5% for 15-minute ahead, ±10% for 1-hour ahead,
    ±20% for day-ahead forecasts. During weather transitions, double the
    uncertainty buffer. Use ensemble forecasting to quantify prediction
    confidence intervals.""",

    """Forecast Error Compensation: When actual generation deviates from
    forecast by more than 15%, trigger automatic rebalancing. Maintain a
    rolling 4-hour forecast error tracker to detect systematic bias.
    Recalibrate models when mean absolute percentage error exceeds 12%
    over a 24-hour window.""",

    # Peak shaving
    """Peak Shaving via Storage: Deploy stored energy during evening demand
    peaks (17:00-21:00) when solar generation declines. Size storage to cover
    at least 2 hours of peak demand gap. Begin discharge when solar output
    drops below 30% of midday peak to ensure smooth transition.""",

    """Peak Demand Management: Combine solar forecasting with load forecasting
    to predict net demand peaks. Pre-charge storage during solar surplus for
    evening peak shaving. Target reducing peak grid demand by 15-20% through
    coordinated solar-storage-demand response.""",

    # Voltage stability
    """Voltage Stability Management: Solar inverters should operate in
    volt-VAR mode to maintain voltage within ±5% of nominal. During rapid
    irradiation changes, activate dynamic reactive power compensation.
    Monitor voltage at point of common coupling (PCC) and adjust inverter
    setpoints every 100ms.""",

    """Power Quality Standards: Total harmonic distortion (THD) from solar
    inverters must remain below 5%. During low-generation periods, inverters
    may need to provide reactive power support even without active power
    output. Implement anti-islanding protection with detection time under
    2 seconds.""",
]
