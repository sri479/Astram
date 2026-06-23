def optimize_resources(prediction_proba, event_type, requires_closure, event_cause):
    """
    Translates leakage-free ML probabilities into real deployment assets.
    """
    impact_score = int(prediction_proba * 100)
    
    # Catch non-disruptive infrastructure issues early before wasting human assets
    minor_triggers = ["pot_holes", "manhole_open"]
    if event_cause in minor_triggers and requires_closure == 0:
        impact_score = min(30, int(impact_score * 0.35))

    # Structural baseline allocations
    manpower = 2
    barricades = 0
    diversion_strategy = "Routine Traffic Monitoring"

    if impact_score >= 65:
        manpower = 10
        barricades = 20
        diversion_strategy = "CRITICAL RISK: Enact wide peripheral detours at preceding 3 intersections."
    elif impact_score >= 40:
        manpower = 6
        barricades = 12
        diversion_strategy = "ELEVATED RISK: Implement local lane splittings at the immediate junction."
    elif impact_score >= 15:
        manpower = 4
        barricades = 4
        diversion_strategy = "MODERATE FLUIDITY: Point-duty monitoring via active manual signaling."

    if event_type == "planned":
        manpower += 3  # Strategic crowd management buffer
        
    if requires_closure == 1:
        barricades = max(barricades, 15)
        if "CRITICAL" not in diversion_strategy:
            diversion_strategy = "CLOSURE FORCED: Reroute approaching traffic away from corridor entryways."

    return {
        "impact_score": min(100, max(0, impact_score)),
        "manpower_required": manpower,
        "barricades_required": barricades,
        "diversion_strategy": diversion_strategy
    }