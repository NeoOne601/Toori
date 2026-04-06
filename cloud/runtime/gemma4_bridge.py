"""
gemma4_bridge.py — Connects JEPA geometric evidence to Gemma 4 on-device language.

CRITICAL DESIGN PRINCIPLE:
  Evidence goes into the prompt FIRST. Gemma 4 narrates measurements, not imagination.
  This is why it cannot hallucinate a telescope into a wig.
"""
from __future__ import annotations
import asyncio, json, re, time
from dataclasses import dataclass
from typing import Any, Optional

_ANCHOR_SYSTEM = (
    "You are a precise visual-spatial narrator for a JEPA world model. "
    "You receive geometric evidence and produce SHORT factual descriptions "
    "grounded ONLY in that evidence. Never add details not in the evidence. "
    "If confidence is below 0.5, say 'uncertain'. One sentence, max 20 words."
)
_QUERY_SYSTEM = (
    "You are a search query parser for a visual memory system. "
    "Convert the user query to a JSON search object. Output ONLY valid JSON. "
    'Schema: {"setu_query":str,"depth_stratum":str|null,"person_filter":str|null,'
    '"time_hint":str|null,"confidence_min":float}'
)
_COREF_SYSTEM = (
    "You are an entity co-reference resolver for a visual tracking system. "
    "Match the user reference to the most likely entity track. "
    'Output ONLY valid JSON: {"entity_id":str|null,"confidence":float,"reasoning":str}'
)
_ALERT_SYSTEM = (
    "You are a proactive safety narrator for live scene monitoring. "
    "One short alert if notable, else output exactly: stable. Max 15 words."
)
_ROLLOUT_SYSTEM = (
    "You are a concise layout and physics narrator. Explain the system's "
    "primary planned action (Plan A) and its fallback (Plan B) in simple, "
    "natural language. Explain why Plan B exists if Plan A is risky. "
    "Maximum 2 sentences."
)
_PROOF_REPORT_SYSTEM = (
    "You are a technical analyst summarizing a session's visual memory performance. "
    "You will receive aggregated metrics (ticks, energy, recoveries, etc) and must "
    "produce a highly detailed 1-2 paragraph natural language summary "
    "evaluating the session's stability and object tracking success."
)


@dataclass
class AnchorNarrationResult:
    text: str; anchor_name: str; depth_stratum: str
    confidence: float; hallucination_risk: float; latency_ms: float
    model: str = "gemma-4-e4b"

@dataclass
class QueryReformulationResult:
    setu_query: str; depth_stratum: Optional[str]; person_filter: Optional[str]
    time_hint: Optional[str]; confidence_min: float; raw_response: str; latency_ms: float

@dataclass
class AlertResult:
    text: str; is_alert: bool; latency_ms: float

class Gemma4Bridge:
    def __init__(self, mlx_call): self._call = mlx_call; self._n = 0; self._ms = 0.0

    async def narrate_anchor(self, anchor_matches, depth_strata, setu_descriptions,
                              observation_id, image_base64=None):
        if not anchor_matches:
            return AnchorNarrationResult("No anchors.","none","unknown",0.0,1.0,0.0)
        top = max(anchor_matches, key=lambda a: a.get("confidence",0))
        name = top.get("template_name","unknown")
        stratum = top.get("depth_stratum","unknown")
        conf = float(top.get("confidence",0.0))
        patches = len(top.get("patch_indices",[]))
        prior = ""
        if setu_descriptions:
            s = setu_descriptions[0]
            st = s.get("description",{}).get("text","") if isinstance(s.get("description"),dict) else ""
            sr = s.get("description",{}).get("hallucination_risk",1.0) if isinstance(s.get("description"),dict) else 1.0
            if st and sr < 0.5: prior = f"\nExisting description (risk={sr:.2f}): {st}"
        depth_ctx = ""
        if isinstance(depth_strata,dict):
            depth_ctx = f"\nDepth confidence: {depth_strata.get('confidence',0):.2f}"
        prompt = (f"JEPA geometric evidence for obs {str(observation_id)[:8]}:\n"
                  f"- Anchor: {name}\n- Stratum: {stratum}\n"
                  f"- Confidence: {conf:.2f}\n- Patch coverage: {patches}/196"
                  f"{depth_ctx}{prior}\n\n"
                  "Narrate in one sentence what this anchor match describes. Max 20 words.")
        t0 = time.perf_counter()
        r = await self._call(prompt=prompt,image_base64=image_base64,system=_ANCHOR_SYSTEM,max_tokens=64)
        ms = (time.perf_counter()-t0)*1000; self._n+=1; self._ms+=ms
        text = (r.get("text") or "").strip() or f"{name.replace('_',' ')} in the {stratum}."
        return AnchorNarrationResult(text,name,stratum,conf,max(0.0,min(1.0,1.0-conf)),ms,r.get("model","gemma-4-e4b"))

    async def reformulate_query(self, natural_query):
        prompt = (f'User memory query: "{natural_query}"\n\nParse to JSON. '
                  "depth_stratum: foreground|midground|background|null. Output ONLY JSON.")
        t0 = time.perf_counter()
        r = await self._call(prompt=prompt,image_base64=None,system=_QUERY_SYSTEM,max_tokens=128)
        ms = (time.perf_counter()-t0)*1000; self._n+=1; self._ms+=ms
        raw = (r.get("text") or "").strip()
        try: parsed = json.loads(re.sub(r"```(?:json)?|```","",raw).strip())
        except Exception: parsed = {"setu_query": natural_query}
        return QueryReformulationResult(
            setu_query=parsed.get("setu_query",natural_query),
            depth_stratum=parsed.get("depth_stratum"),
            person_filter=parsed.get("person_filter"),
            time_hint=parsed.get("time_hint"),
            confidence_min=float(parsed.get("confidence_min",0.3)),
            raw_response=raw, latency_ms=ms)

    async def proactive_alert(self, scene_state, entity_tracks, jepa_tick=None):
        surprise = float((jepa_tick or {}).get("surprise_score") or 0.0)
        occ = scene_state.get("metrics",{}).get("persistence_signal",{}).get("occluded_track_ids",[])
        if surprise < 0.55 and not occ:
            return AlertResult("stable",False,0.0)
        changed = scene_state.get("changed_elements",[])[:3]
        summary = scene_state.get("observed_state_summary","")[:120]
        prompt = (f"World model state:\n  Surprise: {surprise:.2f}\n"
                  f"  Occluded tracks: {len(occ)}\n  Changed: {', '.join(changed) or 'none'}\n"
                  f"  Summary: {summary}\n\nAlert or output: stable")
        t0 = time.perf_counter()
        r = await self._call(prompt=prompt,image_base64=None,system=_ALERT_SYSTEM,max_tokens=48)
        ms = (time.perf_counter()-t0)*1000; self._n+=1; self._ms+=ms
        text = (r.get("text") or "stable").strip()
        is_alert = text.lower() not in ("stable","","no alert","normal")
        return AlertResult(text,is_alert,ms)

    async def narrate_rollout(self, rollout_comparison: dict):
        branches = rollout_comparison.get("ranked_branches", [])
        if not branches:
            return "No rollout plans available to narrate."
        plan_a = branches[0]
        plan_b = branches[1] if len(branches) > 1 else None
        
        def _get_verb(plan):
            return plan.get("candidate_action", {}).get("verb", "unknown")
            
        a_verb = _get_verb(plan_a).replace("_", " ")
        a_risk = plan_a.get("risk_score", 0.0)
        a_blockers = ", ".join(plan_a.get("failure_predicates", [])) or "none"
        
        prompt = (f"Plan A: {a_verb} (Risk: {a_risk:.2f}, Blockers: {a_blockers})\n")
        if plan_b:
            b_verb = _get_verb(plan_b).replace("_", " ")
            b_risk = plan_b.get("risk_score", 0.0)
            prompt += f"Plan B fallback: {b_verb} (Risk: {b_risk:.2f})\n"
            
        prompt += "\nExplain this action plan easily to a user."
        
        t0 = time.perf_counter()
        r = await self._call(prompt=prompt, image_base64=None, system=_ROLLOUT_SYSTEM, max_tokens=128)
        ms = (time.perf_counter()-t0)*1000; self._n+=1; self._ms+=ms
        
        text = (r.get("text") or "").strip()
        return text if text else "Rollout plan calculated safely."

    async def narrate_proof_report(self, summary_stats: dict) -> str:
        prompt = (
            "Session Metrics Summary:\n"
            f"- Data points captured: {summary_stats.get('total_ticks', 0)}\n"
            f"- Average structural surprise score: {summary_stats.get('mean_surprise', 0):.2f}\n"
            f"- Talker events localized: {', '.join(summary_stats.get('talker_events', [])) or 'None'}\n"
            f"- Entity occlusion/recovery operations: {summary_stats.get('recoveries', 0)} "
            f"out of {summary_stats.get('occlusions', 0)} attempts\n"
            f"- Average planning latency: {summary_stats.get('planning_latency', 0):.2f}ms\n\n"
            "Analyze these performance statistics and write a coherent, highly detailed paragraph summarizing the session's stability and overall perception quality."
        )

        t0 = time.perf_counter()
        r = await self._call(prompt=prompt, image_base64=None, system=_PROOF_REPORT_SYSTEM, max_tokens=256)
        ms = (time.perf_counter()-t0)*1000; self._n+=1; self._ms+=ms
        
        text = (r.get("text") or "").strip()
        return text if text else "No narration could be generated for this proof report."

    @property
    def mean_latency_ms(self): return self._ms/self._n if self._n else 0.0
