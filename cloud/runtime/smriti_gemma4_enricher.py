"""
smriti_gemma4_enricher.py — Plugs Gemma 4 narration into ingestion and live tick.

Integration call sites:
  A. smriti_ingestion.py  _process_job()    — after setu_descriptions assigned
  B. service.py           living_lens_tick() — before returning response dict
"""
from __future__ import annotations
import asyncio, logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

def _attr(obj, key, default=None):
    return obj.get(key,default) if isinstance(obj,dict) else getattr(obj,key,default)

class SmetiGemma4Enricher:
    def __init__(self, mlx_provider=None):
        self._p = mlx_provider
        self._ok = mlx_provider is not None
        if not self._ok:
            logger.warning("Gemma4Enricher: no MLX provider — no-op mode")

    async def enrich_ingested_media(self, media):
        existing = _attr(media,"setu_descriptions",[])
        if not self._ok: return existing
        anchors = [a for a in _attr(media,"anchor_matches",[])
                   if isinstance(a,dict) and float(a.get("confidence",0))>=0.55]
        if not anchors: return existing
        try:
            from .gemma4_bridge import Gemma4Bridge
            bridge = Gemma4Bridge(self._mlx)
            n = await asyncio.wait_for(bridge.narrate_anchor(
                anchor_matches=anchors,
                depth_strata=_attr(media,"depth_strata",None) if isinstance(_attr(media,"depth_strata",None),dict) else None,
                setu_descriptions=existing,
                observation_id=str(_attr(media,"id","unknown")),
                image_base64=None), timeout=8.0)
            rec = {"gate":{"passes":True,"consistency_score":n.confidence,"failure_reasons":[],
                           "anchor_name":n.anchor_name,"depth_stratum":n.depth_stratum,
                           "estimated_hallucination_risk":n.hallucination_risk,
                           "uncertainty_map":[],"narrator":"gemma4"},
                   "description":{"text":n.text,"confidence":n.confidence,
                                   "anchor_basis":n.anchor_name,"depth_stratum":n.depth_stratum,
                                   "is_uncertain":n.confidence<0.5,
                                   "hallucination_risk":n.hallucination_risk,
                                   "uncertainty_map":None,"narrator":"gemma4",
                                   "latency_ms":n.latency_ms}}
            return [rec]+list(existing)
        except asyncio.TimeoutError: logger.warning("Gemma4: narration timeout")
        except Exception as e: logger.warning("Gemma4: narration error: %s",e)
        return existing

    async def live_tick_alert(self, tick_result, scene_state, entity_tracks):
        if not self._ok: return {"text":"stable","is_alert":False,"latency_ms":0.0}
        sd = scene_state.dict() if hasattr(scene_state,"dict") else (scene_state if isinstance(scene_state,dict) else {})
        tl = [t.dict() if hasattr(t,"dict") else (t if isinstance(t,dict) else {}) for t in (entity_tracks or [])]
        jt = tick_result.get("jepa_tick") or {}
        try:
            from .gemma4_bridge import Gemma4Bridge
            a = await asyncio.wait_for(Gemma4Bridge(self._mlx).proactive_alert(sd,tl,jt),timeout=5.0)
            return {"text":a.text,"is_alert":a.is_alert,"latency_ms":a.latency_ms}
        except asyncio.TimeoutError: return {"text":"stable","is_alert":False,"latency_ms":0.0}
        except Exception as e: logger.debug("Gemma4: alert error: %s",e); return {"text":"stable","is_alert":False,"latency_ms":0.0}

    async def reformulate_recall_query(self, q):
        if not self._ok: return {"setu_query":q,"depth_stratum":None,"person_filter":None,"time_hint":None,"confidence_min":0.3}
        try:
            from .gemma4_bridge import Gemma4Bridge
            r = await asyncio.wait_for(Gemma4Bridge(self._mlx).reformulate_query(q),timeout=5.0)
            return {"setu_query":r.setu_query,"depth_stratum":r.depth_stratum,
                    "person_filter":r.person_filter,"time_hint":r.time_hint,
                    "confidence_min":r.confidence_min,"gemma4_latency_ms":r.latency_ms}
        except Exception: return {"setu_query":q,"depth_stratum":None,"person_filter":None,"time_hint":None,"confidence_min":0.3}

    async def _mlx(self, prompt, image_base64=None, system=None, max_tokens=256):
        if not self._p: return {"text":"","latency_ms":0.0,"model":"none"}
        loop = asyncio.get_event_loop()
        if hasattr(self._p,"aquery"):
            return await self._p.aquery(prompt=prompt,image_base64=image_base64,system=system,max_tokens=max_tokens)
        return await loop.run_in_executor(None,lambda: self._p.query(prompt=prompt,image_base64=image_base64,system=system,max_tokens=max_tokens))
