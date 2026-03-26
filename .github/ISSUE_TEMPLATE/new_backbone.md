---
name: New backbone or predictor
about: Propose a new perception backend or predictor architecture
title: "[backbone] "
labels: [backend, architecture]
---

## Surface

Describe which surface this affects:

- perception backend
- predictor architecture
- runtime fallback
- SDK contract

## Stable Interface

List the contract that must stay stable:

- input type
- output schema
- health signal
- fallback path
- latency target

## Proposal

Describe the backend or architecture change in plain language.

## Consumer Impact

Explain what existing callers will see change, if anything.

## Validation

Describe how you will test it:

- health check
- runtime tick
- observation storage
- world-state output
- fallback behavior

## Rollout

State whether the change is:

- additive
- a replacement
- gated behind configuration

## Notes

If the change affects UI proof copy or assets, call out whether a CC-BY-SA notice applies to that surface.

