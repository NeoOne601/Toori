# Predictor Architectures

## Stable `f_theta` Interface

`ImmersiveJEPAEngine` owns the predictor contract.

- Input: context patch tokens and mask selections derived from the live frame
- Output: predicted patch states with the same `(N, d)` shape as the context path
- Runtime entrypoint: `tick(frame) -> JEPATick`

## Swap Rules

You can replace the current 4-layer MLP with another predictor if all of the following remain true:

- EMA update still happens before predictor forward
- Energy stays `E_i = ||s_ctx_i - s_pred_i||^2`
- SIGReg remains compatible with the predictor output dimensionality
- Forecast horizons `{1,2,5}` keep working
- The public `JEPATick` field names do not change

## Candidate Directions

- transformer predictors
- recurrent predictors
- masked attention predictors
- diffusion-style latent rollouts

## Validation Requirements

Every predictor proposal should document:

- forecast error behavior across `k={1,2,5}`
- SIGReg health ranges and collapse handling
- planning latency relative to the current MLP
- any shape or masking assumptions introduced by the new architecture
