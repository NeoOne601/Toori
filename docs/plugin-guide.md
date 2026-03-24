# Plugin Guide

## Plugin Model

Toori is consumed as a runtime, not as an embedded UI dependency. Host applications should call the local or remote HTTP API and optionally subscribe to the WebSocket event stream.

## Main Endpoints

- `POST /v1/analyze`
- `POST /v1/query`
- `GET /v1/settings`
- `PUT /v1/settings`
- `GET /v1/providers/health`
- `GET /v1/observations`
- `WS /v1/events`

## Suggested Integration Pattern

1. Host app captures or selects an image.
2. Host app sends the image to `POST /v1/analyze`.
3. Host app renders:
   - the stored observation id
   - nearest prior hits
   - the optional answer
4. Host app subscribes to `WS /v1/events` if it needs live updates.

## SDKs

- [sdk/python/toori_client.py](/Users/macuser/toori/sdk/python/toori_client.py)
- [sdk/typescript/index.ts](/Users/macuser/toori/sdk/typescript/index.ts)
- [sdk/swift/TooriClient.swift](/Users/macuser/toori/sdk/swift/TooriClient.swift)
- [sdk/kotlin/TooriClient.kt](/Users/macuser/toori/sdk/kotlin/TooriClient.kt)

## Example Request

```json
{
  "image_base64": "<base64>",
  "session_id": "desktop-live",
  "query": "What objects are visible on the desk?",
  "decode_mode": "auto",
  "top_k": 5
}
```

## Event Stream

The WebSocket emits JSON event messages with:

- `type`
- `timestamp`
- `payload`

Important event types:

- `observation.created`
- `answer.ready`
- `search.ready`
- `provider.changed`

## Security

- Default auth mode is `loopback`.
- For embedded remote deployments, switch `auth_mode` to `api-key` and set a matching API key.
