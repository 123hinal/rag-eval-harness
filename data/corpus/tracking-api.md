# Shipment Tracking API

## Track a shipment

```
GET /v2/track/{tracking_number}
```

Returns the latest tracking event plus the full event history for the shipment.

## Tracking statuses

Every shipment moves through a standard set of statuses:

- `label_created` — the label was printed; the carrier has not picked it up yet.
- `in_transit` — the parcel is moving through the carrier network.
- `out_for_delivery` — the parcel is on the delivery vehicle.
- `delivered` — the parcel was delivered.
- `exception` — something needs attention: address issue, customs hold, damage, or a missed delivery attempt.

## Polling vs webhooks

For low volumes you can poll the tracking endpoint, but polling is rate limited to 100 requests per second per API key. For production workloads, subscribe to tracking webhooks instead — FleetOps pushes `shipment.in_transit`, `shipment.out_for_delivery`, `shipment.delivered`, and `shipment.exception` events to your endpoint in near real time.

## Handling exceptions

When a shipment enters `exception` status, the event payload includes an `exception_code` (for example `ADDRESS_UNKNOWN` or `CUSTOMS_HOLD`) and a human-readable `exception_message`. Your integration should surface these to the customer and, where possible, offer a resolution path such as address correction.
