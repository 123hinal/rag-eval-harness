# Returns Management

## Create a return

```
POST /v2/returns
```

Provide the original `shipment_id` and a reason code. FleetOps generates a QR-code return label the customer can scan at 60,000+ drop-off locations — no printer needed.

## Tracking returns

Return shipments emit the same tracking events as outbound shipments, plus a `return.delivered` webhook when the parcel reaches your warehouse.

## Automatic refunds

Enable automatic refunds under Settings > Returns to trigger a refund to the original payment method as soon as the `return.delivered` event fires. Refund rules can require an inspection hold of up to 7 days.
