# Shipping Labels

## Create a label

```
POST /v2/labels
```

Pass a `shipment_id` or a full shipment payload plus a `format`. Supported formats are `PDF`, `PNG`, and `ZPL` (for thermal printers). The default paper size is 4x6 inches.

## Batch labels

Up to 100 labels can be created in a single batch request. The response includes a single merged PDF or a ZIP archive of individual files, depending on the `batch_output` parameter.

## Test mode

In test mode (`sk_test_...` keys), labels are watermarked SAMPLE and never produce real postage. Switch to live keys to print shippable labels.
