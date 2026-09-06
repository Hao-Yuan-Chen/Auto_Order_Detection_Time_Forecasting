# Data inputs

Place `AOTA_DE_PROD.csv` in this directory. Required columns include `datehour`, `detection_time`, `order_time`, `actual_orders`, and `no_of_items`. The notebook parses timestamps as `%d/%m/%Y %H:%M` and assumes chronological hourly observations. The legacy date-specific row removal and feature processing expect the original dataset; matching these five names alone does not guarantee compatibility.

Source datasets are not included. This directory ignores local data files by default. Use only data you are authorized to access and share.
