# Local Energy Storage Screening Service

[简体中文](README.md) | **English**

## Startup

On Windows, double-click `start_web.bat` in the repository root. The launcher checks the Java Development Kit, Python dependencies, student model, and port, compiles the local service when needed, and opens the browser.

```text
http://127.0.0.1:8765
```

Set `STORAGE_WEB_PORT` to choose the port, and `STORAGE_PYTHON` to choose the Python executable. To check the runtime without starting the service:

```bat
start_web.bat -Check
```

Startup errors are recorded in `storage/web/startup-error.log`.

## Input format

Use an Excel workbook with the headers below. Each row contains 96 active-power measurements for one customer on one day, sampled every 15 minutes.

```text
数据日期  户号  0:00  0:15  0:30  ...  23:30  23:45
```

Supported formats are `.xlsx` and `.csv`; convert legacy `.xls` files to `.xlsx`. Existing date and customer-number aliases and common date formats are recognized. Records with an invalid date, customer number, or load measurement are not passed to the model.

## Policy and preview

Select the electricity tariff policy that applies to the data. Relative peak, standard, and off-peak roles are generated using the record month. Inference uses active power and relative tariff roles; exact electricity prices are not required.

Results scroll within the table while the header remains visible. Enter a customer-number fragment and click “搜索” (Search) or press Enter to display all matching customers, with matching fragments highlighted in pale yellow. Submit an empty search to restore all results.

## Output

The preview and downloaded workbook contain customer number, storage assessment, total days, and valid days.

The service listens only on the local address. Uploaded files and results are stored in `storage/web/work/`; task directories outside the retention period are cleaned up at startup.
