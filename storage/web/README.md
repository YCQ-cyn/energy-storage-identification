# 本地储能排查服务 / Local Energy Storage Screening Service

## 启动 / Startup

Windows 下双击仓库根目录的 `start_web.bat`。启动器检查 Java 开发工具包、Python 依赖、学生模型和端口，按需编译本地服务并打开浏览器。

On Windows, double-click `start_web.bat` in the repository root. The launcher checks the Java Development Kit, Python dependencies, student model, and port, compiles the local service when needed, and opens the browser.

```text
http://127.0.0.1:8765
```

通过 `STORAGE_WEB_PORT` 指定端口，通过 `STORAGE_PYTHON` 指定 Python 可执行文件。仅检查运行环境而不启动服务：

Set `STORAGE_WEB_PORT` to choose the port, and `STORAGE_PYTHON` to choose the Python executable. To check the runtime without starting the service:

```bat
start_web.bat -Check
```

启动错误保存在 `storage/web/startup-error.log`。 / Startup errors are recorded in `storage/web/startup-error.log`.

## 输入格式 / Input format

推荐使用 Excel 工作簿，表头如下。每行包含一个用户一天的 96 个有功功率采样点，间隔 15 分钟。

Use an Excel workbook with the headers below. Each row contains 96 active-power measurements for one customer on one day, sampled every 15 minutes.

```text
数据日期  户号  0:00  0:15  0:30  ...  23:30  23:45
```

支持 `.xlsx` 和 `.csv`；旧版 `.xls` 请另存为 `.xlsx`。系统自动匹配现有日期及户号列别名，支持常见日期表示方式。日期、户号或任一负荷点无效的记录不进入模型。

Supported formats are `.xlsx` and `.csv`; convert legacy `.xls` files to `.xlsx`. Existing date and customer-number aliases and common date formats are recognized. Records with an invalid date, customer number, or load measurement are not passed to the model.

## 政策与预览 / Policy and preview

选择数据对应的当前电价政策，系统按数据月份生成相对峰、平、谷角色。推理使用有功功率及相对电价角色，不需要具体电价金额。

Select the electricity tariff policy that applies to the data. Relative peak, standard, and off-peak roles are generated using the record month. Inference uses active power and relative tariff roles; exact electricity prices are not required.

结果只在表格区域内滚动，表头保持可见。输入户号片段后点击“搜索”或按回车，显示全部包含该片段的用户，并以浅黄色突出匹配文字。清空输入后搜索可恢复全部结果。

Results scroll within the table while the header remains visible. Enter a customer-number fragment and click “搜索” (Search) or press Enter to display all matching customers, with matching fragments highlighted in pale yellow. Submit an empty search to restore all results.

## 输出 / Output

页面预览及下载工作簿包含：户号、储能情况、总天数、有效天数。

The preview and downloaded workbook contain customer number, storage assessment, total days, and valid days.

服务仅监听本机地址，上传文件和结果保存在 `storage/web/work/`，启动时清理保留期限之外的任务目录。

The service listens only on the local address. Uploaded files and results are stored in `storage/web/work/`; task directories outside the retention period are cleaned up at startup.
