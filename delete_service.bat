@echo off
REM your service name
set service_name=N530_FoodExportationQA

sc delete "%service_name%"

pause