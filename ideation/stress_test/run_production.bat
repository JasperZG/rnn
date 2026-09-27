@echo off
cd /d C:\Users\jaspe\rnn\ideation\stress_test
C:\Users\jaspe\anaconda3\python.exe prod_driver.py --workers 12 --threads 2 >> logs\prod_driver_stdout.log 2>&1
