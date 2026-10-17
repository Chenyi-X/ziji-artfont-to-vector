"""单张验收记录入口；输出范围、缺测及诊断口径见报告。"""
from regression_req import main, ROOT

if __name__ == "__main__":
    raise SystemExit(main(default_out=ROOT / 'out/acceptance', report_name='acceptance_report'))
