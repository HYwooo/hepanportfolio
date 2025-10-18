import baostock as bs
import pandas as pd

try:
    # 登录系统
    lg = bs.login()

    # 查询数据
    rs = bs.query_history_k_data_plus(
        "sz.159919",
        "date,code,open,high,low,close,volume,amount",
        start_date="2024-01-01",
        end_date="2024-10-18",
        frequency="d",
        adjustflag="2",
    )

    if rs.error_code == "0":
        # 处理数据...
        pass
    else:
        print(f"查询失败: {rs.error_msg}")

finally:
    # 确保登出
    bs.logout()
