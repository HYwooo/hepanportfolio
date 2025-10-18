import baostock as bs
import pandas as pd


def get_etf_data(etf_code, start_date, end_date):
    """
    获取ETF前复权数据
    """
    # 登录系统
    lg = bs.login()
    print(f"登录结果: {lg.error_msg}")

    # 查询ETF历史K线数据（前复权）
    rs = bs.query_history_k_data_plus(
        etf_code,  # ETF代码
        "date,code,open,high,low,close,volume,amount,adjustflag",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2",  # 2：前复权
    )

    print(f"查询结果: {rs.error_msg}")

    # 获取数据并转换为DataFrame
    data_list = []
    while (rs.error_code == "0") and rs.next():
        data_list.append(rs.get_row_data())

    if data_list:
        df = pd.DataFrame(data_list, columns=rs.fields)
        # 数据类型转换
        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["date"] = pd.to_datetime(df["date"])

        # 添加一些基本信息列
        df["etf_code"] = etf_code
        df["etf_name"] = get_etf_name(etf_code)  # 获取ETF名称

        print(f"获取到{len(df)}条数据")
        return df
    else:
        print("未获取到数据")
        return pd.DataFrame()

    # 登出系统
    bs.logout()


def get_etf_name(etf_code):
    """
    根据ETF代码获取ETF名称
    """
    # ETF代码到名称的映射
    etf_names = {
        "sz.159919": "华夏沪深300ETF",
        "sh.510300": "华泰柏瑞沪深300ETF",
        "sz.159915": "易方达创业板ETF",
        "sh.510050": "华夏上证50ETF",
        "sh.510500": "南方中证500ETF",
    }
    return etf_names.get(etf_code, "未知ETF")


if __name__ == "__main__":
    # 测试获取沪深300ETF数据
    etf_code = "sz.159919"
    start_date = "2024-01-01"
    end_date = "2024-10-18"

    df = get_etf_data(etf_code, start_date, end_date)

    if not df.empty:
        print("\n前5条数据:")
        print(df.head())

        print("\n数据摘要:")
        print(df.describe())

        # 保存到CSV
        output_file = f"{etf_code.replace('.', '_')}_{start_date}_{end_date}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n数据已保存到: {output_file}")
