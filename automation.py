from src.fofproject.plot import plot_cumulative_returns
from src.fofproject.fund import input_monthly_returns, subset_of_funds
from src.fofproject.utils import parse_month
from datetime import datetime


def presentation_data_update(month:str):

    print(f"Updating presentation data to {month}...")
    funds = input_monthly_returns(r"RETURN DATA.csv", performance_fee=0.2, management_fee=0.01)

    end_month = parse_month(month)
    year = end_month.year
    specific_month = end_month.month
    january = datetime(year, 1, 1)

    print(f"Annualized return of RDGFF: {funds['RDGFF'].annualized_return(funds['RDGFF'].inception_date, end_month)}")
    print(f"Sharpe ratio of RDGFF: {funds['RDGFF'].sharpe_ratio(funds['RDGFF'].inception_date, end_month)}")
    print(f"Sortino ratio of RDGFF: {funds['RDGFF'].sortino_ratio(funds['RDGFF'].inception_date, end_month)}")
    print(f"Cumulative return of RDGFF: {funds['RDGFF'].cumulative_return(funds['RDGFF'].inception_date, end_month)}")
    print(f"{month} return of RDGFF: {funds['RDGFF'].get_monthly_return(year, specific_month)}")    
    print(f"{month} return of MSCI China: {funds['MSCI CHINA'].get_monthly_return(year, specific_month)}")    
    print(f"{month} return of MSCI World: {funds['MSCI WORLD'].get_monthly_return(year, specific_month)}")    
    print(f"Max drawdown of RDGFF: {funds['RDGFF'].max_drawdown(funds['RDGFF'].inception_date, end_month)}")
    print(f"Volatility of RDGFF: {funds['RDGFF'].volatility(funds['RDGFF'].inception_date, end_month)}")
    print(f"Percentage of positive months of RDGFF: {funds['RDGFF'].positive_months(funds['RDGFF'].inception_date, end_month)}")
    print(f"Return in positive months of RDGFF: {funds['RDGFF'].return_in_positive_months(funds['RDGFF'].inception_date, end_month)}")
    print(f"Return in negative months of RDGFF: {funds['RDGFF'].return_in_negative_months(funds['RDGFF'].inception_date, end_month)}")
    print(f"Beta to MSCI China of RDGFF: {funds['RDGFF'].beta_to(funds['MSCI CHINA'], funds['RDGFF'].inception_date, end_month)}")
    print(f"Beta to MSCI World of RDGFF: {funds['RDGFF'].beta_to(funds['MSCI WORLD'], funds['RDGFF'].inception_date, end_month)}")
    print(f"YTD return of RDGFF: {funds['RDGFF'].cumulative_return(january, end_month)}")
    print(f"YTD return of HFRI: {funds['EUREKAHEDGE'].cumulative_return(january, end_month)}")    
    print(f"YTD return of S&P 500: {funds['S&P 500'].cumulative_return(january, end_month)}")
    print(f"YTD return of MSCI China: {funds['MSCI CHINA'].cumulative_return(january, end_month)}")
    print("========== Portfolio holdings =============")
    print("Appendix of our portfolio performance:")
    for name in ['HAO','TAIREN', 'LEXINGTON', 'LIM','FOREST']:
        print(f"{month} return of {name}: {funds[name].get_monthly_return(year, specific_month)}")    
        print(f"Annualized return of {name}: {funds[name].annualized_return(funds[name].inception_date, end_month)}")
        print(f"Volatility of {name}: {funds[name].volatility(funds[name].inception_date, end_month)}")
        print(f"Sharpe ratio of {name}: {funds[name].sharpe_ratio(funds[name].inception_date, end_month)}")
        print(f"Sortino ratio of {name}: {funds[name].sortino_ratio(funds[name].inception_date, end_month)}")

    # Define the sets to plot (first item is the primary fund)
    list_of_plots = [
        ['RDGFF', 'EUREKAHEDGE', 'MSCI CHINA'],
        ['HAO', 'MSCI CHINA'],
        ['TAIREN', 'MSCI CHINA', 'MSCI WORLD'],
        ['LEXINGTON', 'MSCI WORLD', 'S&P 500'],
        ['LIM', 'EUREKAHEDGE', 'TOPIX'],
        ['FOREST', 'EUREKAHEDGE'],
    ]

    for names in list_of_plots:
        # Optional: skip if any required series are missing
        missing = [n for n in names if n not in funds]
        if missing:
            print(f"Skipping {names}: missing {missing} in `funds`.")
            continue

        print(f"Generating cumulative return plot for {', '.join(names)}...")

        plot = plot_cumulative_returns(
            funds=subset_of_funds(funds, names),   # subset based on this group
            title="Cumulative Returns",
            start_month=None,
            end_month=month,
            style="excel",
            language="en",
        )

presentation_data_update("2025-06")