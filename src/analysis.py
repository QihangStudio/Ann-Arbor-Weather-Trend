import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular

# Read Data File
weather_url = "https://raw.githubusercontent.com/QihangStudio/Ann-Arbor-Weather-Trend/main/data.csv"
weather_ori = pd.read_csv(weather_url, header=1)


def ori_printout() -> None:
    print("===== Original Dataset =====")
    print(f"Rows: {weather_ori.shape[0]}. Columns: {weather_ori.shape[1]}")
    print()


def filtered_printout(df: pd.DataFrame) -> None:
    print("===== Dataset after 2003-04-16 =====")
    df_filtered = select_date(df)
    print(df_filtered.head())
    print()


# Data Filtering
def select_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    REQUIRES: df has a 'Date' column with strings or datetime objects.
    MODIFIES: df
    EFFECTS: Converts 'Date' to datetime and returns rows from 2003-04-16 onward.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    return df[df['Date'] >= '2003-04-16']


def get_tavg(df: pd.DataFrame) -> None:
    """
    REQUIRES: df has columns 'TMAX (Degrees Fahrenheit)' and
              'TMIN (Degrees Fahrenheit)'.
    MODIFIES: df
    EFFECTS: Adds/overwrites df['TAVG (Degrees Fahrenheit)'] with the row-wise mean.
    """
    df['TAVG (Degrees Fahrenheit)'] = df[
        ['TMAX (Degrees Fahrenheit)', 'TMIN (Degrees Fahrenheit)']
    ].mean(axis=1)


def prepare_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    REQUIRES: df has a datetime 'Date' column.
    MODIFIES: Nothing.
    EFFECTS: Returns a copy of df with time-related columns added.
    """
    df = df.copy()
    start_date = df['Date'].min()
    df['time_days'] = (df['Date'] - start_date).dt.days
    df['month'] = df['Date'].dt.month
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    return df


def prepare_linear_features(df: pd.DataFrame) -> np.ndarray:
    """
    Constructs X = [1, time_days]
    """
    ones = np.ones(len(df))
    X = np.column_stack((ones, df['time_days']))
    return X


def prepare_seasonal_features(df: pd.DataFrame) -> np.ndarray:
    """
    Constructs X = [1, time_days, sin_month, cos_month]
    """
    ones = np.ones(len(df))
    X = np.column_stack((
        ones,
        df['time_days'],
        df['sin_month'],
        df['cos_month']
    ))
    return X


def solve_normal_function(M: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve least squares using QR decomposition.
    """
    Q, R = np.linalg.qr(M)
    Qty = np.dot(Q.T, y)
    beta = solve_triangular(R, Qty)
    return beta


def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (sse / sst)


def plot_linear_results(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))

    plt.scatter(
        df['Date'],
        df['TAVG (Degrees Fahrenheit)'],
        alpha=0.3,
        label='Observed Data',
        s=10,
        color='gray'
    )

    df_sorted = df.sort_values('Date')
    plt.plot(
        df_sorted['Date'],
        df_sorted['Predicted_Linear'],
        color='blue',
        linewidth=2,
        label='Linear Fit'
    )

    plt.title('Ann Arbor Temperature Trend Analysis (Linear Model)')
    plt.xlabel('Year')
    plt.ylabel('Temperature (°F)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_seasonal_results(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))

    plt.scatter(
        df['Date'],
        df['TAVG (Degrees Fahrenheit)'],
        alpha=0.3,
        label='Observed Data',
        s=10,
        color='gray'
    )

    df_sorted = df.sort_values('Date')
    plt.plot(
        df_sorted['Date'],
        df_sorted['Predicted_Seasonal'],
        color='red',
        linewidth=2,
        label='Linear + Seasonal Fit'
    )

    plt.title('Ann Arbor Temperature Trend Analysis (Sinusoidal Model)')
    plt.xlabel('Year')
    plt.ylabel('Temperature (°F)')
    plt.legend()
    plt.grid(True)
    plt.show()


def main() -> None:
    ori_printout()
    get_tavg(weather_ori)
    filtered_printout(weather_ori)

    df_filtered = select_date(weather_ori)
    df_filtered = prepare_time(df_filtered)

    y = df_filtered['TAVG (Degrees Fahrenheit)'].values

    # -------------------------
    # 1. Linear model
    # TAVG = a0 + a1 * time_days
    # -------------------------
    X_linear = prepare_linear_features(df_filtered)
    beta_linear = solve_normal_function(X_linear, y)
    y_hat_linear = np.dot(X_linear, beta_linear)
    df_filtered['Predicted_Linear'] = y_hat_linear
    r2_linear = calculate_r_squared(y, y_hat_linear)

    print("===== Linear Model =====")
    print(f"Intercept: {beta_linear[0]}")
    print(f"Time coefficient: {beta_linear[1]}")
    print(f"R-squared: {r2_linear:.4f}")
    print()

    # -------------------------
    # 2. Sinusoidal
    # TAVG = b0 + b1*time_days + b2*sin_month + b3*cos_month
    # -------------------------
    X_seasonal = prepare_seasonal_features(df_filtered)
    beta_seasonal = solve_normal_function(X_seasonal, y)
    y_hat_seasonal = np.dot(X_seasonal, beta_seasonal)
    df_filtered['Predicted_Seasonal'] = y_hat_seasonal
    r2_seasonal = calculate_r_squared(y, y_hat_seasonal)

    print("===== Sinusoidal Model =====")
    print(f"Intercept: {beta_seasonal[0]}")
    print(f"Time coefficient: {beta_seasonal[1]}")
    print(f"Sin coefficient: {beta_seasonal[2]}")
    print(f"Cos coefficient: {beta_seasonal[3]}")
    print(f"R-squared: {r2_seasonal:.4f}")
    print()

    # Plots
    plot_linear_results(df_filtered)
    plot_seasonal_results(df_filtered)


if __name__ == '__main__':
    main()
